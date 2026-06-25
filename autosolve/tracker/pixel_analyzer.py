# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
PixelAnalyzer — Texture richness analysis using raw Blender pixel data.

Uses ONLY bpy + numpy (both bundled with Blender) to:
  - Read raw RGBA frames from a MovieClip via Blender's internal pixel buffer
  - Score per-patch texture richness (local variance + gradient magnitude)
  - Build a heatmap of "trackable" zones before markers are placed
  - Score existing marker positions for pre-rejection before tracking

This is the "zero new dependencies" component.
"""

import bpy
import numpy as np
from typing import List, Tuple, Optional, Dict


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# Patch size for individual marker scoring (pixels)
MARKER_PATCH_SIZE = 32

# Grid density for heatmap generation (number of cells per axis)
HEATMAP_GRID = 8

# Minimum texture variance to consider a patch trackable
MIN_TEXTURE_VARIANCE = 0.003

# Weight between variance and gradient magnitude in the combined score
VARIANCE_WEIGHT = 0.6
GRADIENT_WEIGHT = 0.4

# Low-resolution analysis target (resize to this width for speed)
ANALYSIS_WIDTH = 320


# ═══════════════════════════════════════════════════════════════════════════
# PIXEL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class PixelAnalyzer:
    """
    Reads raw pixel data from Blender MovieClips and computes texture
    richness scores to identify trackable regions.

    All computation is pure numpy — no external dependencies required.
    """

    def __init__(self):
        self._frame_cache: Dict[Tuple[str, int], np.ndarray] = {}
        self._cache_max = 5  # Keep at most 5 frames in memory

    # ───────────────────────────────────────────────────────────────────────
    # Frame Access
    # ───────────────────────────────────────────────────────────────────────

    def get_frame_pixels(
        self, clip: bpy.types.MovieClip, frame: int
    ) -> Optional[np.ndarray]:
        """
        Read a frame from the clip as a (H, W, 4) RGBA float32 numpy array.

        Uses Blender's image buffer API. Returns None if the frame cannot
        be read (e.g., image sequence gap, unsupported format).
        """
        cache_key = (clip.name, frame)
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        try:
            # Set the current frame so Blender loads it
            current_frame = bpy.context.scene.frame_current
            scene_frame = frame + clip.frame_start - 1
            bpy.context.scene.frame_set(scene_frame)

            w, h = clip.size
            if w == 0 or h == 0:
                return None

            arr = None
            filepath = bpy.path.abspath(clip.filepath)

            # Option 1: Try OpenCV first (best compatibility, handles video/sequences fast)
            try:
                import cv2
                cap = cv2.VideoCapture(filepath)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)  # cv2 is 0-indexed
                    ret, frame_bgr = cap.read()
                    cap.release()
                    if ret:
                        # Convert BGR to RGB and normalize
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        h_f, w_f, _ = frame_rgb.shape
                        
                        # Resize to clip.size if necessary
                        if (w_f, h_f) != (w, h):
                            frame_rgb = cv2.resize(frame_rgb, (w, h))
                            h_f, w_f = h, w

                        # Create float32 RGBA array
                        arr = np.ones((h_f, w_f, 4), dtype=np.float32)
                        arr[:, :, :3] = frame_rgb.astype(np.float32) / 255.0
            except Exception:
                pass

            # Option 2: Image Sequence fallback (native Blender load without OpenCV)
            if arr is None and clip.source == 'SEQUENCE':
                import re
                import os
                
                # Resolve sequence filepath pattern
                seq_path = filepath
                if '#' in seq_path:
                    match = re.search(r'#+', seq_path)
                    if match:
                        hashes = match.group(0)
                        padded_frame = str(scene_frame).zfill(len(hashes))
                        seq_path = seq_path.replace(hashes, padded_frame)
                elif '%' in seq_path:
                    try:
                        seq_path = seq_path % scene_frame
                    except Exception:
                        pass
                else:
                    # Trailing digits fallback
                    base, ext = os.path.splitext(seq_path)
                    match = re.search(r'(\d+)$', base)
                    if match:
                        digits = match.group(1)
                        padded_frame = str(scene_frame).zfill(len(digits))
                        seq_path = base[:-len(digits)] + padded_frame + ext
                
                if os.path.exists(seq_path):
                    try:
                        # Load single frame as Blender Image
                        img = bpy.data.images.load(seq_path)
                        # Read pixels (numpy array)
                        pixels_flat = np.array(img.pixels, dtype=np.float32)
                        arr = pixels_flat.reshape(h, w, 4)
                        # Clean up loaded image to prevent memory leaks
                        img.user_clear()
                        bpy.data.images.remove(img)
                    except Exception as img_err:
                        print(f"AutoSolve PixelAnalyzer: Image sequence load failed for {seq_path}: {img_err}")

            # Option 3: Last resort fallback (try direct image load of the raw filepath)
            if arr is None and os.path.exists(filepath):
                try:
                    img = bpy.data.images.load(filepath)
                    pixels_flat = np.array(img.pixels, dtype=np.float32)
                    arr = pixels_flat.reshape(h, w, 4)
                    img.user_clear()
                    bpy.data.images.remove(img)
                except Exception:
                    pass

            # Restore original frame
            bpy.context.scene.frame_set(current_frame)

            if arr is None:
                print(f"AutoSolve PixelAnalyzer: Could not extract pixels for frame {frame} of clip '{clip.name}'")
                return None

            # Cache and return
            self._evict_cache_if_needed()
            self._frame_cache[cache_key] = arr
            return arr

        except Exception as e:
            print(f"AutoSolve PixelAnalyzer: Failed to read frame {frame} from {clip.name}: {e}")
            try:
                bpy.context.scene.frame_set(current_frame)
            except Exception:
                pass
            return None

    def get_frame_gray(
        self, clip: bpy.types.MovieClip, frame: int
    ) -> Optional[np.ndarray]:
        """
        Return a grayscale (H, W) float32 array for a clip frame.
        Converts from RGBA using standard luminance weights.
        """
        rgba = self.get_frame_pixels(clip, frame)
        if rgba is None:
            return None
        # Luminance: 0.299R + 0.587G + 0.114B
        return (rgba[:, :, 0] * 0.299 +
                rgba[:, :, 1] * 0.587 +
                rgba[:, :, 2] * 0.114).astype(np.float32)

    # ───────────────────────────────────────────────────────────────────────
    # Texture Scoring
    # ───────────────────────────────────────────────────────────────────────

    def is_patch_rigid(
        self,
        clip: bpy.types.MovieClip,
        frame: int,
        cx: float, cy: float,
        gray_f: Optional[np.ndarray] = None
    ) -> bool:
        """
        Verify if a patch centered at normalized (cx, cy) is rigid/static.
        Uses patch_rigidity.onnx model if available, falling back to a NumPy
        temporal and texture variance analyzer.
        """
        try:
            from .onnx_predictor import OnnxPredictor
            opt = OnnxPredictor.get_instance()

            if gray_f is None:
                gray_f = self.get_frame_gray(clip, frame)
            if gray_f is None:
                return True  # neutral fallback

            h, w = gray_f.shape
            px = int(cx * w)
            py = int(cy * h)

            x0 = max(0, px - 16)
            x1 = min(w, px + 16)
            y0 = max(0, py - 16)
            y1 = min(h, py + 16)

            # Pad boundary patches safely to 32x32
            if (x1 - x0) < 32 or (y1 - y0) < 32:
                patch = np.zeros((32, 32), dtype=np.float32)
                crop = gray_f[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
                ch, cw = crop.shape
                patch[:ch, :cw] = crop
            else:
                patch = gray_f[y0:y1, x0:x1]

            # 1. Try ONNX rigidity classifier prediction (CNN-based)
            if opt.patch_model_available:
                score = opt.predict_patch_rigidity(patch)
                if score is not None:
                    return score > 0.45

            # 2. NumPy Fallback: Texture variance + Simple Temporal variance checks
            p_var = float(np.var(patch))
            if p_var < MIN_TEXTURE_VARIANCE:
                return False  # uniform homogeneous (e.g. flat sky, untrackable)

            # Sample adjacent frames to check temporal consistency (water/foliage)
            prev_frame = max(1, frame - 3)
            next_frame = min(clip.frame_duration, frame + 3)

            gray_prev = self.get_frame_gray(clip, prev_frame)
            gray_next = self.get_frame_gray(clip, next_frame)

            if gray_prev is not None and gray_next is not None:
                p_prev = gray_prev[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
                p_next = gray_next[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]

                if p_prev.shape == patch.shape and p_next.shape == patch.shape:
                    var_prev = float(np.var(p_prev))
                    var_next = float(np.var(p_next))
                    
                    var_drift = float(np.std([var_prev, p_var, var_next]))
                    if var_drift > 0.015:  # High texture volatility indicates foliage/water
                        return False

            return True
        except Exception as e:
            print(f"AutoSolve: Rigidity analysis error: {e}")
            return True

    def score_patch(
        self,
        gray: np.ndarray,
        cx: float, cy: float,
        patch_size: int = MARKER_PATCH_SIZE,
        clip: Optional[bpy.types.MovieClip] = None,
        frame: Optional[int] = None
    ) -> float:
        """
        Score the textural richness and rigidity of a patch around pixel (cx, cy).

        Args:
            gray:       Grayscale (H, W) float32 array (values in [0,1])
            cx, cy:     Patch center in normalized [0,1] image coordinates
            patch_size: Patch side length in pixels
            clip:       Optional MovieClip for temporal rigidity checks
            frame:      Optional frame number for temporal rigidity checks

        Returns:
            Score in [0, 1] where 1 = very trackable & rigid, 0 = flat/dynamic
        """
        h, w = gray.shape
        half = patch_size // 2

        # Convert normalized coords to pixel coords
        px = int(cx * w)
        py = int(cy * h)

        # Clamp to valid range
        x0 = max(0, px - half)
        x1 = min(w, px + half)
        y0 = max(0, py - half)
        y1 = min(h, py + half)

        if (x1 - x0) < 4 or (y1 - y0) < 4:
            return 0.0

        patch = gray[y0:y1, x0:x1]

        # 1. Local variance (captures texture richness)
        variance = float(np.var(patch))

        # 2. Gradient magnitude (captures edge/corner sharpness)
        if patch.shape[0] > 1 and patch.shape[1] > 1:
            grad_x = np.diff(patch, axis=1)
            grad_y = np.diff(patch, axis=0)
            # Align shapes
            min_rows = min(grad_x.shape[0], grad_y.shape[0])
            min_cols = min(grad_x.shape[1], grad_y.shape[1])
            grad_mag = np.sqrt(
                grad_x[:min_rows, :min_cols] ** 2 +
                grad_y[:min_rows, :min_cols] ** 2
            )
            gradient_mean = float(np.mean(grad_mag))
        else:
            gradient_mean = 0.0

        # Normalize: typical max variance ~0.1, max gradient ~0.5
        norm_var = min(1.0, variance / 0.08)
        norm_grad = min(1.0, gradient_mean / 0.3)

        score = VARIANCE_WEIGHT * norm_var + GRADIENT_WEIGHT * norm_grad
        score = float(np.clip(score, 0.0, 1.0))

        # 3. Apply Rigidity/Semantic Masking Filter if context is provided
        if clip is not None and frame is not None:
            # Flip y back for is_patch_rigid which assumes Blender Y-up flipped coords
            cy_flipped = 1.0 - cy
            if not self.is_patch_rigid(clip, frame, cx, cy_flipped, gray_f=gray):
                score *= 0.05  # heavily penalize non-rigid/dynamic regions

        return float(np.clip(score, 0.0, 1.0))

    def score_marker_position(
        self,
        clip: bpy.types.MovieClip,
        track: bpy.types.MovieTrackingTrack,
        frame: int,
        patch_size: int = MARKER_PATCH_SIZE
    ) -> float:
        """
        Score the texture richness at a marker's position in a given frame.

        Returns:
            float in [0, 1] — 0 = untraceable, 1 = excellent texture
            Returns 0.5 (neutral) if pixels cannot be read.
        """
        gray = self.get_frame_gray(clip, frame)
        if gray is None:
            return 0.5  # neutral fallback

        # Get marker position for this frame
        marker = track.markers.find_frame(frame)
        if marker is None or marker.mute:
            return 0.0

        cx, cy = marker.co  # normalized [0,1] in Blender's Y-up coord system
        # Blender uses bottom-left origin; numpy uses top-left — flip Y
        cy_flipped = 1.0 - cy

        return self.score_patch(gray, cx, cy_flipped, patch_size, clip, frame)

    # ───────────────────────────────────────────────────────────────────────
    # Heatmap Generation
    # ───────────────────────────────────────────────────────────────────────

    def build_trackability_heatmap(
        self,
        clip: bpy.types.MovieClip,
        frame: int,
        grid_size: int = HEATMAP_GRID
    ) -> Optional[np.ndarray]:
        """
        Compute a (grid_size x grid_size) trackability heatmap for a frame.

        Each cell contains a score in [0,1] representing how trackable that
        region of the image is.

        Args:
            clip:       The MovieClip to analyze
            frame:      Frame number to analyze
            grid_size:  Number of cells per axis (e.g., 8 → 8×8 = 64 cells)

        Returns:
            numpy array of shape (grid_size, grid_size) or None on failure
        """
        gray = self.get_frame_gray(clip, frame)
        if gray is None:
            return None

        heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
        cell_patch = max(16, min(gray.shape[0], gray.shape[1]) // grid_size)

        for row in range(grid_size):
            for col in range(grid_size):
                # Center of this cell in normalized coords
                cx = (col + 0.5) / grid_size
                cy = (row + 0.5) / grid_size
                heatmap[row, col] = self.score_patch(gray, cx, cy, cell_patch, clip, frame)

        return heatmap

    def get_best_placement_zones(
        self,
        clip: bpy.types.MovieClip,
        frame: int,
        top_n: int = 5,
        grid_size: int = HEATMAP_GRID
    ) -> List[Tuple[float, float, float]]:
        """
        Return the top N placement zones as (cx, cy, score) tuples
        in normalized image coordinates (0→1).

        Results are sorted highest score first.
        """
        heatmap = self.build_trackability_heatmap(clip, frame, grid_size)
        if heatmap is None:
            return []

        # Flatten, sort, take top N
        flat_indices = np.argsort(heatmap.ravel())[::-1][:top_n]
        results = []
        for idx in flat_indices:
            row = idx // grid_size
            col = idx % grid_size
            cx = (col + 0.5) / grid_size
            cy = (row + 0.5) / grid_size
            score = float(heatmap[row, col])
            results.append((cx, cy, score))

        return results

    # ───────────────────────────────────────────────────────────────────────
    # Batch Scoring
    # ───────────────────────────────────────────────────────────────────────

    def score_all_markers(
        self,
        clip: bpy.types.MovieClip,
        tracks: List[bpy.types.MovieTrackingTrack],
        frame: int
    ) -> Dict[str, float]:
        """
        Score all markers in the provided track list at a given frame.

        Returns:
            dict mapping track.name → texture score [0, 1]
        """
        gray = self.get_frame_gray(clip, frame)
        if gray is None:
            return {t.name: 0.5 for t in tracks}

        scores = {}
        for track in tracks:
            marker = track.markers.find_frame(frame)
            if marker is None or marker.mute:
                scores[track.name] = 0.0
                continue

            cx, cy = marker.co
            cy_flipped = 1.0 - cy
            scores[track.name] = self.score_patch(gray, cx, cy_flipped, clip=clip, frame=frame)

        return scores

    def identify_weak_markers(
        self,
        clip: bpy.types.MovieClip,
        tracks: List[bpy.types.MovieTrackingTrack],
        frame: int,
        threshold: float = 0.15
    ) -> List[str]:
        """
        Return names of tracks whose marker texture score is below `threshold`.
        These are candidates for pre-rejection or repositioning.
        """
        scores = self.score_all_markers(clip, tracks, frame)
        return [name for name, score in scores.items() if score < threshold]

    # ───────────────────────────────────────────────────────────────────────
    # Motion Estimation from Pixel Differences
    # ───────────────────────────────────────────────────────────────────────

    def estimate_inter_frame_motion(
        self,
        clip: bpy.types.MovieClip,
        frame_a: int,
        frame_b: int
    ) -> float:
        """
        Estimate motion between two frames using mean absolute pixel difference.

        Returns a normalized motion score [0, 1]:
          - 0.0 = static scene (no change)
          - 1.0 = completely different frames (very high motion)

        This is used as a lightweight motion probe without needing the tracker.
        """
        gray_a = self.get_frame_gray(clip, frame_a)
        gray_b = self.get_frame_gray(clip, frame_b)

        if gray_a is None or gray_b is None:
            return 0.0

        if gray_a.shape != gray_b.shape:
            return 0.0

        diff = np.abs(gray_a.astype(np.float32) - gray_b.astype(np.float32))
        return float(np.clip(np.mean(diff) * 5.0, 0.0, 1.0))

    def quick_motion_profile(
        self,
        clip: bpy.types.MovieClip,
        start: int,
        end: int,
        samples: int = 5
    ) -> Dict[str, float]:
        """
        Sample `samples` evenly-spaced frame pairs across [start, end]
        and compute inter-frame motion scores.

        Returns:
            dict with keys: 'mean_motion', 'max_motion', 'motion_class'
            where 'motion_class' is 'HIGH', 'MEDIUM', or 'LOW'
        """
        if end <= start or samples < 2:
            return {'mean_motion': 0.0, 'max_motion': 0.0, 'motion_class': 'LOW'}

        step = max(1, (end - start) // samples)
        frames = list(range(start, end, step))[:samples]

        motions = []
        for i in range(len(frames) - 1):
            m = self.estimate_inter_frame_motion(clip, frames[i], frames[i + 1])
            motions.append(m)

        if not motions:
            return {'mean_motion': 0.0, 'max_motion': 0.0, 'motion_class': 'LOW'}

        mean_m = float(np.mean(motions))
        max_m = float(np.max(motions))

        if mean_m > 0.25:
            cls = 'HIGH'
        elif mean_m > 0.08:
            cls = 'MEDIUM'
        else:
            cls = 'LOW'

        return {
            'mean_motion': mean_m,
            'max_motion': max_m,
            'motion_class': cls
        }

    # ───────────────────────────────────────────────────────────────────────
    # Cache Management
    # ───────────────────────────────────────────────────────────────────────

    def _evict_cache_if_needed(self):
        """Remove oldest cached frame if cache is full."""
        if len(self._frame_cache) >= self._cache_max:
            oldest_key = next(iter(self._frame_cache))
            del self._frame_cache[oldest_key]

    def clear_cache(self):
        """Clear the pixel frame cache."""
        self._frame_cache.clear()
