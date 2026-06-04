# SPDX-FileCopyrightText: 2026 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve ML Trajectory Extractor using CoTracker & PyTorch (with OpenCV fallback).
Extracts multi-frame tracking coordinates directly from raw video files,
injects synthetic noise and occlusions to create positive/negative survival labels,
and exports JSON solve samples compatible with the training pipeline.
"""

import os
import sys
import json
import argparse
import random
import time
import math
from typing import List, Tuple, Dict, Any

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import schemas if possible
try:
    from ml.schema import ClipMetadata, TrackingSettings, TrackSample, SolveSample, solve_sample_to_dict
    HAS_SCHEMA = True
except ImportError:
    HAS_SCHEMA = False


def classify_region(fx: float, fy: float) -> str:
    """Classify 2D normalized coordinate into one of 9 screen regions."""
    ry = "top" if fy > 0.66 else ("bottom" if fy < 0.33 else "mid")
    rx = "left" if fx < 0.33 else ("right" if fx > 0.66 else "center")
    return "center" if (ry == "mid" and rx == "center") else (f"{ry}-{rx}" if ry != "mid" else f"mid-{rx}")


def run_opencv_tracking(video_path: str, grid_size: int = 8) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
    """
    Fallback tracking using OpenCV Shi-Tomasi corners and Lucas-Kanade optical flow.
    Returns tracking trajectories normalized to [0, 1] and video metadata.
    """
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV is required but not installed.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 24.0

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Could not read first frame of: {video_path}")

    gray_prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Detect Shi-Tomasi corners to initialize tracks
    max_corners = grid_size * grid_size
    pts = cv2.goodFeaturesToTrack(
        gray_prev, 
        maxCorners=max_corners, 
        qualityLevel=0.01, 
        minDistance=20
    )

    if pts is None:
        # Create a grid fallback if no corners are found
        pts = []
        for y in np.linspace(height * 0.1, height * 0.9, grid_size):
            for x in np.linspace(width * 0.1, width * 0.9, grid_size):
                pts.append([[x, y]])
        pts = np.array(pts, dtype=np.float32)

    num_tracks = len(pts)
    trajectories = [[] for _ in range(num_tracks)]
    active_mask = np.ones(num_tracks, dtype=bool)

    # Add initial points
    for idx, p in enumerate(pts):
        x, y = p[0]
        trajectories[idx].append((float(x / width), float(y / height)))

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    p_prev = pts.copy()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track with LK Optical Flow
        p_curr, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, p_prev, None, **lk_params)

        for idx in range(num_tracks):
            if not active_mask[idx]:
                continue

            # If tracking failed or point went out of bounds
            if status[idx][0] == 0:
                active_mask[idx] = False
                continue

            x, y = p_curr[idx][0]
            if x < 0 or x >= width or y < 0 or y >= height:
                active_mask[idx] = False
                continue

            trajectories[idx].append((float(x / width), float(y / height)))

        p_prev = p_curr
        gray_prev = gray_curr

    cap.release()

    meta = {
        "clip_name": os.path.splitext(os.path.basename(video_path))[0],
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count
    }

    return trajectories, meta


def run_cotracker_tracking(video_path: str, grid_size: int = 8) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
    """
    Natively track points using CoTracker (Meta AI) loaded via torch.hub.
    Returns normalized trajectories and clip metadata.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for CoTracker but not installed.")

    # 1. Read video metadata and frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 24.0

    frames = []
    # Read and resize frames to a smaller resolution for CoTracker processing to fit in typical VRAM
    target_h, target_w = 288, 384
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (target_w, target_h))
        frames.append(frame_resized)

    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to read any frames from: {video_path}")

    # Convert to torch tensor [1, T, 3, H, W]
    video_tensor = torch.from_numpy(np.stack(frames, axis=0)).float() # [T, H, W, 3]
    video_tensor = video_tensor.permute(0, 3, 1, 2) # [T, 3, H, W]
    video_tensor = video_tensor.unsqueeze(0) # [1, T, 3, H, W]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_tensor = video_tensor.to(device)

    print(f"Loading CoTracker model from torch.hub on device: {device}...")
    # Clean cache directory if needed, load from hub
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    model = model.to(device)
    model.eval()

    print(f"Running CoTracker inference (grid size {grid_size}x{grid_size})...")
    with torch.no_grad():
        # pred_tracks: [1, T, N, 2], pred_visibility: [1, T, N]
        pred_tracks, pred_visibility = model(video_tensor, grid_size=grid_size)

    # Convert tensors back to CPU numpy
    pred_tracks = pred_tracks.cpu().numpy()[0] # [T, N, 2]
    pred_visibility = pred_visibility.cpu().numpy()[0] # [T, N]

    num_tracks = pred_tracks.shape[1]
    trajectories = [[] for _ in range(num_tracks)]

    for t_idx in range(num_tracks):
        for f_idx in range(len(frames)):
            # If point is visible at this frame, save normalized coords
            if pred_visibility[f_idx, t_idx]:
                px, py = pred_tracks[f_idx, t_idx]
                # Normalize using the target_h and target_w shape of the resized video
                nx = float(px / target_w)
                ny = float(py / target_h)
                # Keep coordinates clipped to [0, 1]
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                trajectories[t_idx].append((nx, ny))
            else:
                # Truncate trajectory if it becomes invisible
                break

    meta = {
        "clip_name": os.path.splitext(os.path.basename(video_path))[0],
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count
    }

    return trajectories, meta


def simulate_variation(
    base_trajectories: List[List[Tuple[float, float]]],
    meta: Dict[str, Any],
    quality: str,
    robust: bool,
    tripod: bool
) -> Dict[str, Any]:
    """
    Simulates tracking settings variations on top of the base video trajectories.
    Injects synthetic noise, occlusions/failures, and computes velocities/jitter.
    """
    random.seed(hash(f"{meta['clip_name']}_{quality}_{robust}_{tripod}") % 1234567)

    # Adjust parameters based on tracking preset
    # Higher quality = less noise, longer lifespan
    if quality == "QUALITY":
        noise_std = 0.001
        survival_base_prob = 0.85
    elif quality == "FAST":
        noise_std = 0.005
        survival_base_prob = 0.50
    else:  # BALANCED
        noise_std = 0.002
        survival_base_prob = 0.70

    if robust:
        survival_base_prob += 0.15
        noise_std *= 0.8

    survival_base_prob = min(0.95, max(0.20, survival_base_prob))

    track_samples = []
    survived_count = 0

    for idx, base_coords in enumerate(base_trajectories):
        if len(base_coords) < 6:
            continue

        fx, fy = base_coords[0]
        region = classify_region(fx, fy)

        # Region-based survival multiplier
        # Center region has higher survival, corners/sides have lower
        region_survival_mult = 1.0
        if "top" in region:
            region_survival_mult *= 0.7
        if "left" in region or "right" in region:
            region_survival_mult *= 0.85
        if "center" in region:
            region_survival_mult *= 1.1

        survival_prob = min(0.98, survival_base_prob * region_survival_mult)
        survived = random.random() < survival_prob
        has_bundle = survived and (random.random() < 0.90)  # Bundling succeeds on 90% of surviving tracks

        # Construct coordinate path
        coords = []
        if survived:
            # Keep full path, add normal tracking noise
            for x, y in base_coords:
                nx = x + random.normalvariate(0.0, noise_std)
                ny = y + random.normalvariate(0.0, noise_std)
                coords.append((max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))))
            survived_count += 1
            average_error = random.uniform(0.15, 0.45)
        else:
            # Simulate failure: truncate track at a random failure frame
            fail_frame = random.randint(5, len(base_coords) - 1)
            
            # Optionally simulate a tracking slip/drift just before failure
            slip_frames = min(5, fail_frame)
            for f_idx in range(fail_frame):
                x, y = base_coords[f_idx]
                if f_idx >= fail_frame - slip_frames:
                    # Inject drifting/slipping noise
                    drift_factor = (f_idx - (fail_frame - slip_frames) + 1) * 0.01
                    nx = x + random.normalvariate(0.0, noise_std + drift_factor)
                    ny = y + random.normalvariate(0.0, noise_std + drift_factor)
                else:
                    nx = x + random.normalvariate(0.0, noise_std)
                    ny = y + random.normalvariate(0.0, noise_std)
                coords.append((max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))))
            
            average_error = random.uniform(0.8, 5.0)

        # Calculate velocities
        velocities = []
        for i in range(1, len(coords)):
            velocities.append((coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1]))

        # Calculate jitter (difference in velocities)
        jitter_scores = []
        for i in range(1, len(velocities)):
            dv_x = velocities[i][0] - velocities[i-1][0]
            dv_y = velocities[i][1] - velocities[i-1][1]
            jitter_scores.append(math.sqrt(dv_x**2 + dv_y**2))

        # Build TrackSample dictionary matching the schema
        track_sample = {
            "track_name": f"track_{idx:03d}",
            "region": region,
            "positions": coords,
            "velocities": velocities,
            "jitter_scores": jitter_scores,
            "lifespan": len(coords),
            "survived": survived,
            "has_bundle": has_bundle,
            "average_error": average_error
        }
        track_samples.append(track_sample)

    # Solve metadata simulation
    total_tracks = len(track_samples)
    bundle_ratio = 0.0
    solve_success = False
    solve_error = 99.0

    if total_tracks > 0:
        bundle_count = sum(1 for t in track_samples if t["has_bundle"])
        bundle_ratio = bundle_count / total_tracks
        
        # In tracking, if bundle ratio > 0.40 and we have enough points, solve succeeds
        if bundle_count >= 8 and bundle_ratio >= 0.40:
            solve_success = True
            # Solve error depends on noise and tripod settings
            solve_error = max(0.1, random.normalvariate(0.4 + noise_std * 100, 0.15))
            if tripod:
                solve_error *= 0.8  # Tripod solves usually have lower error

    # Mock settings matching TrackingSettings dataclass
    settings_dict = {
        "quality_preset": quality,
        "footage_type": "AUTO",
        "robust_mode": robust,
        "tripod_mode": tripod,
        "pattern_size": 11 if quality == "FAST" else (17 if quality == "BALANCED" else 31),
        "search_size": 51 if quality == "FAST" else (71 if quality == "BALANCED" else 121),
        "correlation": 0.55 if quality == "FAST" else (0.70 if quality == "BALANCED" else 0.85),
        "threshold": 0.40 if quality == "FAST" else (0.30 if quality == "BALANCED" else 0.15),
        "motion_model": "LocRot"
    }

    solve_sample = {
        "clip_metadata": meta,
        "settings": settings_dict,
        "tracks": track_samples,
        "solve_success": solve_success,
        "solve_error": solve_error,
        "bundle_count": sum(1 for t in track_samples if t["has_bundle"]),
        "bundle_ratio": bundle_ratio,
        "runtime_seconds": random.uniform(2.0, 15.0)
    }

    return solve_sample


def process_video(video_path: str, args: argparse.Namespace):
    """Run tracking model on the video and generate all 5 settings variations."""
    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nProcessing clip: {clip_name}")

    # Step 1: Perform Tracking to get base trajectories
    trajectories = None
    meta = None
    
    use_cotracker = TORCH_AVAILABLE and not args.no_cotracker
    
    if use_cotracker:
        try:
            trajectories, meta = run_cotracker_tracking(video_path, grid_size=args.grid_size)
            print(f"Successfully extracted {len(trajectories)} trajectories using CoTracker.")
        except Exception as e:
            print(f"CoTracker tracking failed: {e}. Falling back to OpenCV tracking...")
            use_cotracker = False

    if not use_cotracker:
        try:
            trajectories, meta = run_opencv_tracking(video_path, grid_size=args.grid_size)
            print(f"Successfully extracted {len(trajectories)} trajectories using OpenCV LK tracker.")
        except Exception as e:
            print(f"Error: Tracking failed on {video_path}: {e}")
            return

    if not trajectories:
        print(f"No trajectories extracted for clip: {clip_name}")
        return

    # Define standard settings variations to generate
    variations = [
        # (quality, robust_mode, tripod_mode, suffix)
        ("BALANCED", False, False, "balanced_standard"),
        ("FAST", False, False, "fast_standard"),
        ("QUALITY", False, False, "quality_standard"),
        ("BALANCED", True, False, "balanced_robust"),
        ("BALANCED", False, True, "balanced_tripod"),
    ]

    for quality, robust, tripod, suffix in variations:
        out_name = f"{clip_name}_{suffix}.json"
        out_path = os.path.join(args.out_dir, out_name)

        print(f"  -> Generating variation {suffix}...")
        solve_sample = simulate_variation(trajectories, meta, quality, robust, tripod)

        with open(out_path, 'w') as f:
            json.dump(solve_sample, f, indent=4)
        print(f"     Saved dataset variation to: {out_name}")


def main():
    parser = argparse.ArgumentParser(description="AutoSolve PyTorch/CoTracker Trajectory Extractor (Bypassing Blender)")
    parser.add_argument("--video-path", help="Path to a single video clip to process")
    parser.add_argument("--clips-dir", default="ml/clips", help="Directory containing raw video clips")
    parser.add_argument("--out-dir", default="ml/data/raw", help="Directory to save raw JSON output files")
    parser.add_argument("--grid-size", type=int, default=8, help="Grid dimension for tracking points (e.g. 8 for 8x8 grid)")
    parser.add_argument("--no-cotracker", action="store_true", help="Force fallback to OpenCV Lucas-Kanade optical flow tracker")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    if args.video_path:
        if os.path.exists(args.video_path):
            process_video(args.video_path, args)
        else:
            print(f"Error: Video file not found: {args.video_path}")
    else:
        if not os.path.exists(args.clips_dir):
            print(f"Error: Clips directory does not exist: {args.clips_dir}")
            return
            
        supported_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.ogg', '.webm'}
        video_files = []
        for root, _, files in os.walk(args.clips_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in supported_extensions:
                    video_files.append(os.path.join(root, f))

        if not video_files:
            print(f"No supported video clips found in '{args.clips_dir}'")
            return

        print(f"Found {len(video_files)} video clips to process in '{args.clips_dir}'")
        for video_path in video_files:
            process_video(video_path, args)

    print("\nTrajectory extraction complete.")


if __name__ == "__main__":
    main()
