# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
DetectionMixin for SmartTracker.

Governs distributed region-based detection, concentrated annotation-aware
placement, texture quality checks via PixelAnalyzer, feature quality scoring,
and smart exploratory/reinforcement feature detection.
"""

import bpy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any

from .constants import REGIONS
from .utils import get_region, get_region_bounds

# Detection thresholds
DETECTION_THRESHOLD_MULTIPLIER = 0.5


class DetectionMixin:
    """Mixin for SmartTracker feature detection and replenishment."""

    def _get_feature_placement(self) -> str:
        """
        Get the placement mode for detect_features based on annotation_mode setting.
        
        Returns:
            'FRAME' - detect everywhere
            'INSIDE_GPENCIL' - detect only inside annotation
            'OUTSIDE_GPENCIL' - detect only outside annotation
        """
        try:
            annotation_mode = bpy.context.scene.autosolve.annotation_mode
            if annotation_mode == 'INCLUDE':
                return 'INSIDE_GPENCIL'
            elif annotation_mode == 'EXCLUDE':
                return 'OUTSIDE_GPENCIL'
            else:
                return 'FRAME'
        except Exception:
            return 'FRAME'
    
    def _has_active_annotation(self) -> bool:
        """
        Check if there are active annotations and annotation_mode is set.
        
        Returns True when:
        1. annotation_mode is INCLUDE or EXCLUDE (not NONE)
        2. There is actual gpencil/annotation data with strokes
        
        Used to decide between concentrated vs distributed detection.
        """
        try:
            annotation_mode = bpy.context.scene.autosolve.annotation_mode
            if annotation_mode == 'NONE':
                return False
            
            # Check for actual gpencil data on the clip
            gpd = None
            if hasattr(self.clip, "annotation"):
                gpd = self.clip.annotation
            elif hasattr(self.clip, "grease_pencil"):
                gpd = self.clip.grease_pencil
            elif hasattr(bpy.context, "annotation_data"):
                gpd = bpy.context.annotation_data
                
            if gpd and gpd.layers:
                for layer in gpd.layers:
                    if layer.frames:
                        for frame in layer.frames:
                            if frame.strokes:
                                return True
            return False
        except Exception:
            return False

    def detect_in_region(self, region: str, count: int = 3) -> int:
        """
        Detect features within a specific screen region.
        
        Approach: Detect globally with low threshold, then filter to keep
        only features in the target region (up to count).
        """
        # Phase 6: Skip non-rigid regions (waves, water) during detection
        if self._is_non_rigid_region(region):
            print(f"AutoSolve: Skipping {region} - likely non-rigid (water/waves)")
            return 0
        
        bounds = get_region_bounds(region)
        x_min, y_min, x_max, y_max = bounds
        
        existing_tracks = set(self.tracking.tracks)
        
        # Detect globally with low threshold to get many candidates
        threshold = self.current_settings.get('threshold', 0.3) * DETECTION_THRESHOLD_MULTIPLIER
        
        try:
            self._run_ops(
                bpy.ops.clip.detect_features,
                threshold=threshold,
                min_distance=25,
                margin=20,
                placement=self._get_feature_placement()
            )
        except Exception as e:
            print(f"AutoSolve: detect_features failed: {e}")
            return 0
        
        # Filter: keep only tracks in target region, limit to count
        new_tracks = [t for t in self.tracking.tracks if t not in existing_tracks]
        
        if not new_tracks:
            return 0
        
        # Categorize tracks by region
        in_region = []
        outside = []
        
        current_frame = bpy.context.scene.frame_current
        clip_frame = self.scene_to_clip_frame(current_frame)  # Convert to clip-relative
        
        for track in new_tracks:
            # Try to find marker at current frame
            marker = track.markers.find_frame(clip_frame)
            
            # If no marker at exact frame, try to get any marker from this track
            if not marker and len(track.markers) > 0:
                marker = track.markers[0]
            
            if not marker:
                outside.append(track)
                continue
            
            # Check if in target region bounds
            x, y = marker.co.x, marker.co.y
            if x_min <= x <= x_max and y_min <= y <= y_max:
                in_region.append(track)
            else:
                outside.append(track)

        # Keep up to 'count' tracks in the region
        kept = 0
        for track in in_region[:count]:
            self._apply_track_settings(track)
            track.select = False
            kept += 1
        
        # Mark excess and outside tracks for deletion
        for track in in_region[count:] + outside:
            track.select = True
        
        # Delete marked tracks
        if in_region[count:] or outside:
            try:
                self._run_ops(bpy.ops.clip.delete_track)
            except Exception:
                pass
        
        return kept
    
    def _detect_concentrated_in_annotation(self) -> Dict[str, int]:
        """
        Detect features concentrated in annotation region.
        """
        existing_tracks = set(self.tracking.tracks)
        
        # More aggressive detection for denser coverage
        base_threshold = self.current_settings.get('threshold', 0.3)
        threshold = base_threshold * DETECTION_THRESHOLD_MULTIPLIER
        
        placement = self._get_feature_placement()
        
        try:
            self._run_ops(
                bpy.ops.clip.detect_features,
                threshold=threshold,
                min_distance=15,  # Smaller = more dense coverage
                margin=16,
                placement=placement
            )
        except Exception as e:
            print(f"AutoSolve: Concentrated detection failed: {e}")
            return {r: 0 for r in REGIONS}
        
        new_tracks = [t for t in self.tracking.tracks if t not in existing_tracks]
        
        if not new_tracks:
            print("AutoSolve: Concentrated detection found no features")
            return {r: 0 for r in REGIONS}
        
        # Apply settings to all detected tracks (no region-based filtering)
        clip_frame = self.scene_to_clip_frame(bpy.context.scene.frame_current)
        result = {r: 0 for r in REGIONS}
        
        for track in new_tracks:
            self._apply_track_settings(track)
            track.select = False
            
            # Count by region for logging
            marker = track.markers.find_frame(clip_frame)
            if not marker and len(track.markers) > 0:
                marker = track.markers[0]
            if marker:
                region = get_region(marker.co.x, marker.co.y)
                result[region] = result.get(region, 0) + 1
        
        print(f"AutoSolve: Concentrated detection ({placement}): {len(new_tracks)} markers")
        
        # Store for feature extractor
        self._detected_feature_density = result
        
        return result
    
    def detect_all_regions(self, markers_per_region: int = 3, 
                          skip_regions: Optional[Set[str]] = None) -> Dict[str, int]:
        """
        OPTIMIZED: Detect features with single global pass, distribute to all regions.
        """
        # Annotation-aware: concentrate markers instead of distributing
        if self._has_active_annotation():
            return self._detect_concentrated_in_annotation()
        
        skip_regions = skip_regions or set()
        
        # Add non-rigid regions to skip list
        for region in REGIONS:
            if self._is_non_rigid_region(region):
                skip_regions.add(region)
        
        # Also skip known dead zones
        skip_regions.update(self.known_dead_zones)
        
        if skip_regions:
            print(f"AutoSolve: Skipping regions: {', '.join(skip_regions)}")
        
        existing_tracks = set(self.tracking.tracks)
        
        # Use HIGHER threshold for better quality initial features
        base_threshold = self.current_settings.get('threshold', 0.3)
        threshold = max(0.4, base_threshold) * DETECTION_THRESHOLD_MULTIPLIER
        
        # Detect with smaller min_distance to get MORE candidates
        try:
            self._run_ops(
                bpy.ops.clip.detect_features,
                threshold=threshold,
                min_distance=25,
                margin=20,
                placement=self._get_feature_placement()
            )
        except Exception as e:
            print(f"AutoSolve: detect_features failed: {e}")
            return {r: 0 for r in REGIONS}
        
        new_tracks = [t for t in self.tracking.tracks if t not in existing_tracks]
        
        # Verify detection was successful
        if not new_tracks:
            print("AutoSolve: No features detected")
            return {r: 0 for r in REGIONS}
        
        print(f"AutoSolve: Global detection found {len(new_tracks)} candidates")
        
        # Categorize all tracks by region WITH QUALITY SCORE
        tracks_by_region: Dict[str, List[Tuple[Any, float]]] = {r: [] for r in REGIONS}
        detected_per_region: Dict[str, int] = {r: 0 for r in REGIONS}  # For feature density
        no_marker_tracks = []
        
        current_frame = bpy.context.scene.frame_current
        clip_frame = self.scene_to_clip_frame(current_frame)  # Convert to clip-relative
        
        for track in new_tracks:
            marker = track.markers.find_frame(clip_frame)
            if not marker and len(track.markers) > 0:
                marker = track.markers[0]
            
            if not marker:
                no_marker_tracks.append(track)
                continue
            
            # Score the feature based on position quality
            quality = self._score_feature_quality(marker, track)
            
            region = get_region(marker.co.x, marker.co.y)
            tracks_by_region[region].append((track, quality))
            detected_per_region[region] += 1  # Count for feature density
        
        # Store detected counts for feature extractor (before filtering)
        self._detected_feature_density = detected_per_region
        
        # Estimate texture quality per region
        qualities = self._estimate_region_texture_quality()
        
        # Load empirical region weights
        try:
            weights_path = Path(__file__).parent / 'presets' / 'region_weights.json'
            if weights_path.exists():
                with open(weights_path, 'r') as f:
                    region_weights = json.load(f)
            else:
                region_weights = {}
        except Exception as e:
            print(f"AutoSolve: Failed to load region weights preset: {e}")
            region_weights = {}
            
        f_weights = region_weights.get(self.footage_type, {})
        if not f_weights:
            f_weights = region_weights.get('AUTO', {})
            
        max_q = max(qualities.values()) if qualities else 0.0
        
        # Process each region: SORT BY QUALITY, keep top N scaled by weight and quality
        result: Dict[str, int] = {}
        to_delete = list(no_marker_tracks)  # Always delete tracks without markers
        
        for region in REGIONS:
            region_tracks = tracks_by_region[region]
            
            if region in skip_regions:
                # Skip this region entirely - delete all its tracks
                to_delete.extend([t for t, _ in region_tracks])
                result[region] = 0
                continue
            
            # Scale target markers using region weights and texture quality
            weight = f_weights.get(region, 1.0)
            q = qualities.get(region, 1.0)
            norm_q = q / max_q if max_q > 0.0 else 1.0
            
            adjusted_target = markers_per_region * weight * norm_q
            adjusted_markers_per_region = int(round(adjusted_target))
            
            # Automatically skip near-uniform regions (sky, walls)
            is_uniform = False
            # Only skip if we did NOT fall back to density estimation (since density doesn't tell us if it's uniform)
            if not getattr(self, 'last_quality_fallback', False) and max_q > 0.0:
                if q < 0.0005:  # Raw variance threshold for grayscale range [0, 1]
                    is_uniform = True
            
            if is_uniform:
                print(f"AutoSolve: Skipping near-uniform region '{region}' (quality score: {q:.6f})")
                adjusted_markers_per_region = 0
            
            # SORT by quality score (highest first)
            region_tracks.sort(key=lambda x: x[1], reverse=True)
            
            # Keep up to adjusted_markers_per_region of the BEST quality features
            keep_count = min(len(region_tracks), adjusted_markers_per_region)
            
            if adjusted_markers_per_region != markers_per_region:
                print(f"AutoSolve: {region} target scaled from {markers_per_region} to {adjusted_markers_per_region} (weight: {weight:.2f}, quality: {norm_q:.2f})")
            
            for track, quality in region_tracks[:keep_count]:
                self._apply_track_settings(track)
                track.select = False
            
            # Mark excess for deletion
            to_delete.extend([t for t, _ in region_tracks[keep_count:]])
            result[region] = keep_count
        
        # Single batch deletion
        if to_delete:
            for track in to_delete:
                track.select = True
            try:
                self._run_ops(bpy.ops.clip.delete_track)
            except Exception:
                pass
        
        total = sum(result.values())
        active_regions = sum(1 for c in result.values() if c > 0)
        print(f"AutoSolve: Distributed {total} quality-selected markers across {active_regions}/9 regions")
        
        return result

    def _estimate_region_texture_quality(self) -> Dict[str, float]:
        """
        Estimate texture quality (luminance variance) in each region.
        """
        print("AutoSolve: Estimating region texture quality...")
        self.last_quality_fallback = False
        qualities = {}
        
        current_frame = bpy.context.scene.frame_current
        
        try:
            # Use PixelAnalyzer to get the grayscale frame pixels
            clip_frame = self.scene_to_clip_frame(current_frame)
            gray = self.pixel_analyzer.get_frame_gray(self.clip, clip_frame)
            if gray is None:
                raise ValueError("Could not retrieve frame pixels from PixelAnalyzer")
                
            H, W = gray.shape
            
            # Sample 8x8 grid for each region
            for region in REGIONS:
                x_min, y_min, x_max, y_max = get_region_bounds(region)
                px_min_x = int(x_min * W)
                px_max_x = int(x_max * W)
                # Flip y coordinates because numpy uses top-left origin
                py_min_y = int((1.0 - y_max) * H)
                py_max_y = int((1.0 - y_min) * H)
                
                dx = (px_max_x - px_min_x) / 8.0
                dy = (py_max_y - py_min_y) / 8.0
                
                luminances = []
                for i in range(8):
                    for j in range(8):
                        px = int(px_min_x + (i + 0.5) * dx)
                        py = int(py_min_y + (j + 0.5) * dy)
                        px = max(0, min(px, W - 1))
                        py = max(0, min(py, H - 1))
                        
                        luminances.append(float(gray[py, px]))
                        
                if luminances:
                    mean_lum = sum(luminances) / len(luminances)
                    variance = sum((lum - mean_lum) ** 2 for lum in luminances) / len(luminances)
                    qualities[region] = variance
                else:
                    qualities[region] = 0.0
                
            print("AutoSolve: Successfully estimated region texture qualities from image data via PixelAnalyzer.")
            
        except Exception as e:
            print(f"AutoSolve: Failed to estimate texture quality using PixelAnalyzer: {e}")
            self.last_quality_fallback = True
            # Fallback to self._detected_feature_density
            print("AutoSolve: Falling back to detected feature density for texture quality estimation.")
            if hasattr(self, '_detected_feature_density') and self._detected_feature_density:
                max_density = max(self._detected_feature_density.values())
                if max_density > 0:
                    qualities = {
                        r: self._detected_feature_density.get(r, 0) / max_density
                        for r in REGIONS
                    }
                else:
                    qualities = {r: 1.0 for r in REGIONS}
            else:
                qualities = {r: 1.0 for r in REGIONS}
                
        return qualities
    
    def _score_feature_quality(self, marker, track) -> float:
        """
        Score a feature by its quality for tracking.
        """
        x, y = marker.co.x, marker.co.y
        
        # Base score - start at 1.0
        score = 1.0
        
        # Penalty for extreme edges
        edge_margin = 0.08
        if x < edge_margin or x > (1.0 - edge_margin):
            score *= 0.7
        if y < edge_margin or y > (1.0 - edge_margin):
            score *= 0.7
        
        # Small bonus for center region
        center_dist = ((x - 0.5) ** 2 + (y - 0.5) ** 2) ** 0.5
        if center_dist < 0.25:
            score *= 1.1
        
        # Prefer features not too close to other existing tracks
        min_dist_to_existing = self._min_distance_to_existing_tracks(x, y)
        if min_dist_to_existing < 0.03:  # Too close
            score *= 0.6
        elif min_dist_to_existing > 0.1:  # Good distance
            score *= 1.15
            
        # Use PixelAnalyzer to get texture and rigidity score
        try:
            current_frame = bpy.context.scene.frame_current
            clip_frame = self.scene_to_clip_frame(current_frame)
            pixel_score = self.pixel_analyzer.score_marker_position(self.clip, track, clip_frame)
            score *= pixel_score
        except Exception as pe:
            print(f"AutoSolve: Pixel scoring failed for track {track.name}: {pe}")
        
        return min(score, 1.0)
    
    def _min_distance_to_existing_tracks(self, x: float, y: float) -> float:
        """Calculate minimum distance to existing tracks (that we're keeping)."""
        min_dist = float('inf')
        
        current_frame = bpy.context.scene.frame_current
        clip_frame = self.scene_to_clip_frame(current_frame)
        
        for track in self.tracking.tracks:
            if not track.select:  # Only check tracks we're keeping
                marker = track.markers.find_frame(clip_frame)
                if marker:
                    dist = ((marker.co.x - x) ** 2 + (marker.co.y - y) ** 2) ** 0.5
                    min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 1.0

    # Exploratory settings variations for learning what works
    EXPLORATORY_SETTINGS = {
        'top-left': {'pattern_size': 11, 'search_size': 61, 'correlation': 0.75},
        'top-center': {'pattern_size': 15, 'search_size': 71, 'correlation': 0.70},
        'top-right': {'pattern_size': 19, 'search_size': 91, 'correlation': 0.65},
        'mid-left': {'pattern_size': 13, 'search_size': 81, 'correlation': 0.72},
        'center': {'pattern_size': 17, 'search_size': 71, 'correlation': 0.68},
        'mid-right': {'pattern_size': 21, 'search_size': 101, 'correlation': 0.60},
        'bottom-left': {'pattern_size': 15, 'search_size': 91, 'correlation': 0.65},
        'bottom-center': {'pattern_size': 19, 'search_size': 81, 'correlation': 0.70},
        'bottom-right': {'pattern_size': 13, 'search_size': 61, 'correlation': 0.75},
    }
    
    def _get_learned_region_settings(self) -> Dict[str, Dict]:
        """
        Get per-region settings from learning + exploratory baseline.
        """
        region_settings = {}
        
        region_advice = {}
        if hasattr(self, 'predictor') and self.predictor:
            region_advice = self.predictor.get_region_advice()
        
        for region, base_settings in self.EXPLORATORY_SETTINGS.items():
            region_settings[region] = base_settings.copy()
            
            advice = region_advice.get(region, 'normal')
            
            if advice == 'avoid':
                region_settings[region]['search_size'] = int(base_settings['search_size'] * 1.5)
                region_settings[region]['correlation'] = max(0.5, base_settings['correlation'] - 0.1)
                region_settings[region]['avoid'] = True
            elif advice == 'prioritize':
                region_settings[region]['correlation'] = min(0.8, base_settings['correlation'] + 0.05)
                region_settings[region]['prioritize'] = True
        
        return region_settings
    
    def detect_features_smart(self, markers_per_region: int = 3, use_cached_probe: bool = True) -> int:
        """
        SMART DETECTION
        """
        print(f"AutoSolve: Starting feature detection...")
        
        if use_cached_probe and hasattr(self, 'cached_motion_probe') and self.cached_motion_probe:
            probe_results = self.cached_motion_probe
            print(f"AutoSolve: Using cached probe (motion: {probe_results.get('motion_class')})")
            self.motion_class = probe_results.get('motion_class', 'MEDIUM')
        else:
            probe_results = self._run_motion_probe()
            self.cached_motion_probe = probe_results
            self._save_probe_to_cache(probe_results)
        
        detection_frame = self.get_optimal_start_frame()
        bpy.context.scene.frame_set(detection_frame)
        
        motion_class = probe_results.get('motion_class', 'MEDIUM')
        texture_class = probe_results.get('texture_class', 'MEDIUM')
        best_regions = probe_results.get('best_regions', [])
        
        learned_regions = self._get_learned_region_settings()
        has_learned_data = any(
            'prioritize' in v or 'avoid' in v 
            for v in learned_regions.values()
        )
        
        if has_learned_data:
            print(f"AutoSolve: Using learned region settings")
            total = self._detect_with_region_settings(learned_regions, markers_per_region, motion_class)
        else:
            target = markers_per_region if motion_class != 'HIGH' else max(1, markers_per_region - 1)
            total = self._detect_quality_markers(
                motion_class=motion_class,
                texture_class=texture_class,
                markers_per_region=target,
                priority_regions=best_regions
            )
        
        print(f"AutoSolve: Smart detection complete - {total} markers placed")
        
        min_required = max(15, int(self.target_tracks * 0.4))
        if total < min_required:
            print(f"AutoSolve: Only {total} markers (target {self.target_tracks}), adding reinforcements...")
            extra = self._add_reinforcement_markers(total, motion_class)
            total += extra
        
        return total
    
    def _detect_with_region_settings(self, region_settings: Dict[str, Dict], 
                                     markers_per_region: int, motion_class: str) -> int:
        """
        Detect features using per-region learned settings.
        """
        total = 0
        regions = list(region_settings.keys())
        
        regions.sort(key=lambda r: (
            0 if region_settings[r].get('prioritize') else
            2 if region_settings[r].get('avoid') else 1
        ))
        
        for region in regions:
            if region in self.known_dead_zones:
                continue
            
            settings = region_settings[region]
            
            if settings.get('avoid') and motion_class == 'HIGH':
                continue
            
            count = markers_per_region + 1 if settings.get('prioritize') else markers_per_region
            
            old_settings = self.current_settings.copy()
            
            try:
                self.current_settings.update({
                    'pattern_size': settings.get('pattern_size', 15),
                    'search_size': settings.get('search_size', 71),
                    'correlation': settings.get('correlation', 0.70),
                })
                self.configure_settings()

                detected = self.detect_in_region(region, count)
                total += detected

            finally:
                self.current_settings = old_settings
                self.configure_settings()
            
            if detected > 0:
                print(f"AutoSolve: {region}: {detected} markers (learned settings)")
        
        return total

    def _detect_quality_markers(self, motion_class: str, texture_class: str,
                                markers_per_region: int, priority_regions: list = None) -> int:
        """
        Place quality markers based on motion/texture analysis.
        """
        is_4k = self.clip.size[0] >= 3840
        resolution_multiplier = 1.5 if is_4k else 1.0
        
        if motion_class == 'HIGH':
            settings = {
                'pattern_size': int(25 * resolution_multiplier),
                'search_size': int(141 * resolution_multiplier),
                'correlation': 0.55,
                'threshold': 0.20,
                'motion_model': 'Affine',
            }
        elif motion_class == 'MEDIUM':
            settings = {
                'pattern_size': int(19 * resolution_multiplier),
                'search_size': int(101 * resolution_multiplier),
                'correlation': 0.65,
                'threshold': 0.25,
                'motion_model': 'LocRot',
            }
        else:  # LOW
            settings = {
                'pattern_size': int(15 * resolution_multiplier),
                'search_size': int(71 * resolution_multiplier),
                'correlation': 0.72,
                'threshold': 0.30,
                'motion_model': 'Loc',
            }
        
        if texture_class == 'LOW':
            settings['threshold'] *= 0.6
            settings['correlation'] -= 0.1
        
        self.current_settings = settings.copy()
        self.configure_settings()
        
        print(f"AutoSolve: Quality settings - Pattern:{settings['pattern_size']}, "
              f"Search:{settings['search_size']}, Corr:{settings['correlation']:.2f}"
              f"{' (4K scaled)' if is_4k else ''}")
        
        region_results = self.detect_all_regions(markers_per_region=markers_per_region)
        
        if priority_regions:
            priority_found = sum(region_results.get(r, 0) for r in priority_regions)
            print(f"AutoSolve: Priority regions ({', '.join(priority_regions)}): {priority_found} markers")
        
        return sum(region_results.values())
    
    def _add_reinforcement_markers(self, current_count: int, motion_class: str) -> int:
        """
        Add reinforcement markers if we don't have enough.
        """
        needed = max(0, self.target_tracks - current_count)
        if needed == 0:
            return 0
        
        print(f"AutoSolve: Adding {needed} reinforcement markers...")
        
        reliable_regions = ['center', 'mid-left', 'mid-right', 'bottom-center']
        
        added = 0
        for region in reliable_regions:
            if added >= needed:
                break
            detected = self.detect_in_region(region, count=(needed - added))
            added += detected
        
        remaining_regions = [r for r in REGIONS if r not in reliable_regions]
        for region in remaining_regions:
            if added >= needed:
                break
            detected = self.detect_in_region(region, count=(needed - added))
            added += detected
        
        return added
    
    def _apply_exploratory_track_settings(self, track, region: str):
        """Apply region-specific exploratory settings to a track."""
        settings = self.EXPLORATORY_SETTINGS.get(region, self.current_settings)
        
        if hasattr(track, 'pattern_size'):
            track.pattern_size = settings.get('pattern_size', 15)
        if hasattr(track, 'search_size'):
            track.search_size = settings.get('search_size', 71)
        if hasattr(track, 'correlation_min'):
            track.correlation_min = settings.get('correlation', 0.7)
        if hasattr(track, 'motion_model'):
            track.motion_model = settings.get('motion_model', 'LocRot')
