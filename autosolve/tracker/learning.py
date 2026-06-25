# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
LearningMixin for SmartTracker.

Manages clip classification, presets and quality configuration loading, mid-session
adaptation parameters, exponential moving average updates of region confidence,
and temporal/spatial dead-zone calculation from solve failures.
"""

import bpy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from .constants import REGIONS
from .utils import get_region


class LearningMixin:
    """Mixin for SmartTracker preset management and learning functions."""
    SURVIVAL_THRESHOLD = 0.5  # Below 50% ? add markers
    CRITICAL_THRESHOLD = 0.3  # Below 30% ? adapt settings

    def _classify_footage(self) -> str:
        """Classify footage by resolution and fps."""
        width = self.clip.size[0]
        fps = self.clip.fps if (self.clip.fps is not None and self.clip.fps > 0) else 24
        
        if width >= 3840:
            res = '4K'
        elif width >= 1920:
            res = 'HD'
        else:
            res = 'SD'
        
        if fps >= 50:
            fps_class = '60fps'
        elif fps >= 28:
            fps_class = '30fps'
        else:
            fps_class = '24fps'
        
        return f"{res}_{fps_class}"
    
    def _load_initial_settings(self):
        """
        Load initial settings using the unified SettingsPredictor.
        """
        self.current_settings = self.predictor.predict_settings(
            self.clip,
            robust_mode=False,
            footage_type=self.footage_type
        )
        print(f"AutoSolve: Predicted settings for {self.footage_class}: "
              f"pattern={self.current_settings.get('pattern_size')}px, "
              f"search={self.current_settings.get('search_size')}px, "
              f"corr={self.current_settings.get('correlation', 0.7):.2f}")
        
        quality_mult = self.quality_config
        self.current_settings['pattern_size'] = int(
            self.current_settings.get('pattern_size', 15) * quality_mult.get('pattern_size_mult', 1.0)
        )
        self.current_settings['search_size'] = int(
            self.current_settings.get('search_size', 71) * quality_mult.get('search_size_mult', 1.0)
        )
        if 'correlation' in quality_mult:
            if self.quality_preset == 'QUALITY':
                self.current_settings['correlation'] = max(
                    self.current_settings.get('correlation', 0.7),
                    quality_mult['correlation']
                )
            elif self.quality_preset == 'FAST':
                self.current_settings['correlation'] = min(
                    self.current_settings.get('correlation', 0.7),
                    quality_mult['correlation']
                )
        
        if 'motion_model' in quality_mult:
            self.current_settings['motion_model'] = quality_mult['motion_model']
        
        print(f"AutoSolve: After quality preset ({self.quality_preset}): "
              f"pattern={self.current_settings.get('pattern_size')}px, "
              f"search={self.current_settings.get('search_size')}px")
        
        if self.robust_mode:
            self.current_settings['pattern_size'] = int(self.current_settings.get('pattern_size', 15) * 1.4)
            self.current_settings['search_size'] = int(self.current_settings.get('search_size', 71) * 1.4)
            self.current_settings['correlation'] = max(0.45, self.current_settings.get('correlation', 0.7) - 0.15)
            self.current_settings['threshold'] = max(0.08, self.current_settings.get('threshold', 0.3) - 0.12)
            self.current_settings['motion_model'] = 'Affine'
            print(f"AutoSolve: Robust mode - enlarged search areas, lower thresholds")
        
        if self.tripod_mode:
            self.current_settings['motion_model'] = 'Loc'
            self.current_settings['correlation'] = min(0.80, 
                self.current_settings.get('correlation', 0.7) + 0.08)
            tripod_dead_zones = {'top-left', 'top-right', 'bottom-left', 'bottom-right'}
            self.known_dead_zones.update(tripod_dead_zones)
            print(f"AutoSolve: Tripod mode - Loc model, tighter correlation, avoiding corners")
        
        learned_dead_zones = self.predictor.get_dead_zones_for_class(self.footage_class)
        if learned_dead_zones:
            self.known_dead_zones = learned_dead_zones
            print(f"AutoSolve: Using LEARNED dead zones: {', '.join(learned_dead_zones)}")

    def adapt_settings_mid_session(self, survival_rate: float) -> Dict:
        """
        Adapt settings based on current session track survival rate.
        """
        if self.adaptation_count >= self.MAX_ADAPTATIONS:
            print(f"AutoSolve: Max adaptations reached ({self.MAX_ADAPTATIONS})")
            return {'adapted': False, 'reason': 'max_adaptations_reached'}
        
        old_settings = self.current_settings.copy()
        adapted = False
        changes = []
        
        if survival_rate < 0.3:
            new_search = min(151, int(self.current_settings.get('search_size', 71) * 1.3))
            new_pattern = min(31, int(self.current_settings.get('pattern_size', 15) * 1.2))
            
            if new_search != self.current_settings.get('search_size'):
                self.current_settings['search_size'] = new_search
                changes.append(f"search_size: {old_settings.get('search_size')} → {new_search}")
                adapted = True
            if new_pattern != self.current_settings.get('pattern_size'):
                self.current_settings['pattern_size'] = new_pattern
                changes.append(f"pattern_size: {old_settings.get('pattern_size')} → {new_pattern}")
                adapted = True
            
        elif survival_rate < 0.5:
            new_search = min(121, int(self.current_settings.get('search_size', 71) * 1.15))
            
            if new_search != self.current_settings.get('search_size'):
                self.current_settings['search_size'] = new_search
                changes.append(f"search_size: {old_settings.get('search_size')} → {new_search}")
                adapted = True
            
        elif survival_rate > 0.85:
            new_corr = min(0.85, self.current_settings.get('correlation', 0.7) + 0.05)
            if new_corr != self.current_settings.get('correlation'):
                self.current_settings['correlation'] = new_corr
                changes.append(f"correlation: {old_settings.get('correlation'):.2f} → {new_corr:.2f} (tighter)")
                adapted = True
        
        if adapted:
            self.adaptation_count += 1
            self.configure_settings()
            
            print(f"AutoSolve: MID-SESSION ADAPTATION #{self.adaptation_count}")
            for change in changes:
                print(f"  → {change}")
            
            return {'adapted': True, 'changes': changes, 'new_settings': self.current_settings.copy()}
        
        return {'adapted': False, 'reason': 'survival_rate_acceptable'}
    
    def update_region_confidence(self, region_stats: Dict):
        """
        Update region confidence scores based on tracking results.
        """
        LEARNING_RATE = 0.3
        
        for region, stats in region_stats.items():
            total = stats.get('total_tracks', 0)
            successful = stats.get('successful_tracks', 0)
            
            if total < 2:
                continue
            
            current_rate = successful / total
            old_confidence = self.region_confidence.get(region, 0.5)
            
            new_confidence = (1 - LEARNING_RATE) * old_confidence + LEARNING_RATE * current_rate
            self.region_confidence[region] = new_confidence
            
            if new_confidence < 0.25:
                self.known_dead_zones.add(region)
            elif new_confidence > 0.4 and region in self.known_dead_zones:
                self.known_dead_zones.discard(region)
        
        low_conf = [r for r, c in self.region_confidence.items() if c < 0.3]
        high_conf = [r for r, c in self.region_confidence.items() if c > 0.7]
        
        if low_conf:
            print(f"AutoSolve: Low confidence regions: {', '.join(low_conf)}")
        if high_conf:
            print(f"AutoSolve: High confidence regions: {', '.join(high_conf)}")
    
    def get_current_survival_rate(self, frame: Optional[int] = None) -> float:
        """
        Calculate current track survival rate.
        """
        if frame is None:
            frame = bpy.context.scene.frame_current
        
        total_tracks = len(self.tracking.tracks)
        if total_tracks == 0:
            return 0.0
        
        active_at_frame = 0
        clip_frame = self.scene_to_clip_frame(frame)
        for track in self.tracking.tracks:
            marker = track.markers.find_frame(clip_frame)
            if marker and not marker.mute:
                active_at_frame += 1
        
        rate = active_at_frame / total_tracks
        self.last_survival_rate = rate
        return rate
    
    def monitor_and_replenish(self, frame: int, backwards: bool = False) -> Dict:
        """
        Real-time monitoring with surgical replenishment.
        """
        proactive_muted = 0
        proactive_msg = ""
        predictor_ready = False
        if self.track_predictor is not None:
            if hasattr(self.track_predictor, 'track_model_available'):
                predictor_ready = self.track_predictor.track_model_available
            elif hasattr(self.track_predictor, 'model') and self.track_predictor.model is not None:
                predictor_ready = True
                
        if predictor_ready:
            active_tracks = []
            clip_frame = self.scene_to_clip_frame(frame)
            
            track_histories = {}
            for track in self.tracking.tracks:
                coords = []
                for f in range(1, clip_frame + 1):
                    m = track.markers.find_frame(f)
                    if m and not m.mute:
                        coords.append((m.co[0], m.co[1]))
                track_histories[track.name] = coords
                
                current_m = track.markers.find_frame(clip_frame)
                if current_m and not current_m.mute and len(coords) >= 6:
                    active_tracks.append(track)
                    
            if active_tracks:
                features_list = []
                for track in active_tracks:
                    coords = track_histories[track.name]
                    
                    neighbors = []
                    for n_name, n_coords in track_histories.items():
                        if n_name != track.name and len(n_coords) >= 1:
                            neighbors.append(n_coords)
                            
                    curr_m = track.markers.find_frame(clip_frame)
                    fx, fy = curr_m.co[0], curr_m.co[1]
                    ry = 0 if fy > 0.66 else (2 if fy < 0.33 else 1)
                    rx = 0 if fx < 0.33 else (2 if fx > 0.66 else 1)
                    region_idx = ry * 3 + rx
                    
                    FOOTAGE_TYPE_MAP = {
                        'AUTO': 0, 'INDOOR': 1, 'OUTDOOR': 2, 'DRONE': 3, 'HANDHELD': 4,
                        'GIMBAL': 5, 'ACTION': 6, 'VFX': 7, 'SCREEN': 8, 'CINEMATIC': 9
                    }
                    footage_idx = FOOTAGE_TYPE_MAP.get(self.footage_type, 0)
                    
                    from .features import extract_features_from_history
                    feats = extract_features_from_history(
                        coords=coords,
                        neighbors_coords=neighbors,
                        region_idx=region_idx,
                        footage_idx=footage_idx,
                        robust_mode=self.robust_mode
                    )
                    features_list.append(feats)
                    
                features_batch = np.array(features_list, dtype=np.float32)
                
                # Determine the expected input dimension dynamically from the model
                expected_dim = 15
                if self.track_predictor is not None:
                    if hasattr(self.track_predictor, 'track_input_dim'):
                        expected_dim = self.track_predictor.track_input_dim
                    elif hasattr(self.track_predictor, 'input_dim'):
                        expected_dim = self.track_predictor.input_dim

                if features_batch.ndim == 2:
                    features_batch = features_batch[:, :expected_dim]
                
                if hasattr(self.track_predictor, 'predict_track_survival'):
                    probs = self.track_predictor.predict_track_survival(features_batch)
                else:
                    probs = self.track_predictor.predict_survival(features_batch)
                
                for track, prob in zip(active_tracks, probs):
                    if prob < 0.3:
                        marker = track.markers.find_frame(clip_frame)
                        if marker:
                            marker.mute = True
                            proactive_muted += 1
                            
                if proactive_muted > 0:
                    proactive_msg = f"Retired {proactive_muted} weak tracks proactively"
                    print(f"AutoSolve: Proactive replacement - retired {proactive_muted} weak tracks at frame {frame} based on survival predictions")

        result = {
            'frame': frame,
            'survival_rate': self.get_current_survival_rate(frame),
            'markers_added': 0,
            'adapted': False,
            'changes': [proactive_msg] if proactive_msg else [],
        }
        
        if result['survival_rate'] < self.SURVIVAL_THRESHOLD:
            weak_regions = self._identify_weak_regions_at_frame(frame)
            
            for region in weak_regions[:3]:
                added = self.detect_in_region(region, count=1)
                result['markers_added'] += added
                if added > 0:
                    result['changes'].append(f"+{added} in {region}")
            
            if result['markers_added'] > 0:
                self.select_all_tracks()
        
        if result['survival_rate'] < self.CRITICAL_THRESHOLD:
            adaptation = self.adapt_settings_mid_session(result['survival_rate'])
            result['adapted'] = adaptation.get('adapted', False)
            if result['adapted']:
                result['changes'].extend(adaptation.get('changes', []))
        
        if result['markers_added'] > 0 or result['adapted']:
            print(f"AutoSolve: Frame {frame} - survival: {result['survival_rate']:.0%}, "
                  f"added: {result['markers_added']}, adapted: {result['adapted']}")
        
        return result
    
    def _identify_weak_regions_at_frame(self, frame: int) -> List[str]:
        """
        Identify regions with low track coverage at a specific frame.
        """
        all_regions = [
            'top-left', 'top-center', 'top-right',
            'mid-left', 'center', 'mid-right',
            'bottom-left', 'bottom-center', 'bottom-right'
        ]
        
        region_counts = {r: 0 for r in all_regions}
        
        clip_frame = self.scene_to_clip_frame(frame)
        for track in self.tracking.tracks:
            marker = track.markers.find_frame(clip_frame)
            if marker and not marker.mute:
                x, y = marker.co.x, marker.co.y
                region = self._get_region_for_position(x, y)
                if region:
                    region_counts[region] = region_counts.get(region, 0) + 1
        
        for dz in self.known_dead_zones:
            if dz in region_counts:
                del region_counts[dz]
        
        weak = [r for r, count in sorted(region_counts.items(), key=lambda x: x[1]) 
                if count < 2]
        
        return weak
    
    def _get_region_for_position(self, x: float, y: float) -> Optional[str]:
        """Map normalized x,y position to region name."""
        if x > 1 or y > 1:
            x = x / self.clip.size[0] if self.clip.size[0] else x
            y = y / self.clip.size[1] if self.clip.size[1] else y
        
        col = 0 if x < 0.33 else (1 if x < 0.66 else 2)
        row = 2 if y < 0.33 else (1 if y < 0.66 else 0)
        
        region_map = [
            ['top-left', 'top-center', 'top-right'],
            ['mid-left', 'center', 'mid-right'],
            ['bottom-left', 'bottom-center', 'bottom-right']
        ]
        
        return region_map[row][col]
    
    def _blend_settings(self, settings_a: Dict, settings_b: Dict, weight_a: float = 0.5) -> Dict:
        """
        Blend two settings dicts with weighted average.
        """
        weight_b = 1.0 - weight_a
        blended = {}
        
        for key in ['pattern_size', 'search_size']:
            val_a = settings_a.get(key, 15 if key == 'pattern_size' else 71)
            val_b = settings_b.get(key, 15 if key == 'pattern_size' else 71)
            blended[key] = int(val_a * weight_a + val_b * weight_b) | 1
        
        for key in ['correlation', 'threshold']:
            val_a = settings_a.get(key, 0.7 if key == 'correlation' else 0.3)
            val_b = settings_b.get(key, 0.7 if key == 'correlation' else 0.3)
            blended[key] = round(val_a * weight_a + val_b * weight_b, 2)
        
        blended['motion_model'] = settings_a.get('motion_model', settings_b.get('motion_model', 'LocRot'))
        
        return blended
    
    def _get_learned_skip_regions(self) -> Set[str]:
        """
        Get regions to skip based on default settings.
        """
        skip = set()
        SKIP_THRESHOLD = 0.25
        
        region_models = self.predictor.model.get('region_models', {})
        
        for region, data in region_models.items():
            if region not in REGIONS:
                continue
            
            rate = data.get('success_rate', 1.0)
            if rate < SKIP_THRESHOLD:
                skip.add(region)
                    
        return skip
    
    def get_user_priority_regions(self) -> Dict[str, List[str]]:
        """
        Extract priority regions from user-placed markers.
        """
        priority = {
            'high': set(),
            'existing': set(),
        }
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if not markers:
                continue
            
            region = get_region(
                markers[0].co.x, markers[0].co.y
            )
            
            if len(markers) <= 2:
                priority['high'].add(region)
            else:
                priority['existing'].add(region)
        
        return {k: list(v) for k, v in priority.items()}

    def _is_non_rigid_region(self, region: str) -> bool:
        """
        Check if a region is likely to contain non-rigid objects (waves, water, foliage).
        """
        if not hasattr(self, 'cached_motion_probe') or not self.cached_motion_probe:
            return False
        
        probe = self.cached_motion_probe
        region_success = probe.get('region_success', {})
        
        if region in region_success:
            region_data = region_success[region]
            if isinstance(region_data, dict):
                success_rate = region_data.get('success_rate', 1.0)
                if region_data.get('total', 0) > 0:
                    success_rate = region_data.get('success', 0) / region_data['total']
                
                jitters = region_data.get('jitters', [])
                if jitters:
                    avg_jitter = sum(jitters) / len(jitters)
                    all_jitters = []
                    for r, rd in region_success.items():
                        if isinstance(rd, dict):
                            all_jitters.extend(rd.get('jitters', []))
                    
                    global_avg_jitter = sum(all_jitters) / len(all_jitters) if all_jitters else 0.01
                    
                    if global_avg_jitter > 0 and avg_jitter > global_avg_jitter * 2:
                        print(f"AutoSolve: Skipping {region} - high motion variance "
                              f"({avg_jitter:.4f} >> avg {global_avg_jitter:.4f}) - likely water/waves")
                        return True
                
                velocities = region_data.get('velocities', [])
                if velocities and len(velocities) >= 2:
                    avg_v = sum(velocities) / len(velocities)
                    if avg_v > 0:
                        variance = sum((v - avg_v)**2 for v in velocities) / len(velocities)
                        coefficient_of_variation = (variance ** 0.5) / avg_v
                        if coefficient_of_variation > 0.8:
                            print(f"AutoSolve: Skipping {region} - erratic velocity (CoV={coefficient_of_variation:.2f})")
                            return True
            else:
                success_rate = region_data
            
            if success_rate < 0.2:
                print(f"AutoSolve: Skipping {region} - probe showed {success_rate:.0%} success")
                return True
        
        velocities = probe.get('velocities', {})
        if region in velocities:
            region_velocity = velocities[region]
            avg_velocity = probe.get('avg_velocity', 0.01)
            if avg_velocity > 0 and region_velocity > avg_velocity * 3:
                print(f"AutoSolve: Skipping {region} - velocity {region_velocity:.3f} >> avg {avg_velocity:.3f}")
                return True
        
        if region in self.known_dead_zones:
            return True
        
        return False

    def _get_frame_segment(self, frame: int, segment_size: int = 50) -> Tuple[int, int]:
        """Get the segment (start, end) for a given frame."""
        segment_start = (frame // segment_size) * segment_size
        segment_end = segment_start + segment_size
        return (segment_start, segment_end)
    
    def learn_from_failed_tracks(self):
        """
        Analyze tracks that failed reconstruction and update temporal dead zones.
        """
        failed = []
        for track in self.tracking.tracks:
            if not track.has_bundle:
                failed.append(track)
            elif track.has_bundle and track.average_error > 5.0:
                failed.append(track)
        
        if not failed:
            print("AutoSolve: No failed tracks to learn from")
            return
        
        for track in failed:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            avg_x = sum(m.co.x for m in markers) / len(markers)
            avg_y = sum(m.co.y for m in markers) / len(markers)
            region = get_region(avg_x, avg_y)
            
            markers_sorted = sorted(markers, key=lambda m: m.frame)
            start_frame = markers_sorted[0].frame
            end_frame = markers_sorted[-1].frame
            
            for frame in range(start_frame, end_frame + 1, 50):
                segment = self._get_frame_segment(frame)
                if segment not in self.temporal_dead_zones:
                    self.temporal_dead_zones[segment] = {}
                
                if region not in self.temporal_dead_zones[segment]:
                    self.temporal_dead_zones[segment][region] = 0
                
                self.temporal_dead_zones[segment][region] += 1
            
        print(f"AutoSolve: Learned from {len(failed)} failed tracks")
        self._print_temporal_dead_zones()
    
    def _print_temporal_dead_zones(self):
        """Print summary of temporal dead zones."""
        if not self.temporal_dead_zones:
            return
        
        hot_zones = []
        for segment, regions in self.temporal_dead_zones.items():
            for region, count in regions.items():
                if count >= 3:
                    hot_zones.append(f"{region}@{segment[0]}-{segment[1]}: {count} failures")
        
        if hot_zones:
            print(f"AutoSolve: Temporal hot zones: {', '.join(hot_zones[:5])}")
    
    def is_in_temporal_dead_zone(self, x: float, y: float, frame: int) -> bool:
        """
        Check if a position at a specific frame is in a known temporal dead zone.
        """
        segment = self._get_frame_segment(frame)
        if segment not in self.temporal_dead_zones:
            return False
        
        region = get_region(x, y)
        failure_count = self.temporal_dead_zones[segment].get(region, 0)
        
        return failure_count >= 3
