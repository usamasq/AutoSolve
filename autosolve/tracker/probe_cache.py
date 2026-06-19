# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
ProbeCacheMixin for SmartTracker.

Handles user resource cache path resolution, cached probe loading/saving,
and running the full motion probe or quick metadata-based motion estimates.
"""

import bpy
import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .constants import REGIONS
from .utils import get_region


class ProbeCacheMixin:
    """Mixin for SmartTracker motion probing and cache management."""

    def _get_probe_cache_path(self) -> Path:
        """Get path for cached probe data."""
        cache_dir = Path(bpy.utils.user_resource('SCRIPTS')) / 'autosolve' / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Hash clip filepath for unique identifier
        # For packed/embedded clips, use clip name + resolution as fallback
        filepath = self.clip.filepath or f"{self.clip.name}_{self.clip.size[0]}x{self.clip.size[1]}"
        clip_hash = hashlib.md5(filepath.encode()).hexdigest()[:8]
        return cache_dir / f"probe_{clip_hash}.json"
    
    def _try_load_cached_probe(self):
        """Try to load cached probe from disk if valid."""
        cache_path = self._get_probe_cache_path()
        if not cache_path.exists():
            return
        
        try:
            data = json.loads(cache_path.read_text())
            
            # Validate cache is for same clip version
            # Skip mtime check for packed/embedded clips (no filepath)
            if self.clip.filepath:
                try:
                    clip_mtime = os.path.getmtime(bpy.path.abspath(self.clip.filepath))
                    if abs(data.get('clip_mtime', 0) - clip_mtime) >= 1:
                        return  # Cache is stale
                except (FileNotFoundError, OSError):
                    pass  # Can't verify mtime, use cache anyway
            self.cached_motion_probe = data.get('probe_results')
            print(f"AutoSolve: Loaded cached probe for {self.clip.name}")
        except Exception as e:
            print(f"AutoSolve: Could not load cached probe: {e}")
    
    def _save_probe_to_cache(self, probe_results: Dict):
        """Save probe results to disk for reuse."""
        try:
            cache_path = self._get_probe_cache_path()
            # Get mtime for external clips, use 0 for packed/embedded
            clip_mtime = 0
            if self.clip.filepath:
                try:
                    clip_mtime = os.path.getmtime(bpy.path.abspath(self.clip.filepath))
                except (FileNotFoundError, OSError):
                    pass
            
            cache_data = {
                'clip_filepath': self.clip.filepath,
                'clip_mtime': clip_mtime,
                'footage_class': self.footage_class,
                'probe_results': probe_results,
            }
            
            cache_path.write_text(json.dumps(cache_data, indent=2))
            print(f"AutoSolve: Cached probe for {self.clip.name}")
        except Exception as e:
            print(f"AutoSolve: Could not cache probe: {e}")

    def _estimate_motion_quick(self) -> str:
        """
        Quick motion estimate from clip metadata (no tracking needed).
        
        This avoids the expensive full motion probe for obvious cases.
        
        Returns:
            'LOW', 'MEDIUM', or 'HIGH' motion class estimate
        """
        fps = self.clip.fps if self.clip.fps > 0 else 24
        duration = self.clip.frame_duration
        
        # Higher FPS = less motion per frame (smoother footage)
        if fps >= 50:
            fps_class = 'LOW'
        elif fps >= 28:
            fps_class = 'MEDIUM'
        else:
            fps_class = 'HIGH'  # 24fps often has more apparent motion
        
        # Short clips often have dramatic motion
        if duration < 100:
            duration_class = 'HIGH'
        elif duration < 300:
            duration_class = 'MEDIUM'
        else:
            duration_class = 'LOW'
        
        # Footage type hints
        if self.footage_type in ['DRONE', 'ACTION', 'HANDHELD']:
            type_class = 'HIGH'
        elif self.footage_type in ['INDOOR', 'TRIPOD']:
            type_class = 'LOW'
        else:
            type_class = 'MEDIUM'
        
        # Combine: take highest motion estimate
        classes = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        max_class = max([fps_class, duration_class, type_class], key=lambda x: classes[x])
        
        return max_class
    
    def _run_motion_probe(self) -> dict:
        """
        Run a quick motion probe to analyze footage characteristics.
        
        OPTIMIZATION: Now checks quick estimate first and skips full probe
        when not needed (for LOW/MEDIUM motion without robust mode).
        
        Places 1 marker per region, tracks ~20 frames, measures:
        - Average motion velocity
        - Motion variance (jitter)
        - Region success rates
        
        Returns:
            Dict with motion_class, texture_class, best_regions
        """
        # Quick estimation first (no tracking needed)
        quick_class = self._estimate_motion_quick()
        
        # For low/medium motion and no robust mode, skip expensive full probe
        if quick_class != 'HIGH' and not self.robust_mode:
            print(f"AutoSolve: Quick motion estimate: {quick_class} (skipping full probe)")
            # Set motion_class for per-clip learning
            self.motion_class = quick_class
            quick_result = {
                'success': True,
                'motion_class': quick_class,
                'texture_class': 'MEDIUM',
                'best_regions': ['center', 'mid-left', 'mid-right', 'bottom-center'],
                'velocities': {},
                'region_success': {},
                'probe_type': 'quick_estimate'
            }
            self.cached_motion_probe = quick_result.copy()
            
            # Extract visual features for feature density
            try:
                if hasattr(self, 'feature_extractor'):
                    self.feature_extractor.extract_all(tracking_data=quick_result)
                    self.feature_extractor.features.motion_class = self.motion_class
                    print(f"AutoSolve: Visual features extracted (quick path)")
            except Exception as e:
                print(f"AutoSolve: Visual feature extraction skipped: {e}")
                
            return quick_result
        
        print(f"AutoSolve: Running full motion probe (quick estimate: {quick_class})")
        
        result = {
            'success': False,
            'motion_class': quick_class,  # Use quick estimate as baseline
            'texture_class': 'MEDIUM',
            'best_regions': [],
            'velocities': {},
            'region_success': {},
            'probe_type': 'full_probe'
        }
        
        # Save current frame
        original_frame = bpy.context.scene.frame_current
        probe_start = self.clip.frame_start + (self.clip.frame_duration // 4)  # Start at 25%
        
        # NOTE: Don't clear tracks here - let existing tracks be analyzed if any
        # This prevents wasting user-placed markers
        
        # Track existing tracks so we don't delete them later
        existing_tracks = set(t.name for t in self.tracking.tracks)

        # Probe settings: very aggressive to catch motion
        probe_settings = {
            'pattern_size': 21,
            'search_size': 121,  # Large search for testing
            'correlation': 0.55,  # Low correlation to not lose tracks
            'threshold': 0.15,
        }
        
        # Place 1 probe marker per region
        regions = REGIONS.copy()
        import random
        random.shuffle(regions)
        
        probe_count = 0
        for region in regions[:5]:  # Only probe 5 regions for speed
            bpy.context.scene.frame_set(probe_start)
            
            # Apply probe settings
            self.current_settings = probe_settings.copy()
            self.configure_settings()
            
            # Try to detect 1 marker in this region
            detected = self.detect_in_region(region, count=1)
            if detected > 0:
                probe_count += 1
        
        if probe_count < 3:
            print(f"AutoSolve: Probe failed - only {probe_count} markers placed")

            # Select only probe tracks for deletion
            for track in self.tracking.tracks:
                if track.name not in existing_tracks:
                    track.select = True
                else:
                    track.select = False
            try:
                self._run_ops(bpy.ops.clip.delete_track)
            except Exception:
                pass

            return result
        
        # Track forward for 20 frames
        print(f"AutoSolve: Probe tracking {probe_count} markers for 20 frames...")
        self.select_all_tracks()
        
        probe_frames = min(20, self.clip.frame_duration // 4)
        bpy.context.scene.frame_set(probe_start)
        
        for i in range(probe_frames):
            self.track_frame(backwards=False)
            bpy.context.scene.frame_set(probe_start + i + 1)
        
        # Analyze probe results
        velocities = []
        jitters = []
        region_success = {}
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 3:
                continue
            
            markers_sorted = sorted(markers, key=lambda m: m.frame)
            
            # Calculate region early to avoid referencing stale/leaked loop variables
            avg_x = sum(m.co.x for m in markers_sorted) / len(markers_sorted)
            avg_y = sum(m.co.y for m in markers_sorted) / len(markers_sorted)
            region = get_region(avg_x, avg_y)
            
            # Calculate velocity
            total_displacement = 0
            for i in range(1, len(markers_sorted)):
                dx = markers_sorted[i].co.x - markers_sorted[i-1].co.x
                dy = markers_sorted[i].co.y - markers_sorted[i-1].co.y
                total_displacement += (dx**2 + dy**2) ** 0.5
            
            avg_velocity = total_displacement / len(markers_sorted)
            velocities.append(avg_velocity)
            
            # Calculate jitter (variance in velocity)
            if len(markers_sorted) > 3:
                frame_velocities = []
                for i in range(1, len(markers_sorted)):
                    dx = markers_sorted[i].co.x - markers_sorted[i-1].co.x
                    dy = markers_sorted[i].co.y - markers_sorted[i-1].co.y
                    frame_velocities.append((dx**2 + dy**2) ** 0.5)
                
                if frame_velocities:
                    mean_v = sum(frame_velocities) / len(frame_velocities)
                    variance = sum((v - mean_v)**2 for v in frame_velocities) / len(frame_velocities)
                    jitter = variance ** 0.5
                    jitters.append(jitter)
                    
                    # Store per-region jitter (key for water/wave detection)
                    if region not in region_success:
                        region_success[region] = {'total': 0, 'success': 0, 'jitters': [], 'velocities': []}
                    region_success[region].setdefault('jitters', []).append(jitter)
                    region_success[region].setdefault('velocities', []).append(avg_velocity)
            
            # Track region success
            lifespan = len(markers_sorted)
            if region not in region_success:
                region_success[region] = {'total': 0, 'success': 0, 'jitters': [], 'velocities': []}
            region_success[region]['total'] += 1
            if lifespan >= probe_frames * 0.7:  # 70% survival
                region_success[region]['success'] += 1
        
        # Classify motion
        if velocities:
            avg_motion = sum(velocities) / len(velocities)
            if avg_motion > 0.03:
                result['motion_class'] = 'HIGH'
            elif avg_motion > 0.01:
                result['motion_class'] = 'MEDIUM'
            else:
                result['motion_class'] = 'LOW'
            
            result['velocities'] = {
                'avg': avg_motion,
                'max': max(velocities) if velocities else 0,
            }
        
        # Classify texture (based on how many features we could detect)
        if probe_count >= 4:
            result['texture_class'] = 'HIGH'
        elif probe_count >= 2:
            result['texture_class'] = 'MEDIUM'
        else:
            result['texture_class'] = 'LOW'
        
        # Find best regions
        best_regions = []
        for region, stats in region_success.items():
            if stats['total'] > 0:
                rate = stats['success'] / stats['total']
                if rate >= 0.5:
                    best_regions.append(region)
        
        result['best_regions'] = best_regions if best_regions else ['center']
        result['region_success'] = region_success
        result['success'] = True
        
        # Cache the probe results for session recording
        self.cached_motion_probe = result.copy()
        
        # Set motion_class for per-clip learning and sub-classification
        self.motion_class = result.get('motion_class', 'MEDIUM')
        print(f"AutoSolve: Motion class set to {self.motion_class}")
        
        # Extract visual features for ML training data
        try:
            if hasattr(self, 'feature_extractor'):
                # Extract all visual features for ML training
                override = self._get_context_override()
                if override:
                    with bpy.context.temp_override(**override):
                        self.feature_extractor.extract_all(tracking_data=result)
                else:
                    self.feature_extractor.extract_all(tracking_data=result)
                # Sync motion class to feature extractor
                self.feature_extractor.features.motion_class = self.motion_class
                print(f"AutoSolve: Visual features extracted")
        except Exception as e:
            print(f"AutoSolve: Visual feature extraction skipped: {e}")
        
        # Clear probe tracks (only the new ones)
        for track in self.tracking.tracks:
            if track.name not in existing_tracks:
                track.select = True
            else:
                track.select = False
        try:
            self._run_ops(bpy.ops.clip.delete_track)
        except Exception:
            pass
        
        # Restore frame
        bpy.context.scene.frame_set(original_frame)
        
        return result
