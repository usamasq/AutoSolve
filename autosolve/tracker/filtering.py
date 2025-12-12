# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Filtering mixin for SmartTracker.

Contains methods for filtering tracks based on various quality criteria:
- Short track filtering
- Velocity spike filtering
- Track deduplication
- Non-rigid motion filtering
- High error track filtering
"""

import math
from typing import Dict, List

import bpy
from mathutils import Vector


class FilteringMixin:
    """Mixin providing track filtering methods for SmartTracker."""
    
    def filter_short_tracks(self, min_frames: int = 5):
        """Filter short tracks with safeguards."""
        current = len(self.tracking.tracks)
        
        survivors = sum(1 for t in self.tracking.tracks
                       if len([m for m in t.markers if not m.mute]) >= min_frames)
        
        if survivors < self.SAFE_MIN_TRACKS:
            print(f"AutoSolve: Skipping filter (would leave {survivors})")
            return
        
        self.select_all_tracks()
        try:
            self._run_ops(bpy.ops.clip.clean_tracks, frames=min_frames, error=999, action='DELETE_TRACK')
        except TypeError:
            self._run_ops(bpy.ops.clip.clean_tracks, frames=min_frames, error=999, action='DELETE')
        
        print(f"AutoSolve: After filter: {len(self.tracking.tracks)} tracks")
    
    def filter_spikes(self, limit_multiplier: float = 8.0):
        """Filter velocity outliers."""
        current = len(self.tracking.tracks)
        
        if current < self.SAFE_MIN_TRACKS:
            return
        
        track_speeds = {}
        total_speed = 0.0
        count = 0
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            markers.sort(key=lambda x: x.frame)
            displacement = (Vector(markers[-1].co) - Vector(markers[0].co)).length
            duration = abs(markers[-1].frame - markers[0].frame)
            
            if duration > 0:
                speed = displacement / duration
                track_speeds[track.name] = speed
                total_speed += speed
                count += 1
        
        if count == 0:
            return
        
        avg = max(total_speed / count, 0.001)
        limit = avg * limit_multiplier
        
        to_delete = [n for n, s in track_speeds.items() if s > limit]
        max_del = min(len(to_delete), current - self.ABSOLUTE_MIN_TRACKS)
        
        if max_del <= 0:
            return
        
        sorted_tracks = sorted(track_speeds.items(), key=lambda x: x[1], reverse=True)
        to_delete = set(n for n, _ in sorted_tracks[:max_del] if track_speeds[n] > limit)
        
        for track in self.tracking.tracks:
            track.select = track.name in to_delete
        
        if to_delete:
            # Record track failures for ML training before deletion
            if hasattr(self, 'recorder') and self.recorder:
                current_frame = bpy.context.scene.frame_current
                for track in self.tracking.tracks:
                    if track.name in to_delete:
                        markers = [m for m in track.markers if not m.mute]
                        if markers:
                            marker = markers[0]  # Use first marker position
                            self.recorder.record_track_failure(
                                track.name, current_frame, 
                                marker.co.x, marker.co.y, "VELOCITY_SPIKE"
                            )
            
            print(f"AutoSolve: Removing {len(to_delete)} outliers")
            try:
                self._run_ops(bpy.ops.clip.delete_track)
            except:
                pass
            self.select_all_tracks()
    
    def deduplicate_tracks(self, min_distance_px: int = 30):
        """
        Coverage-aware track deduplication.
        
        Only removes close tracks when overall coverage is good.
        Keeps duplicates in sparse regions to maintain solve quality.
        """
        current = len(self.tracking.tracks)
        if current < self.SAFE_MIN_TRACKS:
            return
        
        # Get current coverage assessment
        self.coverage_analyzer.analyze_tracking(self.tracking)
        summary = self.coverage_analyzer.get_coverage_summary()
        regions_with_tracks = summary.get('regions_with_tracks', 0)
        region_counts = summary.get('region_counts', {})
        
        # Only deduplicate if we have good coverage (6+ of 9 regions)
        if regions_with_tracks < 6:
            print(f"AutoSolve: Skipping dedup (only {regions_with_tracks}/9 regions have tracks)")
            return
        
        # Find saturated regions (> 35% of total tracks)
        total = max(sum(region_counts.values()), 1)
        saturated_regions = {r for r, c in region_counts.items() if c / total > 0.35}
        
        if not saturated_regions:
            return
        
        print(f"AutoSolve: Deduplicating in saturated regions: {saturated_regions}")
        
        # Convert min_distance to normalized coords
        width = self.clip.size[0]
        min_dist_norm = 15 / width
        
        # Collect track positions
        track_positions = {}
        current_frame = bpy.context.scene.frame_current
        clip_frame = self.scene_to_clip_frame(current_frame)
        
        for track in self.tracking.tracks:
            marker = track.markers.find_frame(clip_frame)
            if not marker:
                markers = [m for m in track.markers if not m.mute]
                if markers:
                    marker = markers[0]
            
            if marker:
                track_positions[track.name] = (marker.co.x, marker.co.y)
        
        # Find duplicates to remove
        to_delete = set()
        track_list = list(track_positions.items())
        
        for i, (name1, pos1) in enumerate(track_list):
            if name1 in to_delete:
                continue
            
            region1 = self._get_region_for_pos(pos1[0], pos1[1])
            if region1 not in saturated_regions:
                continue
            
            for j, (name2, pos2) in enumerate(track_list[i+1:], i+1):
                if name2 in to_delete:
                    continue
                
                region2 = self._get_region_for_pos(pos2[0], pos2[1])
                if region2 != region1:
                    continue
                
                dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) ** 0.5
                if dist < min_dist_norm:
                    track1 = self.tracking.tracks.get(name1)
                    track2 = self.tracking.tracks.get(name2)
                    
                    if track1 and track2:
                        len1 = len([m for m in track1.markers if not m.mute])
                        len2 = len([m for m in track2.markers if not m.mute])
                        to_delete.add(name1 if len1 < len2 else name2)
        
        # Safety check
        max_delete = min(len(to_delete), current // 10, current - self.SAFE_MIN_TRACKS)
        if max_delete <= 0:
            return
        to_delete = set(list(to_delete)[:max_delete])
        
        if not to_delete:
            return
        
        for track in self.tracking.tracks:
            track.select = track.name in to_delete
        
        print(f"AutoSolve: Removing {len(to_delete)} duplicate tracks")
        try:
            self._run_ops(bpy.ops.clip.delete_track)
        except:
            pass
        self.select_all_tracks()
    
    def _get_region_for_pos(self, x: float, y: float) -> str:
        """Get region name for normalized coordinates."""
        col = 0 if x < 0.33 else (1 if x < 0.66 else 2)
        row = 2 if y < 0.33 else (1 if y < 0.66 else 0)
        regions = [
            ['top-left', 'top-center', 'top-right'],
            ['mid-left', 'center', 'mid-right'],
            ['bottom-left', 'bottom-center', 'bottom-right']
        ]
        return regions[row][col]
    
    def filter_non_rigid_motion(self, jitter_threshold: float = 0.6, coherence_threshold: float = 0.4):
        """
        Filter tracks on non-rigid moving objects like waves, water, foliage.
        
        Detects tracks that have high jitter or move differently from camera motion.
        """
        current = len(self.tracking.tracks)
        if current < self.SAFE_MIN_TRACKS:
            return
        
        track_data = {}
        motion_vectors = []
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 5:
                continue
            
            markers.sort(key=lambda x: x.frame)
            
            velocities = []
            for i in range(1, len(markers)):
                dx = markers[i].co.x - markers[i-1].co.x
                dy = markers[i].co.y - markers[i-1].co.y
                velocities.append((dx, dy))
            
            if not velocities:
                continue
            
            avg_dx = sum(v[0] for v in velocities) / len(velocities)
            avg_dy = sum(v[1] for v in velocities) / len(velocities)
            motion_vec = (avg_dx, avg_dy)
            motion_vectors.append(motion_vec)
            
            jitter = 0.0
            if len(velocities) >= 2:
                vel_changes = []
                for i in range(1, len(velocities)):
                    change_x = abs(velocities[i][0] - velocities[i-1][0])
                    change_y = abs(velocities[i][1] - velocities[i-1][1])
                    vel_changes.append((change_x**2 + change_y**2)**0.5)
                
                if vel_changes:
                    avg_mag = (avg_dx**2 + avg_dy**2)**0.5
                    if avg_mag > 0.0001:
                        jitter = (sum(vel_changes) / len(vel_changes)) / avg_mag
                    else:
                        jitter = sum(vel_changes) / len(vel_changes) * 100
            
            track_data[track.name] = {
                'motion_vec': motion_vec,
                'jitter': jitter,
                'coherence': 0.0,
            }
        
        if len(motion_vectors) < 5:
            return
        
        # Compute median camera motion
        angles = [math.atan2(v[1], v[0]) for v in motion_vectors]
        angles.sort()
        median_angle = angles[len(angles) // 2]
        
        magnitudes = [(v[0]**2 + v[1]**2)**0.5 for v in motion_vectors]
        magnitudes.sort()
        median_mag = magnitudes[len(magnitudes) // 2]
        
        camera_motion = (math.cos(median_angle) * median_mag, math.sin(median_angle) * median_mag)
        camera_mag = (camera_motion[0]**2 + camera_motion[1]**2)**0.5
        
        # Compute coherence for each track
        for name, data in track_data.items():
            mv = data['motion_vec']
            mv_mag = (mv[0]**2 + mv[1]**2)**0.5
            
            if camera_mag < 0.0001 or mv_mag < 0.0001:
                data['coherence'] = 1.0
            else:
                dot = mv[0] * camera_motion[0] + mv[1] * camera_motion[1]
                coherence = dot / (mv_mag * camera_mag)
                data['coherence'] = max(0.0, coherence)
        
        # Identify non-rigid tracks
        # Require BOTH jitter AND incoherence to avoid over-aggressive filtering
        non_rigid = []
        for name, data in track_data.items():
            is_jittery = data['jitter'] > jitter_threshold
            is_incoherent = data['coherence'] < coherence_threshold
            
            # Changed from OR to AND - only remove truly non-rigid tracks
            if is_jittery and is_incoherent:
                non_rigid.append((name, data['jitter'], data['coherence']))
        
        # Cap removal at 30% of tracks AND ensure we keep at least SAFE_MIN_TRACKS
        max_can_delete = min(
            current - self.SAFE_MIN_TRACKS,  # Floor: keep 20 tracks minimum
            int(current * 0.30)               # Cap: max 30% removal per pass
        )
        if len(non_rigid) > max_can_delete:
            non_rigid.sort(key=lambda x: (x[1] - x[2]), reverse=True)
            non_rigid = non_rigid[:max_can_delete]
        
        to_delete = set(n for n, _, _ in non_rigid)
        
        for track in self.tracking.tracks:
            track.select = track.name in to_delete
        
        if to_delete:
            # Record track failures for ML training before deletion
            if hasattr(self, 'recorder') and self.recorder:
                current_frame = bpy.context.scene.frame_current
                for track in self.tracking.tracks:
                    if track.name in to_delete:
                        markers = [m for m in track.markers if not m.mute]
                        if markers:
                            marker = markers[0]
                            self.recorder.record_track_failure(
                                track.name, current_frame,
                                marker.co.x, marker.co.y, "NON_RIGID"
                            )
            
            print(f"AutoSolve: Removing {len(to_delete)} non-rigid tracks")
            try:
                self._run_ops(bpy.ops.clip.delete_track)
            except:
                pass
            self.select_all_tracks()
    
    def filter_high_error(self, max_error: float = 3.0):
        """Filter high error tracks."""
        if not self.tracking.reconstruction.is_valid:
            return
        
        current = len(self.tracking.tracks)
        if current < self.SAFE_MIN_TRACKS:
            return
        
        to_delete = [t.name for t in self.tracking.tracks
                    if t.has_bundle and t.average_error > max_error]
        
        max_can = current - self.ABSOLUTE_MIN_TRACKS
        if len(to_delete) > max_can:
            errors = [(t.name, t.average_error) for t in self.tracking.tracks if t.has_bundle]
            errors.sort(key=lambda x: x[1], reverse=True)
            to_delete = [n for n, _ in errors[:max_can]]
        
        for track in self.tracking.tracks:
            track.select = track.name in to_delete
        
        if to_delete:
            # Record track failures for ML training before deletion
            if hasattr(self, 'recorder') and self.recorder:
                current_frame = bpy.context.scene.frame_current
                for track in self.tracking.tracks:
                    if track.name in to_delete:
                        markers = [m for m in track.markers if not m.mute]
                        if markers:
                            marker = markers[0]
                            self.recorder.record_track_failure(
                                track.name, current_frame,
                                marker.co.x, marker.co.y, "HIGH_ERROR"
                            )
            
            print(f"AutoSolve: Removing {len(to_delete)} high-error tracks")
            try:
                self._run_ops(bpy.ops.clip.delete_track)
            except:
                pass
    
    def cleanup_tracks(self, min_frames: int = 5, spike_multiplier: float = 8.0,
                       jitter_threshold: float = 0.6, coherence_threshold: float = 0.4):
        """
        Unified track cleanup - all filters in one pass.
        
        Combines short track filtering, velocity spike removal,
        and non-rigid motion filtering.
        """
        initial = len(self.tracking.tracks)
        
        # 1. Filter short tracks
        self.filter_short_tracks(min_frames=min_frames)
        
        # 2. Filter velocity spikes
        self.filter_spikes(limit_multiplier=spike_multiplier)
        
        # 3. Filter non-rigid motion (for outdoor/drone footage)
        if self.footage_type in ['DRONE', 'OUTDOOR', 'AUTO']:
            self.filter_non_rigid_motion(
                jitter_threshold=jitter_threshold,
                coherence_threshold=coherence_threshold
            )
        
        # 4. Deduplicate (coverage-aware)
        self.deduplicate_tracks(min_distance_px=30)
        
        final = len(self.tracking.tracks)
        removed = initial - final
        print(f"AutoSolve: Cleanup complete: {initial} → {final} tracks ({removed} removed)")
        
        return removed
