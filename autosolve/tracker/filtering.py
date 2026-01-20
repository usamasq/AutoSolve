# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
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
from collections import defaultdict

import bpy
from mathutils import Vector


class FilteringMixin:
    """Mixin providing track filtering methods for SmartTracker."""
    
    # Flag to preserve short tracks during healing phase
    _healing_pending: bool = False
    
    def mark_healing_pending(self, pending: bool = True):
        """
        Mark that healing phase is pending.
        
        When True, short tracks won't be filtered - they may be healing candidates
        (track segments that can be joined to form complete tracks).
        """
        self._healing_pending = pending
        if pending:
            print("AutoSolve: Preserving short tracks for healing phase")
    
    def filter_motion_spikes(self, threshold: float = 5.0, action: str = 'MUTE') -> int:
        """
        Use Blender's built-in filter_tracks to detect and handle motion spikes.
        
        This catches:
        - Sudden dislocations (track jumps to different feature)
        - Gradual drift (track slowly moves off target)
        - Erratic motion (jittery tracking that doesn't match scene motion)
        
        Args:
            threshold: Sensitivity (lower = more aggressive). Default 5.0
            action: 'MUTE' to mute bad markers, 'DELETE' to remove bad tracks
                    
        Returns:
            Number of tracks affected
        """
        current = len(self.tracking.tracks)
        if current < self.SAFE_MIN_TRACKS:
            return 0
        
        # Count tracks before
        tracks_before = sum(1 for t in self.tracking.tracks 
                          if len([m for m in t.markers if not m.mute]) >= 2)
        
        # Select all tracks for filtering
        self.select_all_tracks()
        
        try:
            # Blender's filter_tracks detects "weirdly looking spikes in motion curves"
            self._run_ops(bpy.ops.clip.filter_tracks, track_threshold=threshold)
            
            # Count tracks after (filter_tracks mutes problem markers)
            tracks_after = sum(1 for t in self.tracking.tracks 
                              if len([m for m in t.markers if not m.mute]) >= 2)
            
            affected = tracks_before - tracks_after
            
            if affected > 0:
                print(f"AutoSolve: Blender filter_tracks detected {affected} drifted/spiked tracks "
                      f"(threshold={threshold})")
                
                # Record for ML training
                if hasattr(self, 'recorder') and self.recorder:
                    current_frame = bpy.context.scene.frame_current
                    for track in self.tracking.tracks:
                        active_markers = [m for m in track.markers if not m.mute]
                        if len(active_markers) < 2 and len(list(track.markers)) >= 2:
                            # This track was affected
                            marker = list(track.markers)[0] if list(track.markers) else None
                            if marker:
                                self.recorder.record_track_failure(
                                    track.name, current_frame,
                                    marker.co.x, marker.co.y, "MOTION_SPIKE"
                                )
            
            return affected
            
        except Exception as e:
            print(f"AutoSolve: filter_tracks failed: {e}")
            return 0
    
    def clean_bad_segments(self, max_error: float = 3.0, min_frames: int = 3) -> int:
        """
        Remove only bad SEGMENTS of tracks, preserving good portions.
        
        Uses Blender's clean_tracks with DELETE_SEGMENTS action to:
        - Remove segments with high reprojection error
        - Keep the good parts of tracks intact
        - Create gaps that can be filled by heal_tracks()
        
        This is much better than deleting entire tracks!
        
        Args:
            max_error: Maximum reprojection error threshold (default 3.0)
            min_frames: Minimum consecutive good frames to keep (default 3)
            
        Returns:
            Number of tracks that had segments removed
        """
        current = len(self.tracking.tracks)
        if current < self.SAFE_MIN_TRACKS:
            return 0
        
        # Count total markers before
        markers_before = sum(
            len([m for m in t.markers if not m.mute]) 
            for t in self.tracking.tracks
        )
        
        self.select_all_tracks()
        
        try:
            # DELETE_SEGMENTS removes only bad portions, keeping good parts
            self._run_ops(
                bpy.ops.clip.clean_tracks, 
                frames=min_frames, 
                error=max_error, 
                action='DELETE_SEGMENTS'
            )
            
            # Count markers after
            markers_after = sum(
                len([m for m in t.markers if not m.mute]) 
                for t in self.tracking.tracks
            )
            
            removed = markers_before - markers_after
            
            if removed > 0:
                # Count how many tracks were affected (have fewer markers now)
                affected_tracks = 0
                for track in self.tracking.tracks:
                    markers = [m for m in track.markers if not m.mute]
                    all_markers = list(track.markers)
                    if len(markers) < len(all_markers):
                        affected_tracks += 1
                
                print(f"AutoSolve: Cleaned {removed} bad markers from {affected_tracks} tracks "
                      f"(max_error={max_error}px)")
                
                # Record for ML training
                if hasattr(self, 'recorder') and self.recorder:
                    current_frame = bpy.context.scene.frame_current
                    for track in self.tracking.tracks:
                        active = [m for m in track.markers if not m.mute]
                        all_m = list(track.markers)
                        if len(active) < len(all_m) and len(active) >= min_frames:
                            # This track had segments removed but wasn't deleted
                            self.recorder.record_track_failure(
                                track.name, current_frame,
                                active[0].co.x if active else 0,
                                active[0].co.y if active else 0,
                                "SEGMENT_CLEANED"
                            )
                
                return affected_tracks
            
            return 0
            
        except Exception as e:
            print(f"AutoSolve: clean_bad_segments failed: {e}")
            return 0
    
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
            # Optimized: Single pass to find min/max frames
            # Avoids O(M log M) sorting and intermediate list creation
            min_frame = float('inf')
            max_frame = float('-inf')
            min_marker = None
            max_marker = None
            active_count = 0

            for m in track.markers:
                if m.mute:
                    continue
                active_count += 1
                if m.frame < min_frame:
                    min_frame = m.frame
                    min_marker = m
                if m.frame > max_frame:
                    max_frame = m.frame
                    max_marker = m

            if active_count < 2 or not min_marker:
                continue
            
            displacement = (Vector(max_marker.co) - Vector(min_marker.co)).length
            duration = abs(max_frame - min_frame)
            
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
        
        if min_dist_norm <= 0:
            return

        # Pre-calculate squared threshold to avoid sqrt in loops
        min_dist_norm_sq = min_dist_norm * min_dist_norm

        # Collect track positions and group by region
        tracks_by_region = {}
        current_frame = bpy.context.scene.frame_current
        clip_frame = self.scene_to_clip_frame(current_frame)
        
        for track in self.tracking.tracks:
            marker = track.markers.find_frame(clip_frame)
            if not marker:
                markers = [m for m in track.markers if not m.mute]
                if markers:
                    marker = markers[0]
            
            if marker:
                pos = (marker.co.x, marker.co.y)
                region = self._get_region_for_pos(pos[0], pos[1])

                if region not in tracks_by_region:
                    tracks_by_region[region] = []
                tracks_by_region[region].append((track.name, pos))
        
        # Find duplicates to remove
        to_delete = set()
        
        # Only check saturated regions
        for region in saturated_regions:
            if region not in tracks_by_region:
                continue
            
            region_tracks = tracks_by_region[region]
            
            # Use spatial hashing for O(N) neighbor search instead of O(N^2)
            grid = defaultdict(list)
            cell_size = min_dist_norm

            for name, pos in region_tracks:
                cx = math.floor(pos[0] / cell_size)
                cy = math.floor(pos[1] / cell_size)
                grid[(cx, cy)].append((name, pos))

            def check_and_mark(n1, p1, n2, p2):
                """Helper to check distance and mark track for deletion."""
                if n1 in to_delete or n2 in to_delete:
                    return

                # Optimized: Compare squared distances to avoid expensive sqrt()
                d_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                if d_sq < min_dist_norm_sq:
                    t1 = self.tracking.tracks.get(n1)
                    t2 = self.tracking.tracks.get(n2)

                    if t1 and t2:
                        l1 = len([m for m in t1.markers if not m.mute])
                        l2 = len([m for m in t2.markers if not m.mute])
                        to_delete.add(n1 if l1 < l2 else n2)

            # Check neighbors
            # Sort keys for deterministic iteration order
            for (cx, cy), entries in sorted(grid.items()):
                # 1. Check pairs within the same cell
                for i in range(len(entries)):
                    name1, pos1 = entries[i]
                    for j in range(i + 1, len(entries)):
                        name2, pos2 = entries[j]
                        check_and_mark(name1, pos1, name2, pos2)

                # 2. Check neighbor cells (East, South, SE, SW)
                neighbor_offsets = [(1, 0), (0, 1), (1, 1), (-1, 1)]
                for dx, dy in neighbor_offsets:
                    neighbor_key = (cx + dx, cy + dy)
                    if neighbor_key in grid:
                        neighbor_entries = grid[neighbor_key]
                        for name1, pos1 in entries:
                            for name2, pos2 in neighbor_entries:
                                check_and_mark(name1, pos1, name2, pos2)
        
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
        """Filter high error tracks while preserving keyframe coverage.
        
        Blender's solver requires at least 8 tracks visible on BOTH keyframes.
        This method respects that constraint in addition to ABSOLUTE_MIN_TRACKS.
        """
        if not self.tracking.reconstruction.is_valid:
            return
        
        current = len(self.tracking.tracks)
        if current < self.SAFE_MIN_TRACKS:
            return
        
        # Get keyframe positions from tracking camera settings
        camera = self.clip.tracking.camera if hasattr(self.clip.tracking, 'camera') else None
        keyframe_a = getattr(camera, 'keyframe_a', 1) if camera else 1
        keyframe_b = getattr(camera, 'keyframe_b', 30) if camera else self.clip.frame_duration
        
        # Helper to check if track has active markers on both keyframes
        def covers_keyframes(track):
            """Check if track has non-muted markers on BOTH keyframes."""
            has_a = False
            has_b = False
            for m in track.markers:
                if m.mute:
                    continue
                if m.frame == keyframe_a:
                    has_a = True
                if m.frame == keyframe_b:
                    has_b = True
                if has_a and has_b:
                    return True
            return False
        
        # Count tracks covering keyframes before filtering
        keyframe_tracks = [t.name for t in self.tracking.tracks if covers_keyframes(t)]
        keyframe_count = len(keyframe_tracks)
        
        to_delete = [t.name for t in self.tracking.tracks
                    if t.has_bundle and t.average_error > max_error]
        
        # Calculate max deletable considering BOTH total count and keyframe coverage
        max_by_total = current - self.ABSOLUTE_MIN_TRACKS
        
        # Separate candidates into keyframe-critical and non-critical
        critical_candidates = [n for n in to_delete if n in keyframe_tracks]
        safe_candidates = [n for n in to_delete if n not in keyframe_tracks]
        
        # Determine how many critical (keyframe) tracks we can afford to lose
        # Must keep at least 8 keyframe tracks
        max_critical_loss = max(0, keyframe_count - 8)
        
        # If we have more critical candidates than we can lose, sort them by error and keep the worst
        if len(critical_candidates) > max_critical_loss:
            # We need error values to sort
            critical_errors = []
            for t in self.tracking.tracks:
                if t.name in critical_candidates and t.has_bundle:
                    critical_errors.append((t.name, t.average_error))

            # Sort by error descending (worst first)
            critical_errors.sort(key=lambda x: x[1], reverse=True)

            # Keep only the worst 'max_critical_loss' tracks
            critical_candidates = [n for n, _ in critical_errors[:max_critical_loss]]

        # Combine lists: all safe candidates + allowed critical candidates
        final_to_delete = safe_candidates + critical_candidates
        
        # Finally, apply the global minimum track count limit
        if len(final_to_delete) > max_by_total:
             # Sort combined list by error
            final_errors = []
            for t in self.tracking.tracks:
                if t.name in final_to_delete and t.has_bundle:
                    final_errors.append((t.name, t.average_error))

            final_errors.sort(key=lambda x: x[1], reverse=True)
            final_to_delete = [n for n, _ in final_errors[:max_by_total]]

        to_delete = final_to_delete
        
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
    
    def average_clustered_tracks(self, proximity_threshold_px: int = 15) -> int:
        """
        Average nearby tracks to create stable anchor points.
        
        Finds clusters of 2-3 tracks within the proximity threshold and
        averages them into single high-quality tracks. This reduces noise
        and improves solve stability.
        
        Args:
            proximity_threshold_px: Minimum distance in pixels (default: 15)
            
        Returns:
            Number of averaged anchor tracks created
        """
        current = len(self.tracking.tracks)
        if current < self.SAFE_MIN_TRACKS:
            return 0
        
        try:
            from .averaging import TrackAverager
            
            # Convert pixel threshold to normalized coords
            width = self.clip.size[0]
            proximity_norm = proximity_threshold_px / width
            
            averager = TrackAverager(proximity_threshold=proximity_norm)
            created = averager.create_anchor_tracks(self.tracking, keep_originals=False)
            
            return created
            
        except Exception as e:
            print(f"AutoSolve: average_clustered_tracks failed: {e}")
            return 0
    
    def cleanup_tracks(self, min_frames: int = 5, spike_multiplier: float = 8.0,
                       jitter_threshold: float = 0.6, coherence_threshold: float = 0.4):
        """
        Unified track cleanup - all filters in one pass.
        
        Combines short track filtering, velocity spike removal,
        and non-rigid motion filtering.
        """
        initial = len(self.tracking.tracks)
        
        # 1. Filter short tracks (skip if healing is pending)
        if getattr(self, '_healing_pending', False):
            print("AutoSolve: Skipping short track filter (healing pending)")
        else:
            self.filter_short_tracks(min_frames=min_frames)
        
        # 2. Filter velocity spikes
        self.filter_spikes(limit_multiplier=spike_multiplier)
        
        # 3. Filter non-rigid motion (for outdoor/drone footage)
        if self.footage_type in ['DRONE', 'OUTDOOR', 'AUTO']:
            self.filter_non_rigid_motion(
                jitter_threshold=jitter_threshold,
                coherence_threshold=coherence_threshold
            )
        
        # 4. Average clustered tracks → creates stable anchors from nearby tracks
        self.average_clustered_tracks(proximity_threshold_px=15)
        
        # 5. Deduplicate (coverage-aware)
        self.deduplicate_tracks(min_distance_px=30)
        
        final = len(self.tracking.tracks)
        removed = initial - final
        print(f"AutoSolve: Cleanup complete: {initial} → {final} tracks ({removed} removed)")
        
        return removed
