# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
CleanupMixin for SmartTracker.

Manages clustering cleanup, lost track extension, track healing integration,
worst-performing track removal, solve refinement iterations, user track identification,
struggling track refinement, good track preservation, and optimal keyframe selection.
"""

import bpy
from typing import List, Dict

from .analyzers import CoverageAnalyzer
from .utils import get_region


class CleanupMixin:
    """Mixin for SmartTracker track cleanup, healing, and refinement operations."""

    def remove_clustered_tracks(self) -> int:
        """
        Remove tracks from over-represented regions to improve balance.
        """
        clustered = self.coverage_analyzer.get_clustered_regions()
        if not clustered:
            return 0
        
        summary = self.coverage_analyzer.get_coverage_summary()
        total = summary['total_tracks']
        target_max = int(total * CoverageAnalyzer.MAX_TRACKS_PER_REGION_PERCENT)
        
        removed = 0
        for region in clustered:
            region_count = summary['region_counts'].get(region, 0)
            excess = region_count - target_max
            
            if excess <= 0:
                continue
            
            tracks_in_region = []
            for track in self.tracking.tracks:
                markers = [m for m in track.markers if not m.mute]
                if len(markers) < 2:
                    continue
                
                avg_x = sum(m.co.x for m in markers) / len(markers)
                avg_y = sum(m.co.y for m in markers) / len(markers)
                if get_region(avg_x, avg_y) == region:
                    lifespan = len(markers)
                    tracks_in_region.append((track.name, lifespan))
            
            tracks_in_region.sort(key=lambda x: x[1])
            
            to_remove = set(name for name, _ in tracks_in_region[:excess])
            for track in self.tracking.tracks:
                track.select = track.name in to_remove
            
            if to_remove:
                try:
                    self._run_ops(bpy.ops.clip.delete_track)
                    removed += len(to_remove)
                    print(f"AutoSolve: Removed {len(to_remove)} excess tracks from {region}")
                except Exception:
                    pass
        
        return removed

    def extend_lost_tracks(self, min_extension: int = 10) -> int:
        """
        Extend tracks that stopped tracking before the clip ends.
        """
        if not self.clip or not self.tracking:
            return 0
        
        clip_duration = self.clip.frame_duration
        
        lost_tracks = []
        
        for track in self.tracking.tracks:
            try:
                markers = [m for m in track.markers if not m.mute]
                if len(markers) < 3:
                    continue
                
                markers_sorted = sorted(markers, key=lambda m: m.frame)
                first_frame = markers_sorted[0].frame
                last_frame = markers_sorted[-1].frame
                
                can_extend_forward = last_frame < (clip_duration - min_extension)
                can_extend_backward = first_frame > (1 + min_extension)
                
                if can_extend_forward or can_extend_backward:
                    lost_tracks.append({
                        'name': track.name,
                        'first_frame': first_frame,
                        'last_frame': last_frame,
                        'extend_forward': can_extend_forward,
                        'extend_backward': can_extend_backward,
                        'lifespan': last_frame - first_frame
                    })
            except (ReferenceError, AttributeError):
                continue
        
        if not lost_tracks:
            print("AutoSolve: No lost tracks to extend")
            return 0
        
        lost_tracks.sort(key=lambda t: t['lifespan'])
        lost_tracks = lost_tracks[:20]
        
        print(f"AutoSolve: Extending {len(lost_tracks)} tracks that stopped early...")
        
        orig_correlation = self.current_settings.get('correlation', 0.7)
        orig_search = self.current_settings.get('search_size', 71)
        
        try:
            if hasattr(self.settings, 'default_correlation_min'):
                self.settings.default_correlation_min = max(0.4, orig_correlation - 0.2)
            if hasattr(self.settings, 'default_search_size'):
                self.settings.default_search_size = int(orig_search * 1.3)
        except (ReferenceError, AttributeError):
            pass
        
        extended = 0
        current_frame = bpy.context.scene.frame_current
        
        try:
            for track_info in lost_tracks:
                track = None
                for t in self.tracking.tracks:
                    if t.name == track_info['name']:
                        track = t
                        break
                
                if not track:
                    continue
                
                for t in self.tracking.tracks:
                    t.select = False
                track.select = True
                
                markers_before = len([m for m in track.markers if not m.mute])
                
                if track_info['extend_forward']:
                    for f in range(track_info['last_frame'], clip_duration):
                        next_m = track.markers.find_frame(f + 1)
                        if next_m and not next_m.mute:
                            break
                        scene_f = self.clip_to_scene_frame(f)
                        bpy.context.scene.frame_set(scene_f)
                        try:
                            self._run_ops(bpy.ops.clip.track_markers, backwards=False, sequence=False)
                        except Exception:
                            pass
                
                if track_info['extend_backward']:
                    for f in range(track_info['first_frame'], 1, -1):
                        prev_m = track.markers.find_frame(f - 1)
                        if prev_m and not prev_m.mute:
                            break
                        scene_f = self.clip_to_scene_frame(f)
                        bpy.context.scene.frame_set(scene_f)
                        try:
                            self._run_ops(bpy.ops.clip.track_markers, backwards=True, sequence=False)
                        except Exception:
                            pass
                
                markers_after = len([m for m in track.markers if not m.mute])
                if markers_after > markers_before:
                    extended += 1
        
        except Exception as e:
            print(f"AutoSolve: Track extension error: {e}")
        
        finally:
            try:
                if hasattr(self.settings, 'default_correlation_min'):
                    self.settings.default_correlation_min = orig_correlation
                if hasattr(self.settings, 'default_search_size'):
                    self.settings.default_search_size = orig_search
                bpy.context.scene.frame_set(current_frame)
            except (ReferenceError, AttributeError):
                pass
        
        if extended > 0:
            print(f"AutoSolve: Extended {extended}/{len(lost_tracks)} lost tracks")
        else:
            print("AutoSolve: Could not extend any tracks (features may have left frame)")
        
        return extended
    
    def heal_tracks(self) -> int:
        """
        Find and heal track gaps using anchor-based interpolation.
        """
        if not self.enable_healing:
            return 0
        
        if self.healer is None:
            from .track_healer import TrackHealer
            self.healer = TrackHealer()
        
        anchors = self.healer.find_anchor_tracks(self.tracking)
        
        if len(anchors) < self.healer.MIN_ANCHOR_TRACKS:
            print(f"AutoSolve: Only {len(anchors)} anchors found - need {self.healer.MIN_ANCHOR_TRACKS}+ for healing")
            return 0
        
        candidates = self.healer.find_healing_candidates(self.tracking)
        
        if not candidates:
            return 0
        
        healed = 0
        attempted = 0
        gap_frames_total = 0
        match_scores_total = 0.0
        below_threshold = 0
        
        for candidate in candidates:
            if candidate.match_score >= self.healer.MIN_MATCH_SCORE:
                attempted += 1
                
                positions = self.healer.interpolate_with_anchors(candidate, anchors, self.tracking)
                success = self.healer.heal_track(candidate, self.tracking, anchors, positions=positions)
                
                training_data = self.healer.collect_training_data(
                    candidate, anchors, positions, success
                )
                
                if success:
                    healed += 1
                    gap_frames_total += candidate.gap_frames
                    match_scores_total += candidate.match_score
            else:
                below_threshold += 1
        
        if healed > 0:
            print(f"AutoSolve: Healed {healed}/{attempted} track gaps "
                  f"({100*healed/attempted:.0f}% success rate)")
        elif attempted > 0:
            print(f"AutoSolve: Healing attempted {attempted} gaps but none succeeded")
        elif below_threshold > 0:
            print(f"AutoSolve: {below_threshold} candidates found but none met score threshold "
                  f"(need >= {self.healer.MIN_MATCH_SCORE}, best: {candidates[0].match_score:.2f})")
        
        merged = self.healer.merge_overlapping_segments(self.tracking)
        if merged > 0:
            print(f"AutoSolve: Merged {merged} overlapping track segments via averaging")
            healed += merged
        
        return healed

    def remove_worst_tracks(self, percentage: float = 0.15) -> int:
        """
        Remove the worst-performing tracks for iterative refinement.
        """
        tracks_with_error = []
        for track in self.tracking.tracks:
            if track.has_bundle:
                tracks_with_error.append((track.name, track.average_error))
        
        if len(tracks_with_error) < self.SAFE_MIN_TRACKS:
            print("AutoSolve: Not enough tracks for removal")
            return 0
        
        tracks_with_error.sort(key=lambda x: x[1], reverse=True)
        
        num_to_remove = max(1, int(len(tracks_with_error) * percentage))
        num_to_remove = min(num_to_remove, len(tracks_with_error) - self.SAFE_MIN_TRACKS)
        
        if num_to_remove <= 0:
            return 0
        
        to_remove = set(name for name, _ in tracks_with_error[:num_to_remove])
        
        for track in self.tracking.tracks:
            track.select = track.name in to_remove
        
        try:
            self._run_ops(bpy.ops.clip.delete_track)
            print(f"AutoSolve: Removed {num_to_remove} worst tracks (errors: "
                  f"{tracks_with_error[0][1]:.2f} - {tracks_with_error[num_to_remove-1][1]:.2f}px)")
        except Exception:
            return 0
        
        return num_to_remove
    
    def should_continue_refinement(self) -> bool:
        """
        Determine if another refinement iteration is needed.
        """
        MAX_REFINEMENT_ITERATIONS = 5
        TARGET_ERROR = 2.0
        
        if self.refinement_iteration >= MAX_REFINEMENT_ITERATIONS:
            print(f"AutoSolve: Max refinement iterations reached ({MAX_REFINEMENT_ITERATIONS})")
            return False
        
        current_error = self.get_solve_error()
        
        if current_error < TARGET_ERROR:
            print(f"AutoSolve: Target error achieved ({current_error:.2f}px < {TARGET_ERROR}px)")
            return False
        
        if current_error < self.best_solve_error:
            improvement = self.best_solve_error - current_error
            self.best_solve_error = current_error
            self.best_bundle_count = self.get_bundle_count()
            
            if improvement < 0.1 and self.refinement_iteration > 1:
                print(f"AutoSolve: Diminishing returns (improvement: {improvement:.2f}px)")
                return False
            
            return True
        else:
            print(f"AutoSolve: No improvement from last iteration")
            return False
    
    def refine_solve(self) -> bool:
        """
        Perform one iteration of solve refinement.
        """
        self.refinement_iteration += 1
        print(f"AutoSolve: Refinement iteration {self.refinement_iteration}")
        
        self.learn_from_failed_tracks()
        
        removed = self.remove_worst_tracks(percentage=0.15)
        if removed == 0:
            print("AutoSolve: Cannot remove more tracks")
            return False
        
        success = self.solve_camera(tripod_mode=False)
        
        if success:
            new_error = self.get_solve_error()
            new_bundles = self.get_bundle_count()
            print(f"AutoSolve: Refinement result - {new_bundles} bundles, {new_error:.2f}px error")
        
        return success

    def identify_user_tracks(self) -> List[str]:
        """
        Identify tracks that appear to be user-placed.
        """
        user_tracks = []
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            
            if 1 <= len(markers) <= 5:
                user_tracks.append(track.name)
            elif hasattr(track, 'lock') and track.lock:
                user_tracks.append(track.name)
        
        if user_tracks:
            print(f"AutoSolve: Identified {len(user_tracks)} user-placed tracks (will protect)")
        
        return user_tracks
    
    def refine_struggling_tracks(self, user_tracks: set = None) -> int:
        """
        Attempt to re-track struggling tracks with more tolerant settings.
        """
        if user_tracks is None:
            user_tracks = set(self.identify_user_tracks())
        
        if not self.tracking or len(self.tracking.tracks) == 0:
            return 0
        
        has_solve = self.tracking.reconstruction.is_valid
        min_lifespan = max(3, self.min_lifespan // 2)
        
        struggling = []
        for track in self.tracking.tracks:
            try:
                markers = [m for m in track.markers if not m.mute]
                if len(markers) < 2:
                    continue
                
                markers_sorted = sorted(markers, key=lambda m: m.frame)
                lifespan = markers_sorted[-1].frame - markers_sorted[0].frame
                
                is_struggling = False
                
                if lifespan < min_lifespan:
                    is_struggling = True
                
                if has_solve and track.has_bundle and track.average_error > 5.0:
                    is_struggling = True
                
                if is_struggling:
                    struggling.append({
                        'name': track.name,
                        'last_frame': markers_sorted[-1].frame,
                        'first_frame': markers_sorted[0].frame,
                        'lifespan': lifespan,
                        'is_user_track': track.name in user_tracks
                    })
            except (ReferenceError, AttributeError):
                continue
        
        if not struggling:
            return 0
        
        struggling.sort(key=lambda t: (not t['is_user_track'], -t['lifespan']))
        struggling = struggling[:15]
        
        user_count = sum(1 for t in struggling if t['is_user_track'])
        print(f"AutoSolve: Refining {len(struggling)} struggling tracks ({user_count} user-placed)...")
        
        orig_correlation = self.current_settings.get('correlation', 0.7)
        orig_search = self.current_settings.get('search_size', 71)
        
        tolerant_correlation = max(0.4, orig_correlation - 0.2)
        tolerant_search = int(orig_search * 1.3)
        
        try:
            if hasattr(self.settings, 'default_correlation_min'):
                self.settings.default_correlation_min = tolerant_correlation
            if hasattr(self.settings, 'default_search_size'):
                self.settings.default_search_size = tolerant_search
        except (ReferenceError, AttributeError):
            pass
        
        extended = 0
        current_frame = bpy.context.scene.frame_current
        
        try:
            for t in self.tracking.tracks:
                t.select = False
            
            struggling_names = {t['name'] for t in struggling}
            for t in self.tracking.tracks:
                if t.name in struggling_names:
                    t.select = True
            
            markers_before = {}
            for t in self.tracking.tracks:
                if t.name in struggling_names:
                    markers_before[t.name] = len([m for m in t.markers if not m.mute])
            
            min_frame = min(t['first_frame'] for t in struggling)
            max_frame = max(t['last_frame'] for t in struggling)
            
            mid_frame = (min_frame + max_frame) // 2
            bpy.context.scene.frame_set(mid_frame)
            
            self._run_ops(bpy.ops.clip.track_markers, backwards=False, sequence=True)
            
            bpy.context.scene.frame_set(mid_frame)
            self._run_ops(bpy.ops.clip.track_markers, backwards=True, sequence=True)
            
            user_extended = []
            for t in self.tracking.tracks:
                if t.name in struggling_names:
                    markers_after = len([m for m in t.markers if not m.mute])
                    if markers_after > markers_before.get(t.name, 0):
                        extended += 1
                        info = next((s for s in struggling if s['name'] == t.name), None)
                        if info and info['is_user_track']:
                            user_extended.append(f"'{t.name}' ({markers_before[t.name]}→{markers_after})")
            
            if user_extended:
                if len(user_extended) <= 3:
                    print(f"AutoSolve: Extended user tracks: {', '.join(user_extended)}")
                else:
                    print(f"AutoSolve: Extended {len(user_extended)} user tracks")
            
        except Exception as e:
            print(f"AutoSolve: Track refinement error (continuing): {e}")
        finally:
            try:
                if hasattr(self.settings, 'default_correlation_min'):
                    self.settings.default_correlation_min = orig_correlation
                if hasattr(self.settings, 'default_search_size'):
                    self.settings.default_search_size = orig_search
                bpy.context.scene.frame_set(current_frame)
            except (ReferenceError, AttributeError):
                pass
        
        if extended > 0:
            print(f"AutoSolve: Successfully refined {extended}/{len(struggling)} tracks")
        
        return extended

    def preserve_good_tracks(self, min_lifespan: int = None, max_error: float = 5.0, refine: bool = True) -> int:
        """
        Keep good existing tracks, only remove problematic ones.
        """
        if min_lifespan is None:
            min_lifespan = max(3, self.min_lifespan // 2)
        
        user_tracks = set(self.identify_user_tracks())
        
        if refine:
            self.refine_struggling_tracks(user_tracks=user_tracks)
        
        has_solve = self.tracking.reconstruction.is_valid
        
        good_tracks = []
        bad_tracks = []
        protected_tracks = []
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            lifespan = 0
            if len(markers) >= 2:
                markers_sorted = sorted(markers, key=lambda m: m.frame)
                lifespan = markers_sorted[-1].frame - markers_sorted[0].frame
            
            if track.name in user_tracks:
                protected_tracks.append(track.name)
                continue
            
            is_good = lifespan >= min_lifespan
            
            if has_solve and track.has_bundle:
                if track.average_error > max_error:
                    is_good = False
            
            if is_good and len(markers) >= 2:
                good_tracks.append(track.name)
            else:
                bad_tracks.append(track.name)
        
        if bad_tracks:
            for track in self.tracking.tracks:
                track.select = track.name in bad_tracks
            
            try:
                self._run_ops(bpy.ops.clip.delete_track)
                msg = f"AutoSolve: Removed {len(bad_tracks)} poor tracks, preserved {len(good_tracks)} good tracks"
                if protected_tracks:
                    msg += f", protected {len(protected_tracks)} user tracks"
                print(msg)
            except Exception:
                pass
        else:
            msg = f"AutoSolve: Preserved all {len(good_tracks)} existing tracks"
            if protected_tracks:
                msg += f" + {len(protected_tracks)} user tracks"
            print(msg)
        
        return len(good_tracks) + len(protected_tracks)

    def select_optimal_keyframes(self) -> bool:
        """
        Select optimal keyframes for camera solve based on parallax.
        """
        camera = self.clip.tracking.camera
        clip_start = self.clip.frame_start
        clip_end = self.clip.frame_start + self.clip.frame_duration - 1
        min_separation = max(10, int(self.clip.frame_duration * 0.2))
        
        frame_tracks = {}
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            for marker in markers:
                if marker.frame not in frame_tracks:
                    frame_tracks[marker.frame] = []
                frame_tracks[marker.frame].append((track.name, marker.co.x, marker.co.y))
        
        if len(frame_tracks) < 2:
            print("AutoSolve: Not enough frames with tracks for keyframe selection")
            return False
        
        valid_frames = [f for f, tracks in frame_tracks.items() if len(tracks) >= 8]
        if len(valid_frames) < 2:
            print("AutoSolve: Not enough frames with 8+ tracks")
            return False
        
        valid_frames.sort()
        
        best_parallax = 0
        best_pair = (valid_frames[0], valid_frames[-1])
        best_common_count = 0
        
        sample_step = max(1, len(valid_frames) // 10)
        sample_frames = valid_frames[::sample_step]
        if valid_frames[-1] not in sample_frames:
            sample_frames.append(valid_frames[-1])
        
        for i, frame_a in enumerate(sample_frames):
            for frame_b in sample_frames[i+1:]:
                if frame_b - frame_a < min_separation:
                    continue
                
                tracks_a = {t[0]: (t[1], t[2]) for t in frame_tracks[frame_a]}
                tracks_b = {t[0]: (t[1], t[2]) for t in frame_tracks[frame_b]}
                common_tracks = set(tracks_a.keys()) & set(tracks_b.keys())
                
                if len(common_tracks) < 8:
                    continue
                
                total_disp = 0
                for track_name in common_tracks:
                    xa, ya = tracks_a[track_name]
                    xb, yb = tracks_b[track_name]
                    disp = ((xb - xa)**2 + (yb - ya)**2)**0.5
                    total_disp += disp
                
                avg_parallax = total_disp / len(common_tracks)
                
                score = avg_parallax * (1 + len(common_tracks) / 50)
                
                if score > best_parallax:
                    best_parallax = score
                    best_pair = (frame_a, frame_b)
                    best_common_count = len(common_tracks)
        
        keyframe_a, keyframe_b = best_pair
        avg_parallax_percent = best_parallax * 100 if best_parallax < 1 else best_parallax
        
        scene_keyframe_a = self.clip_to_scene_frame(keyframe_a)
        scene_keyframe_b = self.clip_to_scene_frame(keyframe_b)
        
        if hasattr(camera, 'keyframe_a') and hasattr(camera, 'keyframe_b'):
            current_a = getattr(camera, 'keyframe_a', 1)
            current_b = getattr(camera, 'keyframe_b', clip_end)
            
            if scene_keyframe_a != current_a or scene_keyframe_b != current_b:
                camera.keyframe_a = scene_keyframe_a
                camera.keyframe_b = scene_keyframe_b
                print(f"AutoSolve: Selected keyframes {scene_keyframe_a} and {scene_keyframe_b} "
                      f"({best_common_count} common tracks, {avg_parallax_percent:.1f}% avg parallax)")
                return True
        else:
            print(f"AutoSolve: Optimal keyframes analysis: frames {scene_keyframe_a} and {scene_keyframe_b} "
                  f"({best_common_count} common tracks, {avg_parallax_percent:.1f}% avg parallax)")
            return True
        
        return False
