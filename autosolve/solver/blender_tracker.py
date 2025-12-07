# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Blender Tracker Automation.

Wraps Blender's built-in motion tracking into a one-call interface.
Compatible with Blender 4.2 - 5.0+
"""

import bpy
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class TrackResult:
    """Result from AutoTracker.run()"""
    success: bool
    num_tracks: int = 0
    average_error: float = 0.0
    error_message: str = ""


class AutoTracker:
    """Automates Blender's motion tracking pipeline."""
    
    # Quality presets
    QUALITY_PRESETS = {
        'FAST': {
            'detect_threshold': 0.1,      # Lower = more features
            'min_distance': 100,
            'margin': 16,
            'min_track_frames': 8,
            'max_error': 2.0,
        },
        'BALANCED': {
            'detect_threshold': 0.3,
            'min_distance': 70,
            'margin': 16,
            'min_track_frames': 12,
            'max_error': 1.0,
        },
        'QUALITY': {
            'detect_threshold': 0.5,
            'min_distance': 50,
            'margin': 16,
            'min_track_frames': 18,
            'max_error': 0.7,
        },
    }
    
    MIN_TRACKS_REQUIRED = 8
    
    def __init__(self, clip: bpy.types.MovieClip):
        self.clip = clip
        self.tracking = clip.tracking
        self.settings = clip.tracking.settings
    
    def run(
        self,
        quality: str = 'BALANCED',
        tripod_mode: bool = False,
        iterative_refine: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> TrackResult:
        """Run the complete tracking pipeline."""
        preset = self.QUALITY_PRESETS.get(quality, self.QUALITY_PRESETS['BALANCED'])
        
        try:
            # Step 0: Configure settings
            self._progress(progress_callback, "Configuring...", 0.05)
            self._configure_settings(tripod_mode)
            
            # Step 1: Clear existing tracks
            self._progress(progress_callback, "Clearing old tracks...", 0.08)
            self._clear_all_tracks()
            
            # Step 2: Detect features
            self._progress(progress_callback, "Detecting features...", 0.10)
            num_detected = self._detect_features(preset)
            
            if num_detected < self.MIN_TRACKS_REQUIRED:
                return TrackResult(
                    success=False,
                    num_tracks=num_detected,
                    error_message=f"Only {num_detected} features detected. Need at least {self.MIN_TRACKS_REQUIRED}. Try lowering Quality.",
                )
            
            # Step 3: Track forward
            self._progress(progress_callback, "Tracking forward...", 0.25)
            self._select_all_tracks()
            result = bpy.ops.clip.track_markers(backwards=False, sequence=True)
            if result != {'FINISHED'}:
                return TrackResult(
                    success=False,
                    error_message="Forward tracking failed.",
                )
            
            # Step 4: Track backward
            self._progress(progress_callback, "Tracking backward...", 0.45)
            self._select_all_tracks()
            bpy.ops.clip.track_markers(backwards=True, sequence=True)
            
            # Step 5: Clean tracks
            self._progress(progress_callback, "Cleaning tracks...", 0.60)
            self._clean_tracks(preset)
            
            # Count remaining tracks
            num_tracks = len(self.tracking.tracks)
            if num_tracks < self.MIN_TRACKS_REQUIRED:
                return TrackResult(
                    success=False,
                    num_tracks=num_tracks,
                    error_message=f"Only {num_tracks} tracks after cleaning. Try 'Fast' quality.",
                )
            
            # Step 6: Solve camera
            self._progress(progress_callback, "Solving camera...", 0.75)
            solve_success = self._solve_camera()
            
            if not solve_success:
                return TrackResult(
                    success=False,
                    num_tracks=num_tracks,
                    error_message="Camera solve failed. Try Tripod Mode.",
                )
            
            # Step 7: Iterative refinement
            if iterative_refine and num_tracks >= 10:
                self._progress(progress_callback, "Refining solve...", 0.85)
                self._iterative_refine(preset)
            
            # Get final stats
            final_tracks = len([t for t in self.tracking.tracks if t.has_bundle])
            avg_error = self.tracking.reconstruction.average_error
            
            self._progress(progress_callback, "Complete!", 1.0)
            
            return TrackResult(
                success=True,
                num_tracks=final_tracks,
                average_error=avg_error,
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return TrackResult(
                success=False,
                error_message=f"Tracking failed: {str(e)}",
            )
    
    def _configure_settings(self, tripod: bool):
        """Configure Blender's tracking settings."""
        s = self.settings
        
        # Tripod mode
        if hasattr(s, 'use_tripod_solver'):
            s.use_tripod_solver = tripod
        
        # Motion model - use Loc for simplicity
        if hasattr(s, 'default_motion_model'):
            s.default_motion_model = 'Loc'
        
        # Pattern size and search size
        if hasattr(s, 'default_pattern_size'):
            s.default_pattern_size = 21
        if hasattr(s, 'default_search_size'):
            s.default_search_size = 71
    
    def _clear_all_tracks(self):
        """Remove all existing tracking markers."""
        if len(self.tracking.tracks) == 0:
            return
            
        self._select_all_tracks()
        try:
            bpy.ops.clip.delete_track()
        except RuntimeError:
            pass
    
    def _detect_features(self, preset: dict) -> int:
        """Detect features on the current frame."""
        # Go to first frame
        bpy.context.scene.frame_set(self.clip.frame_start)
        
        # Detect features with lower threshold for more features
        bpy.ops.clip.detect_features(
            placement='FRAME',
            margin=preset['margin'],
            threshold=preset['detect_threshold'],
            min_distance=preset['min_distance'],
        )
        
        num_detected = len(self.tracking.tracks)
        print(f"AutoSolve: Detected {num_detected} features")
        return num_detected
    
    def _select_all_tracks(self):
        """Select all tracks."""
        for track in self.tracking.tracks:
            track.select = True
    
    def _clean_tracks(self, preset: dict):
        """Clean bad tracks."""
        self._select_all_tracks()
        
        # Try Blender 5.0 first
        try:
            bpy.ops.clip.clean_tracks(
                frames=preset['min_track_frames'],
                error=preset['max_error'],
                action='DELETE_TRACK',
            )
            print(f"AutoSolve: After cleaning, {len(self.tracking.tracks)} tracks remain")
        except TypeError:
            try:
                bpy.ops.clip.clean_tracks(
                    frames=preset['min_track_frames'],
                    error=preset['max_error'],
                    action='DELETE',
                )
            except:
                pass
    
    def _solve_camera(self) -> bool:
        """Run the camera solver."""
        try:
            result = bpy.ops.clip.solve_camera()
            print(f"AutoSolve: solve_camera returned {result}")
        except RuntimeError as e:
            print(f"AutoSolve: solve_camera error: {e}")
            return False
        
        is_valid = self.tracking.reconstruction.is_valid
        if is_valid:
            print(f"AutoSolve: Solve succeeded, error = {self.tracking.reconstruction.average_error:.2f}px")
        else:
            print("AutoSolve: Solve failed - reconstruction not valid")
        return is_valid
    
    def _iterative_refine(self, preset: dict):
        """Remove worst tracks and re-solve."""
        tracks_with_error = []
        for track in self.tracking.tracks:
            if track.has_bundle:
                tracks_with_error.append((track.name, track.average_error))
        
        if len(tracks_with_error) < 10:
            return
        
        tracks_with_error.sort(key=lambda x: x[1], reverse=True)
        
        num_to_remove = max(1, len(tracks_with_error) // 10)
        tracks_to_remove = [
            name for name, error in tracks_with_error[:num_to_remove]
            if error > preset['max_error']
        ]
        
        if not tracks_to_remove:
            return
        
        # Deselect all, select only bad tracks
        for track in self.tracking.tracks:
            track.select = track.name in tracks_to_remove
        
        try:
            bpy.ops.clip.delete_track()
            bpy.ops.clip.solve_camera()
        except RuntimeError:
            pass
    
    @staticmethod
    def _progress(callback, message: str, progress: float):
        if callback:
            callback(message, progress)


def sync_scene_to_clip(clip: bpy.types.MovieClip):
    """Sync scene frame range and FPS to match the clip."""
    scene = bpy.context.scene
    
    # Frame range
    scene.frame_start = clip.frame_start
    scene.frame_end = clip.frame_start + clip.frame_duration - 1
    
    # FPS
    fps = clip.fps
    if fps > 0:
        if fps == int(fps):
            scene.render.fps = int(fps)
            scene.render.fps_base = 1.0
        else:
            # Handle common fractional FPS
            if abs(fps - 23.976) < 0.01:
                scene.render.fps = 24000
                scene.render.fps_base = 1001
            elif abs(fps - 29.97) < 0.01:
                scene.render.fps = 30000
                scene.render.fps_base = 1001
            elif abs(fps - 59.94) < 0.01:
                scene.render.fps = 60000
                scene.render.fps_base = 1001
            else:
                scene.render.fps = round(fps)
                scene.render.fps_base = 1.0
    
    # Resolution
    if clip.size[0] > 0 and clip.size[1] > 0:
        scene.render.resolution_x = clip.size[0]
        scene.render.resolution_y = clip.size[1]
        scene.render.resolution_percentage = 100
    
    print(f"AutoSolve: Scene synced - frames {scene.frame_start}-{scene.frame_end}, {fps:.2f} fps")
