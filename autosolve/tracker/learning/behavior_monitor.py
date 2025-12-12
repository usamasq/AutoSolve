# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
BehaviorMonitor - Timer-based monitor for manual tracking/solve operations.

Monitors clip state changes to detect when user performs manual tracking
or solves using Blender's built-in tools (not AutoSolve).

Usage:
    monitor = BehaviorMonitor(clip)
    
    # In a timer callback (every 2 seconds):
    changes = monitor.check_changes()
    if changes:
        behavior_recorder.record_manual_action(changes)
"""

from typing import Dict, Optional
import bpy


class BehaviorMonitor:
    """
    Timer-based monitor for detecting manual tracking/solve operations.
    
    Since Blender has no callback for built-in operators, this class
    polls clip state periodically to detect changes.
    
    IMPORTANT: Users can work on multiple clips in one session.
    This monitor tracks a specific clip and should be reset when
    the user switches to a different clip.
    """
    
    def __init__(self, clip: bpy.types.MovieClip):
        """
        Initialize monitor with current clip state.
        
        Args:
            clip: The MovieClip to monitor
        """
        self.clip = clip
        self.is_valid = True
        
        # Store clip fingerprint for multi-clip detection
        self.clip_fingerprint = self._generate_fingerprint(clip)
        
        # Snapshot initial state
        try:
            self.last_track_count = len(clip.tracking.tracks)
            self.last_solve_valid = clip.tracking.reconstruction.is_valid
            self.last_solve_error = self._get_solve_error()
            
            # Track individual track names to detect additions/deletions
            self.last_track_names = set(t.name for t in clip.tracking.tracks)
        except (AttributeError, ReferenceError):
            self.is_valid = False
            self.last_track_count = 0
            self.last_solve_valid = False
            self.last_solve_error = 0.0
            self.last_track_names = set()
    
    def _generate_fingerprint(self, clip: bpy.types.MovieClip) -> str:
        """Generate a fingerprint to identify this specific clip."""
        try:
            import hashlib
            # Use resolution, fps, duration as fingerprint (privacy-safe)
            data = f"{clip.size[0]}x{clip.size[1]}_{clip.fps}_{clip.frame_duration}"
            return hashlib.sha256(data.encode()).hexdigest()[:12]
        except (AttributeError, ReferenceError):
            return ""
    
    def is_same_clip(self, clip: bpy.types.MovieClip) -> bool:
        """
        Check if the given clip is the same as the one being monitored.
        
        Used to detect when user switches clips and need to save/reset behavior.
        """
        if not self.is_valid or not clip:
            return False
        try:
            other_fingerprint = self._generate_fingerprint(clip)
            return self.clip_fingerprint == other_fingerprint
        except (AttributeError, ReferenceError):
            return False
    
    def _get_solve_error(self) -> float:
        """Get current solve error, or 0.0 if invalid."""
        try:
            if self.clip.tracking.reconstruction.is_valid:
                return self.clip.tracking.reconstruction.average_error
        except (AttributeError, ReferenceError):
            pass
        return 0.0
    
    def check_changes(self) -> Dict:
        """
        Poll for changes since last check.
        
        Call this from a timer every 2 seconds to detect:
        - Track additions (manual tracking)
        - Track deletions (manual cleanup)
        - Solve changes (manual solve from Blender's Solve panel)
        
        Returns:
            Dict with detected changes, empty if no changes.
            Keys:
                - 'action': 'manual_track', 'manual_delete', 'manual_solve'
                - 'track_delta': number of tracks added/removed
                - 'tracks_added': list of new track names
                - 'tracks_removed': list of deleted track names
                - 'solve_changed': bool
                - 'was_valid': bool
                - 'is_valid': bool
                - 'old_error': float
                - 'new_error': float
        """
        if not self.is_valid:
            return {}
        
        changes = {}
        
        try:
            # Check if clip is still valid
            _ = self.clip.name
            
            # Check track count changes
            current_track_count = len(self.clip.tracking.tracks)
            current_track_names = set(t.name for t in self.clip.tracking.tracks)
            
            if current_track_count != self.last_track_count:
                delta = current_track_count - self.last_track_count
                changes['track_delta'] = delta
                
                if delta > 0:
                    changes['action'] = 'manual_track'
                    changes['tracks_added'] = list(current_track_names - self.last_track_names)
                else:
                    changes['action'] = 'manual_delete'
                    changes['tracks_removed'] = list(self.last_track_names - current_track_names)
                
                self.last_track_count = current_track_count
                self.last_track_names = current_track_names
            
            # Check for solve changes
            current_valid = self.clip.tracking.reconstruction.is_valid
            current_error = self._get_solve_error()
            
            # Detect solve change: validity toggled OR error changed significantly
            error_changed = abs(current_error - self.last_solve_error) > 0.01
            validity_changed = current_valid != self.last_solve_valid
            
            if validity_changed or (current_valid and error_changed):
                changes['solve_changed'] = True
                changes['was_valid'] = self.last_solve_valid
                changes['is_valid'] = current_valid
                changes['old_error'] = self.last_solve_error
                changes['new_error'] = current_error
                
                # If solve just became valid or error improved, this is a manual solve
                if current_valid and (not self.last_solve_valid or current_error < self.last_solve_error):
                    changes['action'] = 'manual_solve'
                
                self.last_solve_valid = current_valid
                self.last_solve_error = current_error
            
            return changes
            
        except (ReferenceError, AttributeError):
            # Clip was deleted or invalidated
            self.is_valid = False
            return {}
    
    def reset_snapshot(self):
        """Reset state snapshot after AutoSolve completes."""
        try:
            self.last_track_count = len(self.clip.tracking.tracks)
            self.last_solve_valid = self.clip.tracking.reconstruction.is_valid
            self.last_solve_error = self._get_solve_error()
            self.last_track_names = set(t.name for t in self.clip.tracking.tracks)
        except (AttributeError, ReferenceError):
            self.is_valid = False
