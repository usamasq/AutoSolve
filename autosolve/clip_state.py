# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Multi-clip state manager for AutoSolve.

Handles per-clip state isolation when users work with multiple clips
in the same Blender file. Ensures:
1. Learning data (sessions, behavior) is tied to specific clips
2. UI state (solve status, error, progress) is per-clip
3. Proper save/restore when switching between clips

Usage:
    manager = ClipStateManager()
    
    # Get state for current clip
    state = manager.get_state(clip)
    
    # On clip change, handles saving old clip's data
    manager.on_clip_changed(old_clip, new_clip)
"""

import hashlib
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import bpy


@dataclass
class ClipState:
    """Per-clip runtime state."""
    fingerprint: str = ""
    
    # Solve results
    has_solve: bool = False
    solve_error: float = 0.0
    point_count: int = 0
    solve_success: bool = False
    
    # Tracking progress
    is_solving: bool = False
    solve_progress: float = 0.0
    solve_status: str = ""
    
    # Last solve settings (for behavior learning)
    last_settings: Dict = field(default_factory=dict)
    last_footage_class: str = ""
    
    # Behavior tracking
    behavior_recorder: Any = None
    behavior_monitor: Any = None


class ClipStateManager:
    """
    Manages per-clip state for multi-clip workflows.
    
    Key design:
    - Each clip is identified by a privacy-safe fingerprint (resolution, fps, duration)
    - State is stored in a dictionary keyed by fingerprint
    - On clip switch, old clip's behavior is saved, new clip's state is loaded
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern - one manager per session."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._states: Dict[str, ClipState] = {}
        self._current_fingerprint: str = ""
        self._initialized = True
    
    def _generate_fingerprint(self, clip: bpy.types.MovieClip) -> str:
        """Generate privacy-safe fingerprint for clip identification."""
        if not clip:
            return ""
        try:
            # Use resolution, fps, frame count (no filename/path)
            data = f"{clip.size[0]}x{clip.size[1]}_{clip.fps}_{clip.frame_duration}"
            return hashlib.sha256(data.encode()).hexdigest()[:12]
        except (AttributeError, ReferenceError):
            return ""
    
    def get_state(self, clip: bpy.types.MovieClip) -> ClipState:
        """
        Get or create state for a clip.
        
        Returns ClipState for the given clip, creating if needed.
        """
        if not clip:
            return ClipState()
        
        fingerprint = self._generate_fingerprint(clip)
        if not fingerprint:
            return ClipState()
        
        if fingerprint not in self._states:
            self._states[fingerprint] = ClipState(fingerprint=fingerprint)
        
        return self._states[fingerprint]
    
    def get_current_clip_state(self) -> Optional[ClipState]:
        """Get state for the currently active clip in the editor."""
        clip = self._get_active_clip()
        if clip:
            return self.get_state(clip)
        return None
    
    def _get_active_clip(self) -> Optional[bpy.types.MovieClip]:
        """Get the currently active clip from context."""
        try:
            # Try various context paths
            if hasattr(bpy.context, 'edit_movieclip') and bpy.context.edit_movieclip:
                return bpy.context.edit_movieclip
            
            # Try space data
            for area in bpy.context.screen.areas:
                if area.type == 'CLIP_EDITOR':
                    space = area.spaces.active
                    if hasattr(space, 'clip') and space.clip:
                        return space.clip
            
            return None
        except (AttributeError, ReferenceError):
            return None
    
    def is_same_clip(self, clip: bpy.types.MovieClip) -> bool:
        """Check if the given clip is the same as the last tracked clip."""
        if not clip:
            return False
        fingerprint = self._generate_fingerprint(clip)
        return fingerprint == self._current_fingerprint
    
    def on_clip_changed(self, old_clip: bpy.types.MovieClip, new_clip: bpy.types.MovieClip):
        """
        Handle clip switch - save old clip's data, prepare new clip.
        
        Call this when detecting that user has switched to a different clip.
        """
        # Save behavior data for old clip
        if old_clip:
            old_state = self.get_state(old_clip)
            self._save_clip_behavior(old_state)
        
        # Update current fingerprint
        if new_clip:
            self._current_fingerprint = self._generate_fingerprint(new_clip)
        else:
            self._current_fingerprint = ""
    
    def _save_clip_behavior(self, state: ClipState):
        """Save pending behavior data for a clip."""
        try:
            if state.behavior_recorder and hasattr(state.behavior_recorder, 'is_monitoring'):
                if state.behavior_recorder.is_monitoring:
                    behavior = state.behavior_recorder.stop_monitoring(None, None)
                    if behavior:
                        state.behavior_recorder.save_behavior(behavior)
                        print(f"AutoSolve: Saved behavior for clip {state.fingerprint[:8]}")
        except Exception as e:
            print(f"AutoSolve: Error saving clip behavior: {e}")
    
    def set_current_clip(self, clip: bpy.types.MovieClip):
        """Set the current active clip (call after successful solve)."""
        if clip:
            self._current_fingerprint = self._generate_fingerprint(clip)
    
    def sync_to_blender_properties(self, clip: bpy.types.MovieClip, scene: bpy.types.Scene):
        """
        Sync ClipState to Blender's scene properties for UI access.
        
        Call this when clip changes to update the UI.
        """
        if not clip or not scene or not hasattr(scene, 'autosolve'):
            return
        
        state = self.get_state(clip)
        settings = scene.autosolve
        
        settings.has_solve = state.has_solve
        settings.solve_error = state.solve_error
        settings.point_count = state.point_count
        settings.is_solving = state.is_solving
        settings.solve_progress = state.solve_progress
        settings.solve_status = state.solve_status
    
    def update_from_blender(self, clip: bpy.types.MovieClip, scene: bpy.types.Scene):
        """
        Update ClipState from Blender's current tracking state.
        
        Call this after a solve completes to capture the actual results.
        """
        if not clip or not scene:
            return
        
        state = self.get_state(clip)
        
        try:
            tracking = clip.tracking
            reconstruction = tracking.reconstruction
            
            state.has_solve = reconstruction.is_valid
            state.solve_error = reconstruction.average_error if reconstruction.is_valid else 0.0
            
            # Count bundles
            bundle_count = sum(1 for t in tracking.tracks if t.has_bundle)
            state.point_count = bundle_count
            
        except (AttributeError, ReferenceError):
            pass
    
    def clear_all(self):
        """Clear all clip states (call on file close)."""
        # Save any pending behavior first
        for state in self._states.values():
            self._save_clip_behavior(state)
        
        self._states.clear()
        self._current_fingerprint = ""


# Global singleton instance
_clip_manager: Optional[ClipStateManager] = None


def get_clip_manager() -> ClipStateManager:
    """Get the global clip state manager."""
    global _clip_manager
    if _clip_manager is None:
        _clip_manager = ClipStateManager()
    return _clip_manager


def reset_clip_manager():
    """Reset the global clip state manager singleton.
    
    Should be called during addon unregister to prevent stale state
    when addon is disabled/enabled without restarting Blender.
    """
    global _clip_manager
    if _clip_manager is not None:
        _clip_manager.clear_all()
    _clip_manager = None
