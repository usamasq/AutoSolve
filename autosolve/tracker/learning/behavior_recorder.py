# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
BehaviorRecorder - Records user behavior patterns for ML training.

Captures what settings users adjust, which tracks they delete/refine,
and whether their changes improved the solve.
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import bpy

from ..constants import REGIONS, EDGE_REGIONS
from ..utils import get_region, infer_deletion_reason, get_behavior_dir


@dataclass
class SettingsAdjustment:
    """Record of a settings change made by user."""
    setting_name: str
    before: float
    after: float
    delta: float


@dataclass
class TrackDeletion:
    """Record of a track deleted by user."""
    track_name: str
    region: str
    lifespan: int
    had_bundle: bool
    reprojection_error: float
    inferred_reason: str  # high_error, short_lifespan, edge_region, jittery, user_manual


@dataclass
class MarkerRefinement:
    """Record of a marker position refined by user."""
    track_name: str
    frame: int
    old_position: Tuple[float, float]
    new_position: Tuple[float, float]
    displacement_px: float


@dataclass
class ReSolveResult:
    """Record of user re-solving after edits."""
    attempted: bool = False
    error_before: float = 0.0
    error_after: float = 0.0
    improvement: float = 0.0
    improved: bool = False


@dataclass
class BehaviorData:
    """Complete behavior data for a session."""
    schema_version: int = 1
    session_id: str = ""
    timestamp: str = ""
    
    editing_duration_seconds: float = 0.0
    
    settings_adjustments: Dict[str, Dict] = field(default_factory=dict)
    re_solve: Dict = field(default_factory=dict)
    track_deletions: List[Dict] = field(default_factory=list)
    track_disables: List[Dict] = field(default_factory=list)
    marker_refinements: List[Dict] = field(default_factory=list)


class BehaviorRecorder:
    """
    Records user behavior between AutoSolve runs.
    
    Captures:
    - Settings adjustments (what did user change before re-running?)
    - Track deletions (which auto-placed tracks did user remove?)
    - Marker refinements (where did user adjust marker positions?)
    - Re-solve success (did user's changes improve the result?)
    
    Usage:
        recorder = BehaviorRecorder(clip)
        recorder.start_monitoring(initial_settings, initial_error)
        
        # ... user makes edits ...
        
        behavior = recorder.stop_monitoring(final_settings, final_error)
        recorder.save_behavior(behavior)
    """
    
    # Using REGIONS from constants.py
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = get_behavior_dir()
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.clip = None
        self.is_monitoring = False
        self.start_time: Optional[datetime] = None
        self.session_id = ""
        
        # Initial state snapshots
        self.initial_settings: Dict = {}
        self.initial_error: float = 0.0
        self.initial_tracks: Dict[str, Dict] = {}  # name -> {region, markers: {frame: (x,y)}}
    
    def start_monitoring(self, clip: bpy.types.MovieClip, 
                         settings: Dict, solve_error: float,
                         session_id: str = ""):
        """
        Start monitoring user behavior after AutoSolve completes.
        
        Args:
            clip: The MovieClip being tracked
            settings: Current tracker settings
            solve_error: Current solve error
            session_id: ID to link with session file
        """
        self.clip = clip
        self.start_time = datetime.now()
        self.is_monitoring = True
        self.session_id = session_id or self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Snapshot initial state
        self.initial_settings = settings.copy()
        self.initial_error = solve_error
        self.initial_tracks = self._snapshot_tracks()
        
        print(f"AutoSolve: Started behavior monitoring ({len(self.initial_tracks)} tracks)")
    
    def stop_monitoring(self, final_settings: Dict = None, 
                        final_error: float = None) -> Optional[BehaviorData]:
        """
        Stop monitoring and return behavior data.
        
        Args:
            final_settings: Settings after user edits (if re-solved)
            final_error: Solve error after user edits (if re-solved)
            
        Returns:
            BehaviorData with all detected changes
        """
        if not self.is_monitoring:
            return None
        
        self.is_monitoring = False
        
        # Calculate duration
        duration = 0.0
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
        
        # Snapshot final state
        final_tracks = self._snapshot_tracks()
        
        # Detect changes
        deletions = self._find_deletions(self.initial_tracks, final_tracks)
        disables = self._find_disables(self.initial_tracks, final_tracks)
        refinements = self._find_refinements(self.initial_tracks, final_tracks)
        
        # Settings adjustments
        settings_adj = {}
        if final_settings:
            for key in ['pattern_size', 'search_size', 'correlation', 'threshold']:
                if key in self.initial_settings and key in final_settings:
                    before = self.initial_settings[key]
                    after = final_settings[key]
                    if before != after:
                        settings_adj[key] = {
                            'before': before,
                            'after': after,
                            'delta': after - before
                        }
        
        # Re-solve result
        re_solve = {'attempted': False}
        if final_error is not None and final_error != self.initial_error:
            improvement = self.initial_error - final_error
            re_solve = {
                'attempted': True,
                'error_before': self.initial_error,
                'error_after': final_error,
                'improvement': improvement,
                'improved': improvement > 0
            }
        
        behavior = BehaviorData(
            schema_version=1,
            session_id=self.session_id,
            timestamp=self.start_time.isoformat() if self.start_time else "",
            editing_duration_seconds=duration,
            settings_adjustments=settings_adj,
            re_solve=re_solve,
            track_deletions=[asdict(d) for d in deletions],
            track_disables=[{'track_name': name, 'region': region} 
                           for name, region in disables],
            marker_refinements=[asdict(r) for r in refinements],
        )
        
        print(f"AutoSolve: Behavior recorded - {len(deletions)} deletions, "
              f"{len(refinements)} refinements, {len(settings_adj)} settings changed")
        
        return behavior
    
    def save_behavior(self, behavior: BehaviorData):
        """Save behavior data to disk."""
        if not behavior:
            return
        
        filename = f"{behavior.session_id}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(behavior), f, indent=2)
        
        print(f"AutoSolve: Saved behavior to {filename}")
    
    def _snapshot_tracks(self) -> Dict[str, Dict]:
        """Capture current state of all tracks with marker positions."""
        try:
            # Check if clip is valid (handle deleted/invalidated pointer)
            if not self.clip:
                return {}
            
            # Accessing properties will raise ReferenceError if StructRNA is gone
            if not self.clip.tracking:
                return {}
                
            tracks = {}
            for track in self.clip.tracking.tracks:
                markers = {}
                avg_x, avg_y = 0.0, 0.0
                count = 0
                
                for marker in track.markers:
                    if not marker.mute:
                        markers[marker.frame] = (marker.co.x, marker.co.y)
                        avg_x += marker.co.x
                        avg_y += marker.co.y
                        count += 1
                
                if count > 0:
                    avg_x /= count
                    avg_y /= count
                
                tracks[track.name] = {
                    'region': get_region(avg_x, avg_y),
                    'lifespan': len(markers),
                    'has_bundle': track.has_bundle,
                    'error': track.average_error if track.has_bundle else 0.0,
                    'hidden': track.hide,
                    'markers': markers
                }
            
            return tracks
            
        except (ReferenceError, AttributeError):
            # Clip was removed or invalidated
            return {}
    
    def _find_deletions(self, before: Dict, after: Dict) -> List[TrackDeletion]:
        """Find tracks that were deleted by user."""
        deletions = []
        
        for name, data in before.items():
            if name not in after:
                reason = self._infer_deletion_reason(data)
                deletions.append(TrackDeletion(
                    track_name=name,
                    region=data['region'],
                    lifespan=data['lifespan'],
                    had_bundle=data['has_bundle'],
                    reprojection_error=data['error'],
                    inferred_reason=reason
                ))
        
        return deletions
    
    def _find_disables(self, before: Dict, after: Dict) -> List[Tuple[str, str]]:
        """Find tracks that were disabled (hidden) by user."""
        disables = []
        
        for name, data in before.items():
            if name in after:
                if not data['hidden'] and after[name]['hidden']:
                    disables.append((name, data['region']))
        
        return disables
    
    def _find_refinements(self, before: Dict, after: Dict) -> List[MarkerRefinement]:
        """Find markers that were repositioned by user."""
        try:
            if not self.clip:
                return []
                
            refinements = []
            
            # Cache clip size to avoid repeated potential access issues
            width = self.clip.size[0]
            height = self.clip.size[1]
            
            for name, before_data in before.items():
                if name not in after:
                    continue
                
                after_data = after[name]
                before_markers = before_data['markers']
                after_markers = after_data['markers']
                
                # Check each marker that exists in both
                for frame, (bx, by) in before_markers.items():
                    if frame in after_markers:
                        ax, ay = after_markers[frame]
                        
                        # Calculate displacement in pixels
                        dx = (ax - bx) * width
                        dy = (ay - by) * height
                        displacement = (dx**2 + dy**2) ** 0.5
                        
                        # Only record significant refinements (> 0.5 pixels)
                        if displacement > 0.5:
                            refinements.append(MarkerRefinement(
                                track_name=name,
                                frame=frame,
                                old_position=(bx, by),
                                new_position=(ax, ay),
                                displacement_px=round(displacement, 2)
                            ))
            
            return refinements
            
        except (ReferenceError, AttributeError):
            return []
    
    def _infer_deletion_reason(self, track_data: Dict) -> str:
        """Infer why user deleted this track."""
        return infer_deletion_reason(track_data)
    
    # _get_region removed - use from ..utils import instead
    
    def get_behavior_count(self) -> int:
        """Get count of saved behavior files."""
        if not self.data_dir.exists():
            return 0
        return len(list(self.data_dir.glob('*.json')))
