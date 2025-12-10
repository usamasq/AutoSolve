# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
UserEditRecorder - Records pro user editing patterns.

Captures what tracks users delete, refine, or exclude from solving
to learn from expert behavior.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from datetime import datetime

from ..constants import REGIONS, EDGE_REGIONS
from ..utils import get_region


@dataclass
class TrackSnapshot:
    """Snapshot of a single track's state."""
    name: str
    lifespan: int
    start_frame: int
    end_frame: int
    region: str
    has_bundle: bool
    error: float
    is_enabled: bool  # Not hidden/muted
    jitter_score: float = 0.0
    avg_velocity: float = 0.0


@dataclass
class DeletedTrack:
    """Record of a deleted track with inferred reason."""
    name: str
    lifespan: int
    region: str
    had_bundle: bool
    error: float
    reason: str  # short_lifespan, edge_region, high_error, jittery, user_manual


@dataclass  
class EditSession:
    """Complete record of user edits between solves."""
    session_id: str
    timestamp: str
    
    # Before/after states
    tracks_before: int
    tracks_after: int
    
    # What changed
    deleted_tracks: List[Dict] = field(default_factory=list)
    disabled_tracks: List[str] = field(default_factory=list)
    enabled_tracks: List[str] = field(default_factory=list)
    
    # Solve participation
    tracks_used_for_solve: List[str] = field(default_factory=list)
    tracks_excluded_from_solve: List[str] = field(default_factory=list)
    
    # Timing
    time_spent_seconds: float = 0.0
    
    # Outcome (if user ran another solve)
    follow_up_solve_error: Optional[float] = None


class UserEditRecorder:
    """
    Records user editing patterns between solves.
    
    Usage:
        recorder = UserEditRecorder(clip)
        recorder.start_monitoring()  # After AutoSolve completes
        
        # ... user makes edits ...
        
        recorder.stop_monitoring()   # Before next solve
        edit_session = recorder.get_edit_session()
    """
    
    # Using REGIONS from constants.py
    
    def __init__(self, clip):
        """
        Initialize recorder for a clip.
        
        Args:
            clip: bpy.types.MovieClip to monitor
        """
        self.clip = clip
        self.initial_snapshot: List[TrackSnapshot] = []
        self.start_time: Optional[datetime] = None
        self.is_monitoring = False
        self.session_id = ""
    
    def start_monitoring(self):
        """Take initial snapshot after AutoSolve completes."""
        try:
            if not self.clip or not self.clip.tracking:
                return
            
            self.initial_snapshot = self._take_snapshot()
            self.start_time = datetime.now()
            self.is_monitoring = True
            self.session_id = self.start_time.strftime("%Y%m%d_%H%M%S")
            
            print(f"AutoSolve: Started monitoring edits ({len(self.initial_snapshot)} tracks)")
        except (ReferenceError, AttributeError):
            print("AutoSolve: Failed to start edit monitoring - MovieClip invalid")
            self.is_monitoring = False
    
    def stop_monitoring(self) -> Optional[EditSession]:
        """
        Stop monitoring and return the edit session.
        
        Returns:
            EditSession with all detected changes, or None if not monitoring
        """
        if not self.is_monitoring:
            return None
        
        self.is_monitoring = False
        
        # Take final snapshot
        final_snapshot = self._take_snapshot()
        if final_snapshot is None:
            print("AutoSolve: Edit recording aborted - MovieClip invalidated")
            return None
        
        # Calculate time spent
        time_spent = 0.0
        if self.start_time:
            time_spent = (datetime.now() - self.start_time).total_seconds()
        
        # Detect changes
        deleted = self._find_deleted_tracks(self.initial_snapshot, final_snapshot)
        disabled = self._find_disabled_tracks(self.initial_snapshot, final_snapshot)
        enabled = self._find_enabled_tracks(self.initial_snapshot, final_snapshot)
        
        # Solve participation (enabled tracks with bundles)
        used_for_solve = [t.name for t in final_snapshot if t.has_bundle and t.is_enabled]
        excluded = [t.name for t in final_snapshot if t.has_bundle and not t.is_enabled]
        
        session = EditSession(
            session_id=self.session_id,
            timestamp=self.start_time.isoformat() if self.start_time else "",
            tracks_before=len(self.initial_snapshot),
            tracks_after=len(final_snapshot),
            deleted_tracks=[asdict(d) for d in deleted],
            disabled_tracks=disabled,
            enabled_tracks=enabled,
            tracks_used_for_solve=used_for_solve,
            tracks_excluded_from_solve=excluded,
            time_spent_seconds=time_spent,
        )
        
        print(f"AutoSolve: Edit session recorded - {len(deleted)} deleted, "
              f"{len(disabled)} disabled, {time_spent:.1f}s editing")
        
        return session
    
    def _take_snapshot(self) -> Optional[List[TrackSnapshot]]:
        """Capture current state of all tracks."""
        snapshots = []
        
        try:
            # Check for ReferenceError explicitly on access
            if not self.clip:
                return None
            
            # This access can raise ReferenceError if clip is freed
            tracks = self.clip.tracking.tracks
            
            for track in tracks:
                markers = [m for m in track.markers if not m.mute]
                if len(markers) < 2:
                    continue
                
                markers.sort(key=lambda x: x.frame)
                lifespan = markers[-1].frame - markers[0].frame
                
                # Calculate average position for region
                avg_x = sum(m.co.x for m in markers) / len(markers)
                avg_y = sum(m.co.y for m in markers) / len(markers)
                
                snapshots.append(TrackSnapshot(
                    name=track.name,
                    lifespan=lifespan,
                    start_frame=markers[0].frame,
                    end_frame=markers[-1].frame,
                    region=get_region(avg_x, avg_y),
                    has_bundle=track.has_bundle,
                    error=track.average_error if track.has_bundle else 0.0,
                    is_enabled=not track.hide,
                ))
                
        except ReferenceError:
            print("AutoSolve: MovieClip removed while monitoring")
            return None
        except Exception as e:
            print(f"AutoSolve: Error taking snapshot: {e}")
            return None
        
        return snapshots
    
    def _find_deleted_tracks(self, before: List[TrackSnapshot], 
                             after: List[TrackSnapshot]) -> List[DeletedTrack]:
        """Find tracks that were deleted between snapshots."""
        after_names = {t.name for t in after}
        deleted = []
        
        for track in before:
            if track.name not in after_names:
                reason = self._infer_deletion_reason(track)
                deleted.append(DeletedTrack(
                    name=track.name,
                    lifespan=track.lifespan,
                    region=track.region,
                    had_bundle=track.has_bundle,
                    error=track.error,
                    reason=reason,
                ))
        
        return deleted
    
    def _find_disabled_tracks(self, before: List[TrackSnapshot],
                               after: List[TrackSnapshot]) -> List[str]:
        """Find tracks that were disabled (hidden) by user."""
        after_map = {t.name: t for t in after}
        disabled = []
        
        for track in before:
            if track.name in after_map:
                after_track = after_map[track.name]
                if track.is_enabled and not after_track.is_enabled:
                    disabled.append(track.name)
        
        return disabled
    
    def _find_enabled_tracks(self, before: List[TrackSnapshot],
                              after: List[TrackSnapshot]) -> List[str]:
        """Find tracks that were re-enabled by user."""
        after_map = {t.name: t for t in after}
        enabled = []
        
        for track in before:
            if track.name in after_map:
                after_track = after_map[track.name]
                if not track.is_enabled and after_track.is_enabled:
                    enabled.append(track.name)
        
        return enabled
    
    def _infer_deletion_reason(self, track: TrackSnapshot) -> str:
        """Infer why user deleted this track."""
        # Short lifespan
        if track.lifespan < 10:
            return "short_lifespan"
        
        # Edge region (often problematic due to distortion)
        if track.region in EDGE_REGIONS:
            return "edge_region"
        
        # High reprojection error
        if track.error > 2.0:
            return "high_error"
        
        # Jittery track
        if track.jitter_score > 0.5:
            return "jittery"
        
        # No obvious reason - expert judgment
        return "user_manual"
    
    # _get_region removed - use from ..utils import instead
