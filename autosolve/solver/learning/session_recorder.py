# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
SessionRecorder - Records tracking sessions for learning.

Stores detailed telemetry about what worked and what didn't.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
import bpy


@dataclass
class TrackTelemetry:
    """Telemetry for a single track."""
    name: str
    lifespan: int
    start_frame: int
    end_frame: int
    region: str
    avg_velocity: float
    jitter_score: float
    success: bool
    contributed_to_solve: bool = False
    reprojection_error: float = 0.0


@dataclass
class SessionData:
    """Complete data for a tracking session."""
    # Metadata
    timestamp: str
    clip_name: str
    iteration: int
    duration_seconds: float
    
    # Footage characteristics
    resolution: tuple
    fps: float
    frame_count: int
    
    # Settings used
    settings: Dict
    
    # Results
    success: bool
    solve_error: float
    total_tracks: int
    successful_tracks: int
    bundle_count: int
    
    # Detailed track data
    tracks: List[Dict] = field(default_factory=list)
    
    # Region analysis
    region_stats: Dict = field(default_factory=dict)
    dead_zones: List[str] = field(default_factory=list)
    sweet_spots: List[str] = field(default_factory=list)


class SessionRecorder:
    """
    Records and stores tracking session data for ML training.
    
    Data is stored locally in JSON format for:
    - Offline analysis and model training
    - Future prediction improvements
    - Debug and optimization insights
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Use Blender's user data directory
            self.data_dir = Path(bpy.utils.user_resource('DATAFILES')) / 'eztrack' / 'sessions'
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[SessionData] = None
        self.start_time: Optional[datetime] = None
    
    def start_session(self, clip: bpy.types.MovieClip, settings: Dict):
        """Start recording a new session."""
        self.start_time = datetime.now()
        
        self.current_session = SessionData(
            timestamp=self.start_time.isoformat(),
            clip_name=clip.name,
            iteration=0,
            duration_seconds=0.0,
            resolution=(clip.size[0], clip.size[1]),
            fps=clip.fps if clip.fps > 0 else 24,
            frame_count=clip.frame_duration,
            settings=settings.copy(),
            success=False,
            solve_error=999.0,
            total_tracks=0,
            successful_tracks=0,
            bundle_count=0,
        )
    
    def record_iteration(self, iteration: int, settings: Dict, analysis: Dict):
        """Record data for a single iteration."""
        if not self.current_session:
            return
        
        self.current_session.iteration = iteration
        self.current_session.settings = settings.copy()
        
        self.current_session.total_tracks = analysis.get('total_tracks', 0)
        self.current_session.successful_tracks = analysis.get('successful_tracks', 0)
        self.current_session.region_stats = analysis.get('region_stats', {})
        self.current_session.dead_zones = analysis.get('dead_zones', [])
        self.current_session.sweet_spots = analysis.get('sweet_spots', [])
    
    def record_tracks(self, tracking):
        """Record detailed track telemetry."""
        if not self.current_session:
            return
        
        self.current_session.tracks.clear()
        
        for track in tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            markers.sort(key=lambda x: x.frame)
            
            # Calculate metrics
            lifespan = markers[-1].frame - markers[0].frame
            
            # Average position for region
            avg_x = sum(m.co.x for m in markers) / len(markers)
            avg_y = sum(m.co.y for m in markers) / len(markers)
            region = self._get_region(avg_x, avg_y)
            
            # Velocity
            from mathutils import Vector
            displacement = (Vector(markers[-1].co) - Vector(markers[0].co)).length
            avg_velocity = displacement / max(lifespan, 1)
            
            # Jitter
            jitter = self._calculate_jitter(markers)
            
            telemetry = TrackTelemetry(
                name=track.name,
                lifespan=lifespan,
                start_frame=markers[0].frame,
                end_frame=markers[-1].frame,
                region=region,
                avg_velocity=avg_velocity,
                jitter_score=jitter,
                success=lifespan >= 5,
                contributed_to_solve=track.has_bundle,
                reprojection_error=track.average_error if track.has_bundle else 0.0,
            )
            
            self.current_session.tracks.append(asdict(telemetry))
    
    def end_session(self, success: bool, solve_error: float, bundle_count: int):
        """Finalize and save the session."""
        if not self.current_session or not self.start_time:
            return
        
        end_time = datetime.now()
        self.current_session.duration_seconds = (end_time - self.start_time).total_seconds()
        self.current_session.success = success
        self.current_session.solve_error = solve_error
        self.current_session.bundle_count = bundle_count
        
        # Save to disk
        self._save_session()
    
    def _save_session(self):
        """Save session data to JSON file."""
        if not self.current_session:
            return
        
        # Generate filename
        timestamp = self.current_session.timestamp.replace(':', '-').replace('.', '-')
        clip_name = self.current_session.clip_name.replace('.', '_').replace(' ', '_')
        filename = f"{timestamp[:19]}_{clip_name}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self.current_session), f, indent=2)
        
        print(f"AutoSolve: Saved session data to {filepath}")
    
    def load_sessions(self, clip_name: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Load previous sessions, optionally filtered by clip name."""
        sessions = []
        
        for filepath in sorted(self.data_dir.glob('*.json'), reverse=True):
            if len(sessions) >= limit:
                break
            
            try:
                with open(filepath) as f:
                    data = json.load(f)
                
                if clip_name and data.get('clip_name') != clip_name:
                    continue
                
                sessions.append(data)
            except Exception as e:
                print(f"AutoSolve: Error loading {filepath}: {e}")
        
        return sessions
    
    def get_aggregate_stats(self) -> Dict:
        """Calculate aggregate statistics across all sessions."""
        sessions = self.load_sessions()
        
        if not sessions:
            return {}
        
        total = len(sessions)
        successful = sum(1 for s in sessions if s.get('success'))
        
        avg_error = sum(s.get('solve_error', 0) for s in sessions if s.get('success')) / max(successful, 1)
        avg_iterations = sum(s.get('iteration', 0) + 1 for s in sessions) / total
        
        # Aggregate region stats
        region_totals = {}
        for session in sessions:
            for region, stats in session.get('region_stats', {}).items():
                if region not in region_totals:
                    region_totals[region] = {'total': 0, 'successful': 0}
                region_totals[region]['total'] += stats.get('total_tracks', 0)
                region_totals[region]['successful'] += stats.get('successful_tracks', 0)
        
        region_success_rates = {
            r: s['successful'] / max(s['total'], 1)
            for r, s in region_totals.items()
        }
        
        return {
            'total_sessions': total,
            'successful_sessions': successful,
            'success_rate': successful / total,
            'avg_solve_error': avg_error,
            'avg_iterations': avg_iterations,
            'region_success_rates': region_success_rates,
        }
    
    def _get_region(self, x: float, y: float) -> str:
        """Get region name from normalized coordinates."""
        col = 0 if x < 0.33 else (1 if x < 0.66 else 2)
        row = 2 if y < 0.33 else (1 if y < 0.66 else 0)
        
        regions = [
            ['top-left', 'top-center', 'top-right'],
            ['mid-left', 'center', 'mid-right'],
            ['bottom-left', 'bottom-center', 'bottom-right']
        ]
        return regions[row][col]
    
    def _calculate_jitter(self, markers) -> float:
        """Calculate jitter score."""
        if len(markers) < 3:
            return 0.0
        
        from mathutils import Vector
        velocities = []
        for i in range(1, len(markers)):
            v = (Vector(markers[i].co) - Vector(markers[i-1].co)).length
            velocities.append(v)
        
        if not velocities:
            return 0.0
        
        avg = sum(velocities) / len(velocities)
        if avg == 0:
            return 0.0
        
        variance = sum((v - avg) ** 2 for v in velocities) / len(velocities)
        return (variance ** 0.5) / avg
