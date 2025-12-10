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
from typing import Dict, List, Optional, Tuple
import bpy

from ..utils import get_region, calculate_jitter, get_sessions_dir


@dataclass
class TrackTelemetry:
    """Telemetry for a single track."""
    # Required fields (no defaults)
    name: str
    lifespan: int
    start_frame: int
    end_frame: int
    region: str
    avg_velocity: float
    jitter_score: float
    success: bool
    # Optional fields (have defaults)
    contributed_to_solve: bool = False
    reprojection_error: float = 0.0
    # ML Enhancement: Sampled trajectory for RNN training
    trajectory: List[List[float]] = field(default_factory=list)  # [[x,y], [x,y], ...]
    trajectory_sample_rate: int = 5  # Every Nth frame


@dataclass
class CameraIntrinsics:
    """Camera intrinsics and lens distortion data."""
    focal_length_mm: float = 0.0
    focal_length_px: float = 0.0
    sensor_width_mm: float = 36.0
    pixel_aspect: float = 1.0
    principal_point: List[float] = field(default_factory=lambda: [0.5, 0.5])
    # Distortion model and coefficients
    distortion_model: str = 'POLYNOMIAL'  # POLYNOMIAL, DIVISION, NUKE, BROWN
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    # Division model
    division_k1: float = 0.0
    division_k2: float = 0.0
    # Nuke model
    nuke_k1: float = 0.0
    nuke_k2: float = 0.0
    # Brown-Conrady model
    brown_k1: float = 0.0
    brown_k2: float = 0.0
    brown_k3: float = 0.0
    brown_k4: float = 0.0
    brown_p1: float = 0.0
    brown_p2: float = 0.0


@dataclass
class SessionData:
    """Complete data for a tracking session."""
    # Metadata (required fields - no defaults) - MUST COME FIRST
    timestamp: str
    clip_name: str
    iteration: int
    duration_seconds: float
    resolution: Tuple[int, int]
    fps: float
    frame_count: int
    settings: Dict
    success: bool
    solve_error: float
    total_tracks: int
    successful_tracks: int
    bundle_count: int
    
    # Optional fields (have defaults) - MUST COME AFTER REQUIRED FIELDS
    # Schema version
    schema_version: int = 1
    
    # Detailed track data
    tracks: List[Dict] = field(default_factory=list)
    
    # Region analysis
    region_stats: Dict = field(default_factory=dict)
    dead_zones: List[str] = field(default_factory=list)
    sweet_spots: List[str] = field(default_factory=list)
    
    # ML Enhancement: Camera intrinsics
    camera_intrinsics: Dict = field(default_factory=dict)
    
    # ML Enhancement: Global motion descriptors
    global_motion_vector: List[float] = field(default_factory=lambda: [0.0, 0.0])
    motion_consistency: float = 0.0  # Std dev of per-track velocities
    
    # ML Enhancement: Optical Flow Descriptors (raw metrics for ML)
    # All values are continuous and normalized for direct ML consumption
    optical_flow: Dict = field(default_factory=lambda: {
        # Velocity statistics (normalized: pixels/frame / image_diagonal)
        'velocity_mean': 0.0,       # Average track movement per frame
        'velocity_std': 0.0,        # Velocity standard deviation
        'velocity_max': 0.0,        # Maximum velocity observed
        
        # Parallax detection (0.0 = uniform motion, 1.0 = strong depth variance)
        'parallax_score': 0.0,      # Variance between track motion vectors
        
        # Motion direction (unit vector of dominant camera movement)
        'dominant_direction': [0.0, 0.0],  # [dx, dy] normalized
        'direction_entropy': 0.0,   # 0.0 = all same direction, 1.0 = random
        
        # Temporal stability
        'velocity_acceleration': 0.0,  # Change in velocity over clip
        'track_dropout_rate': 0.0,     # Fraction of tracks that fail early
    })
    
    # ML Enhancement: Failure classification
    failure_type: str = 'NONE'  # NONE, BLUR, CONTRAST, CUT, DRIFT, INSUFFICIENT
    frame_of_failure: Optional[int] = None
    
    # Motion probe results (persisted for learning)
    motion_probe_results: Dict = field(default_factory=dict)
    
    # Mid-session adaptation history
    adaptation_history: List[Dict] = field(default_factory=list)
    region_confidence: Dict = field(default_factory=dict)
    
    # ML Enhancement: Per-frame samples for temporal analysis
    # Format: [{"frame": int, "active_tracks": int, "tracks_lost": int, "avg_velocity": float}, ...]
    frame_samples: List[Dict] = field(default_factory=list)
    
    # ML Enhancement: Source video metadata (anonymized)
    source_metadata: Dict = field(default_factory=lambda: {
        'file_extension': '',
        'file_size_mb': 0.0,
        'codec_hint': '',  # Inferred from extension
    })
    
    # ML Enhancement: Pre-solve confidence estimate
    pre_solve_confidence: Dict = field(default_factory=lambda: {
        'confidence': 0.0,  # 0-1 estimate of solve success
        'parallax_score': 0.0,
        'track_distribution_score': 0.0,
        'warnings': [],
    })


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
            self.data_dir = Path(bpy.utils.user_resource('DATAFILES')) / 'autosolve' / 'sessions'
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[SessionData] = None
        self.start_time: Optional[datetime] = None
    
    def start_session(self, clip: bpy.types.MovieClip, settings: Dict):
        """Start recording a new session."""
        self.start_time = datetime.now()
        
        # Extract camera intrinsics
        camera_intrinsics = self._extract_camera_intrinsics(clip)
        
        # Extract source video metadata
        source_metadata = self._extract_source_metadata(clip)
        
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
            camera_intrinsics=camera_intrinsics,
            source_metadata=source_metadata,
        )
    
    def _extract_camera_intrinsics(self, clip: bpy.types.MovieClip) -> Dict:
        """Extract camera intrinsics and lens distortion from clip."""
        try:
            cam = clip.tracking.camera
            
            intrinsics = CameraIntrinsics(
                focal_length_mm=cam.focal_length,
                focal_length_px=cam.focal_length_pixels,
                sensor_width_mm=cam.sensor_width,
                pixel_aspect=cam.pixel_aspect,
                principal_point=[cam.principal_point[0], cam.principal_point[1]],
                distortion_model=cam.distortion_model,
                # Polynomial model
                k1=cam.k1,
                k2=cam.k2,
                k3=cam.k3,
                # Division model
                division_k1=cam.division_k1,
                division_k2=cam.division_k2,
                # Nuke model
                nuke_k1=cam.nuke_k1,
                nuke_k2=cam.nuke_k2,
                # Brown-Conrady model
                brown_k1=cam.brown_k1,
                brown_k2=cam.brown_k2,
                brown_k3=cam.brown_k3,
                brown_k4=cam.brown_k4,
                brown_p1=cam.brown_p1,
                brown_p2=cam.brown_p2,
            )
            
            return asdict(intrinsics)
            return asdict(intrinsics)
        except (AttributeError, TypeError, ReferenceError) as e:
            print(f"AutoSolve: Error extracting camera intrinsics: {e}")
            return asdict(CameraIntrinsics())  # Return defaults
    
    def _extract_source_metadata(self, clip: bpy.types.MovieClip) -> Dict:
        """
        Extract metadata from source video file.
        
        Collects anonymized info about the source file that may affect tracking:
        - File extension (codec hint)
        - File size (compression level indicator)
        """
        metadata = {
            'file_extension': '',
            'file_size_mb': 0.0,
            'codec_hint': '',
        }
        
        try:
            filepath = bpy.path.abspath(clip.filepath)
            if filepath:
                from pathlib import Path
                path = Path(filepath)
                
                metadata['file_extension'] = path.suffix.lower()
                
                # Infer codec from extension
                codec_hints = {
                    '.mp4': 'h264/h265',
                    '.mov': 'prores/h264',
                    '.mkv': 'various',
                    '.avi': 'legacy',
                    '.webm': 'vp9/av1',
                    '.mxf': 'professional',
                }
                metadata['codec_hint'] = codec_hints.get(metadata['file_extension'], 'unknown')
                
                # Get file size (anonymized to MB range)
                if path.exists():
                    size_bytes = path.stat().st_size
                    metadata['file_size_mb'] = round(size_bytes / (1024 * 1024), 1)
        except Exception as e:
            print(f"AutoSolve: Error extracting source metadata: {e}")
        
        return metadata
    
    def record_pre_solve_confidence(self, confidence_data: Dict):
        """
        Record pre-solve confidence estimate.
        
        Args:
            confidence_data: Dict with keys:
                - confidence: 0-1 overall estimate
                - parallax_score: 0-1 depth variation
                - track_distribution_score: 0-1 coverage quality
                - warnings: List of warning strings
        """
        if not self.current_session:
            return
        
        self.current_session.pre_solve_confidence = {
            'confidence': confidence_data.get('confidence', 0.0),
            'parallax_score': confidence_data.get('parallax_score', 0.0),
            'track_distribution_score': confidence_data.get('track_distribution_score', 0.0),
            'warnings': confidence_data.get('warnings', []),
        }
    
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
    
    def record_tracks(self, tracking, trajectory_sample_rate: int = 5):
        """Record detailed track telemetry with trajectory data for ML training."""
        if not self.current_session:
            return
        
        self.current_session.tracks.clear()
        
        # Collect velocities for global motion computation
        all_velocities = []
        all_motion_vectors = []
        
        for track in tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            markers.sort(key=lambda x: x.frame)
            
            try:
                import math
                # Validate marker coordinates - skip tracks with NaN/Inf
                has_invalid = False
                for m in markers:
                    if math.isnan(m.co.x) or math.isnan(m.co.y) or math.isinf(m.co.x) or math.isinf(m.co.y):
                        has_invalid = True
                        break
                if has_invalid:
                    continue
                
                # Calculate metrics
                lifespan = markers[-1].frame - markers[0].frame
                
                # Average position for region
                avg_x = sum(m.co.x for m in markers) / len(markers)
                avg_y = sum(m.co.y for m in markers) / len(markers)
                region = get_region(avg_x, avg_y)
                
                # Velocity
                from mathutils import Vector
                displacement = (Vector(markers[-1].co) - Vector(markers[0].co)).length
                avg_velocity = displacement / max(lifespan, 1)
                all_velocities.append(avg_velocity)
                
                # Motion vector (dx, dy) for global motion
                dx = markers[-1].co.x - markers[0].co.x
                dy = markers[-1].co.y - markers[0].co.y
                all_motion_vectors.append((dx, dy))
                
                # Jitter
                jitter = calculate_jitter(markers)
                
                # ML Enhancement: Sample trajectory every N frames
                trajectory = []
                for i, marker in enumerate(markers):
                    if i % trajectory_sample_rate == 0:
                        trajectory.append([round(marker.co.x, 4), round(marker.co.y, 4)])
                
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
                    trajectory=trajectory,
                    trajectory_sample_rate=trajectory_sample_rate,
                )
            
                self.current_session.tracks.append(asdict(telemetry))
                self.current_session.tracks.append(asdict(telemetry))
            except (AttributeError, TypeError, ValueError, ZeroDivisionError, ReferenceError) as e:
                # Skip tracks with invalid data
                continue
        
        # Compute global motion descriptors
        if all_motion_vectors:
            avg_dx = sum(v[0] for v in all_motion_vectors) / len(all_motion_vectors)
            avg_dy = sum(v[1] for v in all_motion_vectors) / len(all_motion_vectors)
            self.current_session.global_motion_vector = [round(avg_dx, 5), round(avg_dy, 5)]
        
        if all_velocities:
            mean_v = sum(all_velocities) / len(all_velocities)
            variance = sum((v - mean_v) ** 2 for v in all_velocities) / len(all_velocities)
            self.current_session.motion_consistency = round(1.0 - min(variance ** 0.5 / max(mean_v, 0.001), 1.0), 3)
        
        # Compute region_stats from recorded tracks (for ML training)
        region_stats = {}
        for track_data in self.current_session.tracks:
            region = track_data.get('region', 'center')
            if region not in region_stats:
                region_stats[region] = {'total_tracks': 0, 'successful_tracks': 0, 'avg_lifespan': 0.0}
            region_stats[region]['total_tracks'] += 1
            if track_data.get('contributed_to_solve', False):
                region_stats[region]['successful_tracks'] += 1
            # Update running average lifespan
            count = region_stats[region]['total_tracks']
            old_avg = region_stats[region]['avg_lifespan']
            new_lifespan = track_data.get('lifespan', 0)
            region_stats[region]['avg_lifespan'] = old_avg + (new_lifespan - old_avg) / count
        
        self.current_session.region_stats = region_stats
        
        # Compute optical flow descriptors for ML
        self._compute_optical_flow(all_velocities, all_motion_vectors, tracking)
    
    def _compute_optical_flow(self, velocities: List[float], motion_vectors: List, tracking):
        """
        Compute comprehensive optical flow metrics for ML training.
        
        All values are continuous and normalized for direct use in neural networks.
        """
        if not self.current_session:
            return
        
        of = self.current_session.optical_flow
        
        # Velocity statistics
        if velocities:
            of['velocity_mean'] = round(sum(velocities) / len(velocities), 6)
            of['velocity_std'] = round((sum((v - of['velocity_mean'])**2 for v in velocities) / len(velocities))**0.5, 6)
            of['velocity_max'] = round(max(velocities), 6)
        
        # Parallax detection: variance in motion direction between tracks
        # Low parallax = tripod/uniform motion, High = drone/depth variation
        if len(motion_vectors) >= 3:
            avg_dx = sum(v[0] for v in motion_vectors) / len(motion_vectors)
            avg_dy = sum(v[1] for v in motion_vectors) / len(motion_vectors)
            
            # Compute variance from mean direction
            dir_variance = sum(
                ((v[0] - avg_dx)**2 + (v[1] - avg_dy)**2) 
                for v in motion_vectors
            ) / len(motion_vectors)
            
            # Normalize: sqrt(variance) / magnitude of average motion
            avg_magnitude = (avg_dx**2 + avg_dy**2)**0.5
            if avg_magnitude > 0.0001:
                of['parallax_score'] = round(min(1.0, (dir_variance**0.5) / avg_magnitude), 4)
            else:
                of['parallax_score'] = 0.0
            
            # Dominant direction (unit vector)
            if avg_magnitude > 0.0001:
                of['dominant_direction'] = [
                    round(avg_dx / avg_magnitude, 4),
                    round(avg_dy / avg_magnitude, 4)
                ]
            
            # Direction entropy: how varied are the motion directions?
            # Use angle variance as proxy for entropy
            import math
            angles = [math.atan2(v[1], v[0]) for v in motion_vectors if (v[0]**2 + v[1]**2) > 1e-10]
            if angles:
                mean_angle = sum(angles) / len(angles)
                angle_variance = sum((a - mean_angle)**2 for a in angles) / len(angles)
                # Normalize to 0-1 (pi/2 radians variance = 1.0)
                of['direction_entropy'] = round(min(1.0, angle_variance / (math.pi/2)**2), 4)
        
        # Track dropout rate
        tracks = self.current_session.tracks
        if tracks:
            total_possible_lifespan = self.current_session.frame_count
            early_failures = sum(1 for t in tracks if t.get('lifespan', 0) < total_possible_lifespan * 0.3)
            of['track_dropout_rate'] = round(early_failures / len(tracks), 4)
    
    def record_motion_probe(self, probe_results: Dict):
        """Record motion probe results for ML training."""
        if not self.current_session:
            return
        
        # Store sanitized probe data (remove internal state)
        self.current_session.motion_probe_results = {
            'motion_class': probe_results.get('motion_class', 'MEDIUM'),
            'texture_class': probe_results.get('texture_class', 'MEDIUM'),
            'best_regions': probe_results.get('best_regions', []),
            'velocities': probe_results.get('velocities', {}),
            'region_success': probe_results.get('region_success', {}),
        }
        print(f"AutoSolve: Recorded motion probe - {probe_results.get('motion_class', '?')} motion")
    
    def record_failure_diagnostics(self, failure_type: str, frame_of_failure: Optional[int] = None):
        """
        Record failure diagnostics from FailureDiagnostics analysis.
        
        Args:
            failure_type: One of MOTION_BLUR, LOW_CONTRAST, RAPID_MOTION, etc.
            frame_of_failure: Frame where failure was detected
        """
        if not self.current_session:
            return
        
        self.current_session.failure_type = failure_type
        self.current_session.frame_of_failure = frame_of_failure
        print(f"AutoSolve: Recorded failure - {failure_type} at frame {frame_of_failure}")
    
    def record_adaptation_history(self, adaptation_summary: Dict):
        """
        Record mid-session adaptation history.
        
        Args:
            adaptation_summary: From SmartTracker.get_adaptation_summary()
        """
        if not self.current_session:
            return
        
        self.current_session.adaptation_history = adaptation_summary.get('adaptation_history', [])
        self.current_session.region_confidence = adaptation_summary.get('region_confidence', {})
    
    def record_frame_sample(self, frame: int, tracking, prev_active_count: int = 0):
        """
        Record per-frame statistics for ML temporal analysis.
        
        Call this periodically during tracking (e.g., every 10 frames) to capture
        temporal dynamics for RNN/LSTM training.
        
        Args:
            frame: Current frame number
            tracking: Blender tracking object
            prev_active_count: Active track count from previous sample (to compute tracks_lost)
        """
        if not self.current_session:
            return 0
        
        # Count active tracks at this frame
        active_count = 0
        velocities = []
        
        for track in tracking.tracks:
            marker = track.markers.find_frame(frame)
            if marker and not marker.mute:
                active_count += 1
                
                # Compute velocity from previous marker
                prev_marker = track.markers.find_frame(frame - 1)
                if prev_marker and not prev_marker.mute:
                    dx = marker.co.x - prev_marker.co.x
                    dy = marker.co.y - prev_marker.co.y
                    v = (dx**2 + dy**2) ** 0.5
                    velocities.append(v)
        
        # Compute average velocity
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0.0
        
        # Compute tracks lost since last sample
        tracks_lost = max(0, prev_active_count - active_count) if prev_active_count > 0 else 0
        
        sample = {
            'frame': frame,
            'active_tracks': active_count,
            'tracks_lost': tracks_lost,
            'avg_velocity': round(avg_velocity, 6),
        }
        
        self.current_session.frame_samples.append(sample)
        return active_count  # Return for next call's prev_active_count
    
    # Alias for backward compatibility
    def finalize_session(self, success: bool, solve_error: float, bundle_count: int):
        """Alias for end_session for backward compatibility."""
        self.end_session(success, solve_error, bundle_count)
    
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
    
    def _save_edit_session(self, edit_session):
        """
        Save user edit session data to JSON file.
        
        Args:
            edit_session: EditSession dataclass from UserEditRecorder
        """
        from dataclasses import asdict
        
        try:
            # Create edits subdirectory
            edits_dir = self.data_dir.parent / 'edits'
            edits_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = edit_session.timestamp.replace(':', '-').replace('.', '-')
            filename = f"edits_{timestamp[:19]}.json"
            filepath = edits_dir / filename
            
            # Convert to dict and sanitize
            edit_dict = asdict(edit_session)
            edit_dict = self._sanitize_for_json(edit_dict)
            
            with open(filepath, 'w') as f:
                json.dump(edit_dict, f, indent=2)
            
            print(f"AutoSolve: Saved edit session to {filepath}")
        except (OSError, IOError, TypeError) as e:
            print(f"AutoSolve: Error saving edit session: {e}")
    
    def _save_session(self):
        """Save session data to JSON file."""
        if not self.current_session:
            return
        
        try:
            # Generate filename with sanitized characters
            timestamp = self.current_session.timestamp.replace(':', '-').replace('.', '-')
            # Sanitize clip name - remove/replace problematic characters
            clip_name = self.current_session.clip_name
            for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
                clip_name = clip_name.replace(char, '_')
            clip_name = clip_name.replace('.', '_').replace(' ', '_')
            
            filename = f"{timestamp[:19]}_{clip_name}.json"
            filepath = self.data_dir / filename
            
            # Convert session to dict and sanitize for JSON serialization
            session_dict = asdict(self.current_session)
            session_dict = self._sanitize_for_json(session_dict)
            
            with open(filepath, 'w') as f:
                json.dump(session_dict, f, indent=2)
            
            print(f"AutoSolve: Saved session data to {filepath}")
        except (OSError, IOError, TypeError) as e:
            print(f"AutoSolve: Error saving session: {e}")
    
    def _sanitize_for_json(self, data):
        """Recursively sanitize data for JSON serialization."""
        import math
        
        if isinstance(data, dict):
            return {k: self._sanitize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_for_json(v) for v in data]
        elif isinstance(data, tuple):
            return list(data)  # Convert tuples to lists
        elif isinstance(data, float):
            if math.isnan(data) or math.isinf(data):
                return 0.0  # Replace NaN/Inf with 0
            return round(data, 6)  # Limit precision
        elif isinstance(data, (int, str, bool, type(None))):
            return data
        else:
            return str(data)  # Convert unknown types to string
    
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
    
    def get_statistics(self, clip_name: Optional[str] = None) -> Dict:
        """Get aggregate statistics from all sessions."""
        sessions = self.load_sessions(clip_name)
        
        if not sessions:
            return {
                'total_sessions': 0,
                'successful_sessions': 0,
                'success_rate': 0.0,
                'avg_solve_error': 0.0,
                'avg_iterations': 0.0,
                'region_success_rates': {},
            }
        
        total = len(sessions)
        successful = sum(1 for s in sessions if s.get('success'))
        
        # Calculate average error from successful sessions only
        successful_sessions = [s for s in sessions if s.get('success')]
        avg_error = (
            sum(s.get('solve_error', 0) for s in successful_sessions) / 
            max(len(successful_sessions), 1)
        )
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
            'success_rate': successful / total if total > 0 else 0.0,
            'avg_solve_error': avg_error,
            'avg_iterations': avg_iterations,
            'region_success_rates': region_success_rates,
        }

    
    # _get_region and _calculate_jitter removed - use from ..utils import instead
