# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
SessionRecorder - Records tracking sessions for learning.

Stores detailed telemetry about what worked and what didn't.
"""

import hashlib
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
    """Telemetry for a single track - designed for ML training."""
    # Required fields (no defaults)
    name: str
    lifespan: int
    start_frame: int
    end_frame: int
    region: str
    avg_velocity: float
    jitter_score: float
    success: bool  # Survived long enough (lifespan >= threshold)
    
    # Optional fields (have defaults)
    contributed_to_solve: bool = False
    reprojection_error: float = 0.0
    
    # ML Enhancement: Per-marker data for survival prediction
    initial_position: List[float] = field(default_factory=lambda: [0.0, 0.0])  # [x, y] at start
    feature_quality_score: float = 0.0  # Quality score from detection (0-1)
    
    # ML Enhancement: Sampled trajectory for RNN training
    trajectory: List[List[float]] = field(default_factory=list)  # [[x,y], [x,y], ...]
    trajectory_sample_rate: int = 5  # Every Nth frame
    
    # ML Enhancement: Per-segment velocities for acceleration detection
    trajectory_velocities: List[float] = field(default_factory=list)  # [v1, v2, ...] between samples
    
    # ML Enhancement: Per-sample confidence (pseudo-correlation from velocity smoothness)
    # High confidence = smooth motion, Low confidence = jerky/hunting motion
    trajectory_confidence: List[float] = field(default_factory=list)  # [0.98, 0.95, 0.72, ...]
    avg_confidence: float = 1.0  # Average confidence across all samples
    
    # ML Enhancement v5: Endpoint signatures for track healing/re-identification
    start_signature: Dict = field(default_factory=lambda: {
        'position': [0.0, 0.0],
        'velocity': [0.0, 0.0],
        'acceleration': [0.0, 0.0],
        'confidence': 1.0,
    })
    end_signature: Dict = field(default_factory=lambda: {
        'position': [0.0, 0.0],
        'velocity': [0.0, 0.0],
        'acceleration': [0.0, 0.0],
        'confidence': 1.0,
    })
    
    # Is this track a good anchor for healing other tracks?
    is_anchor_candidate: bool = False
    anchor_quality: float = 0.0  # 0-1 quality as reference track


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
    
    # ML Enhancement v2: Clip fingerprint for per-clip learning
    clip_fingerprint: str = ""
    
    # Session linkage: link to previous session for multi-attempt analysis
    previous_session_id: str = ""
    contributor_id: str = ""  # Anonymous ID per Blender install (distinguishes users)
    
    # ML Enhancement v2: Visual features from feature_extractor
    # NOTE: motion_class is now stored ONLY in visual_features (removed redundancy)
    visual_features: Dict = field(default_factory=dict)
    
    # ML Enhancement v2: Track failure logging
    # Format: [{"track_name": str, "frame": int, "position": [x, y], "reason": str}, ...]
    track_failures: List[Dict] = field(default_factory=list)
    
    # NOTE: flow_direction_histogram and flow_magnitude_histogram are now stored
    # ONLY in visual_features (removed redundancy)
    
    # ML Enhancement v3: Marker survival summary for quick analysis
    marker_survival_summary: Dict = field(default_factory=lambda: {
        'total_markers': 0,
        'survived_markers': 0,
        'survival_rate': 0.0,
        'avg_lifespan': 0.0,
        'avg_quality_score': 0.0,
        'quality_vs_survival_correlation': 0.0,  # For validating quality score
        'per_region': {},  # {"region": {"total": N, "survived": M, "rate": 0.X}, ...}
    })
    
    # ML Enhancement v4: Zoom/dolly detection from track trajectories
    # Uses existing trajectory data to detect and characterize zoom movements
    zoom_analysis: Dict = field(default_factory=lambda: {
        'is_zoom_detected': False,       # True if significant radial motion detected
        'zoom_direction': 'NONE',        # ZOOM_IN, ZOOM_OUT, NONE
        'scale_timeline': [],            # [1.0, 1.02, 1.05, ...] per trajectory sample
        'estimated_fl_ratio': 1.0,       # Final/initial scale ratio (>1=zoom out, <1=zoom in)
        'scale_variance': 0.0,           # Low=zoom (uniform), high=dolly (parallax)
        'is_uniform_scale': False,       # True if zoom-like, False if dolly-like
        'radial_convergence': 0.0,       # -1=converging (zoom in), +1=diverging (zoom out)
        'confidence': 0.0,               # Detection confidence (0-1)
    })
    
    # ML Enhancement v5: Track healing telemetry
    anchor_tracks: List[Dict] = field(default_factory=list)
    # Format: [{"name": str, "start_frame": int, "end_frame": int, "quality": float}]
    
    healing_attempts: List[Dict] = field(default_factory=list)
    # Full InterpolationTrainingData for each healing attempt
    
    healing_stats: Dict = field(default_factory=lambda: {
        'candidates_found': 0,
        'heals_attempted': 0,
        'heals_successful': 0,
        'avg_gap_frames': 0.0,
        'avg_match_score': 0.0,
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
        
        try:
            # Extract camera intrinsics
            camera_intrinsics = self._extract_camera_intrinsics(clip)
            
            # Extract source video metadata
            source_metadata = self._extract_source_metadata(clip)
            
            # Generate anonymous session ID (privacy: no clip names)
            session_id = self._generate_anonymous_id(clip)
            
            # Get clip size safely
            resolution = (
                clip.size[0] if clip.size[0] > 0 else 0,
                clip.size[1] if clip.size[1] > 0 else 0
            )
            fps = clip.fps if clip.fps > 0 else 24
            frame_count = clip.frame_duration if clip.frame_duration > 0 else 1
            
            self.current_session = SessionData(
                timestamp=self.start_time.isoformat(),
                clip_name=session_id,  # Anonymous hash, not actual filename
                iteration=0,
                duration_seconds=0.0,
                resolution=resolution,
                fps=fps,
                frame_count=frame_count,
                settings=settings.copy() if settings else {},
                success=False,
                solve_error=999.0,
                total_tracks=0,
                successful_tracks=0,
                bundle_count=0,
                camera_intrinsics=camera_intrinsics,
                source_metadata=source_metadata,
            )
        except (AttributeError, ReferenceError, TypeError) as e:
            print(f"AutoSolve: Error starting session: {e}")
            self.current_session = None
    
    def _generate_anonymous_id(self, clip: bpy.types.MovieClip) -> str:
        """
        Generate anonymous session ID for privacy.
        
        Does NOT include clip filename or path - only uses:
        - Resolution
        - FPS
        - Frame count
        - Current timestamp
        """
        try:
            # Edge case: clip may be invalid or deleted
            size_x = clip.size[0] if clip.size[0] > 0 else 0
            size_y = clip.size[1] if clip.size[1] > 0 else 0
            fps = clip.fps if clip.fps > 0 else 24  # Fallback to 24fps
            duration = clip.frame_duration if clip.frame_duration > 0 else 1
            
            data = f"{size_x}x{size_y}_{fps}_{duration}_{datetime.now().timestamp()}"
            return hashlib.sha256(data.encode()).hexdigest()[:8]
        except (AttributeError, ReferenceError, TypeError):
            # Clip is invalid - generate random ID as fallback
            import random
            return hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]
    
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
                trajectory_velocities = []
                prev_pos = None
                for i, marker in enumerate(markers):
                    if i % trajectory_sample_rate == 0:
                        pos = [round(marker.co.x, 4), round(marker.co.y, 4)]
                        trajectory.append(pos)
                        
                        # Compute velocity between this sample and previous
                        if prev_pos is not None:
                            dx = pos[0] - prev_pos[0]
                            dy = pos[1] - prev_pos[1]
                            velocity = (dx**2 + dy**2)**0.5 / trajectory_sample_rate
                            trajectory_velocities.append(round(velocity, 6))
                        prev_pos = pos
                
                # ML Enhancement: Compute trajectory confidence (pseudo-correlation)
                # Based on velocity smoothness - sudden changes indicate low confidence
                trajectory_confidence = []
                if len(trajectory_velocities) >= 2:
                    trajectory_confidence.append(1.0)  # First point has no history
                    for i in range(1, len(trajectory_velocities)):
                        # Acceleration = change in velocity
                        accel = abs(trajectory_velocities[i] - trajectory_velocities[i-1])
                        # Convert to confidence: high acceleration = low confidence
                        # Threshold: 0.005 normalized velocity change = near-zero confidence
                        confidence = max(0.0, 1.0 - accel / 0.005)
                        trajectory_confidence.append(round(confidence, 3))
                elif len(trajectory_velocities) == 1:
                    trajectory_confidence = [1.0]
                
                avg_confidence = sum(trajectory_confidence) / len(trajectory_confidence) if trajectory_confidence else 1.0
                
                # ML Enhancement: Initial position for survival prediction
                initial_pos = [round(markers[0].co.x, 4), round(markers[0].co.y, 4)]
                
                # ML Enhancement: Feature quality score
                # Based on distance from edges and center preference
                x, y = markers[0].co.x, markers[0].co.y
                edge_margin = 0.08
                quality = 1.0
                # Penalty for edges
                if x < edge_margin or x > (1.0 - edge_margin):
                    quality *= 0.7
                if y < edge_margin or y > (1.0 - edge_margin):
                    quality *= 0.7
                # Bonus for center
                center_dist = ((x - 0.5) ** 2 + (y - 0.5) ** 2) ** 0.5
                if center_dist < 0.25:
                    quality *= 1.1
                # Cap at 1.0
                quality = min(1.0, quality)
                
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
                    initial_position=initial_pos,
                    feature_quality_score=round(quality, 3),
                    trajectory=trajectory,
                    trajectory_sample_rate=trajectory_sample_rate,
                    trajectory_velocities=trajectory_velocities,
                    trajectory_confidence=trajectory_confidence,
                    avg_confidence=round(avg_confidence, 3),
                )
            
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
        
        # Update summary fields from recorded tracks
        self.current_session.total_tracks = len(self.current_session.tracks)
        self.current_session.successful_tracks = sum(
            1 for t in self.current_session.tracks if t.get('contributed_to_solve', False)
        )
        
        # ML Enhancement v3: Compute marker survival summary
        self._compute_marker_survival_summary()
        
        # Compute optical flow descriptors for ML
        self._compute_optical_flow(all_velocities, all_motion_vectors, tracking)
    
    def _compute_marker_survival_summary(self):
        """
        Compute marker survival summary for ML analysis.
        
        Includes per-region breakdown and quality-vs-survival correlation
        to validate the feature_quality_score metric.
        """
        if not self.current_session or not self.current_session.tracks:
            return
        
        tracks = self.current_session.tracks
        
        # Basic counts
        total = len(tracks)
        survived = sum(1 for t in tracks if t.get('success', False))
        
        # Lifespans and quality scores
        lifespans = [t.get('lifespan', 0) for t in tracks]
        quality_scores = [t.get('feature_quality_score', 0.0) for t in tracks]
        
        # Per-region breakdown
        per_region = {}
        for t in tracks:
            region = t.get('region', 'center')
            if region not in per_region:
                per_region[region] = {'total': 0, 'survived': 0, 'rate': 0.0}
            per_region[region]['total'] += 1
            if t.get('success', False):
                per_region[region]['survived'] += 1
        
        # Calculate rates
        for region, stats in per_region.items():
            if stats['total'] > 0:
                stats['rate'] = round(stats['survived'] / stats['total'], 3)
        
        # Simple correlation between quality and survival
        # (Pearson correlation approximation)
        correlation = 0.0
        if total >= 3 and any(q > 0 for q in quality_scores):
            survival_binary = [1.0 if t.get('success', False) else 0.0 for t in tracks]
            
            mean_q = sum(quality_scores) / total
            mean_s = sum(survival_binary) / total
            
            numerator = sum((q - mean_q) * (s - mean_s) for q, s in zip(quality_scores, survival_binary))
            denom_q = sum((q - mean_q) ** 2 for q in quality_scores) ** 0.5
            denom_s = sum((s - mean_s) ** 2 for s in survival_binary) ** 0.5
            
            if denom_q > 0 and denom_s > 0:
                correlation = numerator / (denom_q * denom_s)
        
        self.current_session.marker_survival_summary = {
            'total_markers': total,
            'survived_markers': survived,
            'survival_rate': round(survived / total, 3) if total > 0 else 0.0,
            'avg_lifespan': round(sum(lifespans) / total, 1) if total > 0 else 0.0,
            'avg_quality_score': round(sum(quality_scores) / total, 3) if total > 0 else 0.0,
            'quality_vs_survival_correlation': round(correlation, 3),
            'per_region': per_region,
        }
    
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
        
        # Velocity acceleration: compute from trajectory velocities
        # Positive = speeding up, Negative = slowing down
        self._compute_velocity_acceleration()
        
        # Compute zoom analysis from trajectory data
        self._compute_zoom_analysis()
    
    def _compute_velocity_acceleration(self):
        """
        Compute velocity acceleration from trajectory velocity data.
        
        Uses per-track trajectory_velocities to detect if motion is:
        - Accelerating (positive value)
        - Decelerating (negative value)
        - Constant (near zero)
        """
        if not self.current_session or not self.current_session.tracks:
            return
        
        tracks = self.current_session.tracks
        all_accelerations = []
        
        for track in tracks:
            velocities = track.get('trajectory_velocities', [])
            if len(velocities) < 3:
                continue
            
            # Compute acceleration: average change in velocity
            velocity_changes = [velocities[i+1] - velocities[i] for i in range(len(velocities)-1)]
            avg_acceleration = sum(velocity_changes) / len(velocity_changes)
            all_accelerations.append(avg_acceleration)
        
        if all_accelerations:
            # Average across all tracks
            avg_global_acceleration = sum(all_accelerations) / len(all_accelerations)
            self.current_session.optical_flow['velocity_acceleration'] = round(avg_global_acceleration, 6)
    
    def _compute_zoom_analysis(self):
        """
        Compute zoom/dolly detection from existing trajectory data.
        
        Uses the already-collected trajectory samples to detect:
        - Zoom In: tracks converge toward principal point (uniform scale decrease)
        - Zoom Out: tracks diverge from principal point (uniform scale increase)
        - Dolly: same pattern but with high parallax (non-uniform scale)
        
        The key insight: zoom changes scale uniformly, while dolly creates parallax.
        """
        if not self.current_session or not self.current_session.tracks:
            return
        
        tracks = self.current_session.tracks
        
        # Get principal point (default to center if not available)
        pp = self.current_session.camera_intrinsics.get('principal_point', [0.5, 0.5])
        if not pp or len(pp) != 2:
            pp = [0.5, 0.5]
        
        # Group trajectory samples by time index
        per_time_samples = {}  # {time_idx: [distances_from_pp, ...]}
        
        for track in tracks:
            trajectory = track.get('trajectory', [])
            if len(trajectory) < 3:
                continue
            
            for i, pos in enumerate(trajectory):
                if not pos or len(pos) != 2:
                    continue
                
                # Distance from principal point
                dist = ((pos[0] - pp[0])**2 + (pos[1] - pp[1])**2)**0.5
                
                # Skip tracks very close to center (less reliable for scale detection)
                if dist < 0.02:
                    continue
                
                if i not in per_time_samples:
                    per_time_samples[i] = []
                per_time_samples[i].append(dist)
        
        # Need at least 3 time samples with data
        if len(per_time_samples) < 3:
            return
        
        # Compute average distance at each time sample
        time_indices = sorted(per_time_samples.keys())
        avg_distances = []
        for t in time_indices:
            if per_time_samples[t]:
                avg_distances.append(sum(per_time_samples[t]) / len(per_time_samples[t]))
        
        if len(avg_distances) < 3 or avg_distances[0] < 0.01:
            return
        
        # Compute scale timeline (normalize to first frame)
        scale_timeline = [d / avg_distances[0] for d in avg_distances]
        
        # Compute overall metrics
        start_to_end_ratio = scale_timeline[-1] / scale_timeline[0] if scale_timeline[0] > 0 else 1.0
        
        # Scale variance: how much does scale deviate from linear interpolation?
        # Low variance = uniform scaling (zoom), high variance = parallax (dolly)
        if len(scale_timeline) > 2:
            # Linear fit: expected scale at each point
            slope = (scale_timeline[-1] - scale_timeline[0]) / (len(scale_timeline) - 1)
            expected = [scale_timeline[0] + slope * i for i in range(len(scale_timeline))]
            scale_variance = sum((s - e)**2 for s, e in zip(scale_timeline, expected)) / len(scale_timeline)
        else:
            scale_variance = 0.0
        
        # Thresholds for detection
        ZOOM_THRESHOLD = 0.05  # 5% total scale change to be considered a zoom
        VARIANCE_THRESHOLD = 0.01  # Low variance = uniform (zoom-like)
        
        is_zoom_detected = abs(start_to_end_ratio - 1.0) > ZOOM_THRESHOLD
        is_uniform_scale = scale_variance < VARIANCE_THRESHOLD
        
        # Radial convergence: negative = converging (zoom in), positive = diverging (zoom out)
        radial_convergence = (start_to_end_ratio - 1.0)
        radial_convergence_normalized = radial_convergence / max(abs(radial_convergence), 0.001) if radial_convergence != 0 else 0.0
        
        # Determine direction
        if start_to_end_ratio > (1.0 + ZOOM_THRESHOLD):
            zoom_direction = 'ZOOM_OUT'
        elif start_to_end_ratio < (1.0 - ZOOM_THRESHOLD):
            zoom_direction = 'ZOOM_IN'
        else:
            zoom_direction = 'NONE'
        
        # Confidence: based on how many tracks contributed and variance
        tracks_with_trajectory = sum(1 for t in tracks if len(t.get('trajectory', [])) >= 3)
        confidence = min(1.0, tracks_with_trajectory / 10.0) * (1.0 if is_uniform_scale else 0.5)
        
        self.current_session.zoom_analysis = {
            'is_zoom_detected': is_zoom_detected,
            'zoom_direction': zoom_direction,
            'scale_timeline': [round(s, 4) for s in scale_timeline],
            'estimated_fl_ratio': round(start_to_end_ratio, 4),
            'scale_variance': round(scale_variance, 6),
            'is_uniform_scale': is_uniform_scale,
            'radial_convergence': round(radial_convergence_normalized, 4),
            'confidence': round(confidence, 3),
        }
        
        if is_zoom_detected:
            print(f"AutoSolve: Zoom detected - {zoom_direction} "
                  f"(scale: {start_to_end_ratio:.2f}x, "
                  f"{'uniform' if is_uniform_scale else 'parallax'}, "
                  f"confidence: {confidence:.0%})")
    
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
    
    def record_track_failure(self, track_name: str, frame: int, x: float, y: float, reason: str = "LOST"):
        """
        Record when and where a track failed.
        
        Args:
            track_name: Name of the failed track
            frame: Frame number where track lost lock
            x, y: Normalized position of track at failure
            reason: Failure reason (LOST, DRIFT, BLUR, OCCLUSION)
        """
        if not self.current_session:
            return
        
        self.current_session.track_failures.append({
            'track_name': track_name,
            'frame': frame,
            'position': [round(x, 4), round(y, 4)],
            'reason': reason,
        })
    
    def record_visual_features(self, features: Dict):
        """
        Record visual features from FeatureExtractor.
        
        Args:
            features: Dict from FeatureExtractor.to_dict()
        """
        if not self.current_session:
            return
        
        self.current_session.visual_features = features
        
        # Also extract to top-level fields for easy access
        if 'clip_fingerprint' in features:
            self.current_session.clip_fingerprint = features['clip_fingerprint']
        if 'motion_class' in features:
            self.current_session.motion_class = features['motion_class']
        if 'flow_direction_histogram' in features:
            self.current_session.flow_direction_histogram = features['flow_direction_histogram']
        if 'flow_magnitude_histogram' in features:
            self.current_session.flow_magnitude_histogram = features['flow_magnitude_histogram']
    
    def record_clip_fingerprint(self, fingerprint: str):
        """
        Record clip fingerprint for per-clip learning.
        
        Args:
            fingerprint: Hash string identifying this exact clip
        """
        if not self.current_session:
            return
        
        self.current_session.clip_fingerprint = fingerprint
    
    def record_session_linkage(self, previous_session_id: str = "", iteration: int = 1):
        """
        Record session linkage for multi-attempt analysis.
        
        Links this session to the previous session for the same clip,
        enabling analysis of how user iterations improve tracking.
        
        Args:
            previous_session_id: Session ID of the previous attempt on this clip
            iteration: Which attempt this is (1, 2, 3...)
        """
        if not self.current_session:
            return
        
        self.current_session.previous_session_id = previous_session_id
        self.current_session.iteration = iteration
    
    def record_motion_class(self, motion_class: str):
        """
        Record motion classification for sub-classification.
        
        Args:
            motion_class: One of LOW, MEDIUM, HIGH
        """
        if not self.current_session:
            return
        
        self.current_session.motion_class = motion_class
    
    def record_contributor_id(self, contributor_id: str):
        """
        Record anonymous contributor ID for multi-user data distinction.
        
        Args:
            contributor_id: Anonymous ID from get_contributor_id()
        """
        if not self.current_session:
            return
        
        self.current_session.contributor_id = contributor_id
    
    def record_flow_histograms(self, direction_histogram: List[float], magnitude_histogram: List[float]):
        """
        Record optical flow histograms for NN training.
        
        Args:
            direction_histogram: 8-bin histogram of flow directions
            magnitude_histogram: 5-bin histogram of flow magnitudes
        """
        if not self.current_session:
            return
        
        if len(direction_histogram) == 8:
            self.current_session.flow_direction_histogram = direction_histogram
        if len(magnitude_histogram) == 5:
            self.current_session.flow_magnitude_histogram = magnitude_histogram
    
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
        Save user edit session data to JSON file atomically.
        
        Args:
            edit_session: BehaviorData dataclass from BehaviorRecorder
        """
        from dataclasses import asdict
        import tempfile
        
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
            
            # Write to temp file first, then rename (atomic)
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.json',
                dir=edits_dir,
                delete=False
            ) as tmp:
                json.dump(edit_dict, tmp, indent=2)
                tmp_path = tmp.name
            
            # Atomic replace
            os.replace(tmp_path, filepath)
            
            print(f"AutoSolve: Saved edit session to {filepath}")
        except (OSError, IOError, TypeError) as e:
            print(f"AutoSolve: Error saving edit session: {e}")
            # Clean up temp file if it exists
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    def _save_session(self):
        """Save session data to JSON file atomically."""
        if not self.current_session:
            return
        
        import tempfile
        
        try:
            # Generate filename with timestamp and anonymous session ID
            timestamp = self.current_session.timestamp.replace(':', '-').replace('.', '-')
            # clip_name is already an anonymous hash (e.g., "a7f3c2b1")
            session_id = self.current_session.clip_name
            
            filename = f"{timestamp[:19]}_{session_id}.json"
            filepath = self.data_dir / filename
            
            # Convert session to dict and sanitize for JSON serialization
            session_dict = asdict(self.current_session)
            session_dict = self._sanitize_for_json(session_dict)
            
            # Write to temp file first, then rename (atomic)
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.json',
                dir=self.data_dir,
                delete=False
            ) as tmp:
                json.dump(session_dict, tmp, indent=2)
                tmp_path = tmp.name
            
            # Atomic replace
            os.replace(tmp_path, filepath)
            
            print(f"AutoSolve: Saved session data to {filepath}")
        except (OSError, IOError, TypeError) as e:
            print(f"AutoSolve: Error saving session: {e}")
            # Clean up temp file if it exists
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
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
