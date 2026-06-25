# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
SmartTracker with Full Learning Integration.

Hybrid approach:
- Ships with pre-trained defaults (from developer training)
"""

import bpy
import json
import os
import hashlib
import numpy as np
from mathutils import Vector
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Mixins containing extracted methods
from .validation import ValidationMixin
from .filtering import FilteringMixin
from .probe_cache import ProbeCacheMixin
from .detection import DetectionMixin
from .strategic import StrategicMixin
from .learning import LearningMixin
from .cleanup import CleanupMixin

# Analyzer classes extracted to analyzers.py
from .analyzers import TrackStats, RegionStats, CoverageData, TrackAnalyzer, CoverageAnalyzer
# Constants needed for regions
from .constants import REGIONS
# Utility functions
from .utils import get_region, get_region_bounds
# Neural Engine: pixel-based trackability scorer (zero new deps)
from .pixel_analyzer import PixelAnalyzer
# Neural Engine: camera reconstruction readback
from .reconstruction_reader import ReconstructionReader


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# Detection thresholds
DETECTION_THRESHOLD_MULTIPLIER = 0.5  # Multiplier for detection threshold in region detection
PROBE_SURVIVAL_THRESHOLD = 0.7  # 70% survival required for probe markers to be considered successful

# Motion classification thresholds (normalized velocity per frame)
MOTION_HIGH_THRESHOLD = 0.03  # Velocity > 3% of frame = HIGH motion
MOTION_MEDIUM_THRESHOLD = 0.01  # Velocity > 1% of frame = MEDIUM motion

# Velocity multipliers for non-rigid detection
NON_RIGID_VELOCITY_MULT = 3.0  # Region velocity > 3x avg indicates non-rigid (water/waves)

# Region success thresholds
REGION_LOW_SUCCESS = 0.2  # Below 20% = problematic region (skip in detection)
REGION_DEAD_CONFIDENCE = 0.25  # Below 25% confidence = mark as dead zone
REGION_REVIVAL_CONFIDENCE = 0.4  # Above 40% confidence = remove from dead zones

# Temporal dead zone threshold
TEMPORAL_DEAD_ZONE_FAILURES = 3  # 3+ failures in a time segment = dead zone

# Velocity spike detection
VELOCITY_SPIKE_THRESHOLD = 0.1  # 10% of frame displacement in one step = spike
VELOCITY_SPIKE_SEVERE = 0.2  # 20% of frame = severe spike (auto-mute)


# ═══════════════════════════════════════════════════════════════════════════
# PRE-TRAINED DEFAULTS (Developer-tuned baselines)
# ═══════════════════════════════════════════════════════════════════════════

PRETRAINED_DEFAULTS = {
    # By footage class
    'HD_24fps': {
        'pattern_size': 17,
        'search_size': 91,
        'correlation': 0.68,
        'threshold': 0.28,
        'motion_model': 'LocRot',
    },
    'HD_30fps': {
        'pattern_size': 15,
        'search_size': 71,
        'correlation': 0.70,
        'threshold': 0.30,
        'motion_model': 'LocRot',
    },
    'HD_60fps': {
        'pattern_size': 13,
        'search_size': 51,
        'correlation': 0.72,
        'threshold': 0.35,
        'motion_model': 'LocRot',
    },
    '4K_24fps': {
        'pattern_size': 61,
        'search_size': 251,
        'correlation': 0.62,
        'threshold': 0.22,
        'motion_model': 'Affine',
    },
    '4K_30fps': {
        'pattern_size': 55,
        'search_size': 231,
        'correlation': 0.62,
        'threshold': 0.22,
        'motion_model': 'LocRot',
    },
    '4K_60fps': {
        'pattern_size': 49,
        'search_size': 201,
        'correlation': 0.65,
        'threshold': 0.25,
        'motion_model': 'LocRot',
    },
    'SD_24fps': {
        'pattern_size': 13,
        'search_size': 81,
        'correlation': 0.70,
        'threshold': 0.30,
        'motion_model': 'LocRot',
    },
    'SD_30fps': {
        'pattern_size': 11,
        'search_size': 61,
        'correlation': 0.72,
        'threshold': 0.35,
        'motion_model': 'Loc',
    },
}

FOOTAGE_TYPE_ADJUSTMENTS = {
    'AUTO': {},
    'INDOOR': {
        'correlation': 0.72,
        'threshold': 0.30,
    },
    'OUTDOOR': {
        'dead_zones': ['top-center'],
        'threshold': 0.25,
    },
    'DRONE': {
        'search_size_mult': 1.3,
        'pattern_size_mult': 1.2,
        'correlation': 0.60,
        'threshold': 0.20,
        'dead_zones': ['top-left', 'top-center', 'top-right'],
        'motion_model': 'Affine',
    },
    'HANDHELD': {
        'search_size_mult': 1.2,
        'correlation': 0.65,
        'motion_model': 'LocRot',
    },
    'GIMBAL': {
        'search_size_mult': 0.9,
        'correlation': 0.72,
        'threshold': 0.32,
        'motion_model': 'LocRotScale',
    },
    'ACTION': {
        'search_size_mult': 1.5,
        'pattern_size_mult': 1.3,
        'correlation': 0.50,
        'threshold': 0.15,
        'motion_model': 'Affine',
    },
    'VFX': {
        'correlation': 0.75,
        'threshold': 0.35,
        'motion_model': 'LocRotScale',
    },
    'SCREEN': {
        'correlation': 0.80,
        'threshold': 0.20,
        'motion_model': 'LocRot',
    },
    'CINEMATIC': {
        'pattern_size_mult': 1.4,
        'search_size_mult': 1.2,
        'correlation': 0.65,
        'threshold': 0.25,
        'motion_model': 'Affine',
    },
}

PRETRAINED_DEAD_ZONES = {
    'DRONE': ['top-left', 'top-center', 'top-right'],
    'OUTDOOR': ['top-center'],
    'INDOOR': [],
    'AUTO': [],
    'SCREEN': [],
    'CINEMATIC': [],
}

TIERED_SETTINGS = {
    'ultra_aggressive': {
        'pattern_size': 31,
        'search_size': 150,
        'correlation': 0.45,
        'threshold': 0.08,
        'motion_model': 'Affine',
    },
    'aggressive': {
        'pattern_size': 27,
        'search_size': 130,
        'correlation': 0.50,
        'threshold': 0.12,
        'motion_model': 'Affine',
    },
    'moderate': {
        'pattern_size': 21,
        'search_size': 100,
        'correlation': 0.60,
        'threshold': 0.20,
        'motion_model': 'Affine',
    },
    'balanced': {
        'pattern_size': 15,
        'search_size': 71,
        'correlation': 0.70,
        'threshold': 0.30,
        'motion_model': 'LocRot',
    },
    'selective': {
        'pattern_size': 13,
        'search_size': 61,
        'correlation': 0.75,
        'threshold': 0.40,
        'motion_model': 'LocRot',
    },
}

QUALITY_PRESET_SETTINGS = {
    'FAST': {
        'target_tracks': 60,
        'pattern_size_mult': 0.85,
        'search_size_mult': 0.9,
        'correlation': 0.62,
        'cleanup_threshold': 3.5,
        'min_lifespan': 8,
        'max_iterations': 2,
        'replenish_count': 1,
        'motion_model': 'LocRot',
    },
    'BALANCED': {
        'target_tracks': 100,
        'pattern_size_mult': 1.0,
        'search_size_mult': 1.0,
        'correlation': 0.70,
        'cleanup_threshold': 2.5,
        'min_lifespan': 8,
        'max_iterations': 3,
        'replenish_count': 1,
        'motion_model': 'LocRotScale',
    },
    'QUALITY': {
        'target_tracks': 140,
        'pattern_size_mult': 1.25,
        'search_size_mult': 1.15,
        'correlation': 0.75,
        'cleanup_threshold': 1.5,
        'min_lifespan': 10,
        'max_iterations': 4,
        'replenish_count': 2,
        'motion_model': 'LocRotScale',
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# SMART TRACKER (Main Class)
# ═══════════════════════════════════════════════════════════════════════════

class SmartTracker(
    ValidationMixin,
    FilteringMixin,
    ProbeCacheMixin,
    DetectionMixin,
    StrategicMixin,
    LearningMixin,
    CleanupMixin
):
    """
    Adaptive Learning Tracker with Hybrid Model.
    
    Uses:
    1. Pre-trained defaults (shipped with addon)
    2. Per-session analysis (real-time adaptation)
    """
    
    ABSOLUTE_MIN_TRACKS = 12
    SAFE_MIN_TRACKS = 20
    MAX_ITERATIONS = 3
    
    def __init__(self, clip: bpy.types.MovieClip, robust_mode: bool = False, 
                 footage_type: str = 'AUTO', quality_preset: str = 'BALANCED',
                 tripod_mode: bool = False):
        self.clip = clip
        self.tracking = clip.tracking
        self.settings = clip.tracking.settings
        self.robust_mode = robust_mode
        self.footage_type = footage_type
        self.quality_preset = quality_preset
        self.tripod_mode = tripod_mode
        
        # Get quality preset configuration
        self.quality_config = QUALITY_PRESET_SETTINGS.get(quality_preset, 
                                                          QUALITY_PRESET_SETTINGS['BALANCED'])
        
        # Override class constants based on quality preset
        self.MAX_ITERATIONS = self.quality_config.get('max_iterations', 3)
        self.target_tracks = self.quality_config.get('target_tracks', 35)
        self.cleanup_threshold = self.quality_config.get('cleanup_threshold', 2.5)
        self.min_lifespan = self.quality_config.get('min_lifespan', 12)
        self.replenish_count = self.quality_config.get('replenish_count', 1)
        
        # Learning components
        self.analyzer = TrackAnalyzer()
        
        # Predictor for static settings
        from .settings_predictor import SettingsPredictor
        self.predictor = SettingsPredictor()
        
        # Track quality predictor — ONNX-first, numpy fallback
        self.track_predictor = None
        try:
            from .onnx_predictor import OnnxPredictor
            onnx_pred = OnnxPredictor.get_instance()
            if onnx_pred.track_model_available:
                self.track_predictor = onnx_pred
                print("AutoSolve: TrackPredictor using ONNX inference")
            else:
                print("AutoSolve: ONNX track model not found, loading numpy fallback")
        except Exception as e:
            print(f"AutoSolve: OnnxPredictor init failed: {e}")

        if self.track_predictor is None:
            try:
                from .track_predictor import TrackPredictor
                self.track_predictor = TrackPredictor()
                print("AutoSolve: Loaded numpy fallback TrackPredictor")
            except Exception as e:
                print(f"AutoSolve: Failed to initialize TrackPredictor fallback: {e}")

        # Neural Engine: pixel-based trackability scorer
        self.pixel_analyzer = PixelAnalyzer()

        # Neural Engine: camera reconstruction reader
        self.reconstruction_reader = ReconstructionReader()
        
        self.motion_class: Optional[str] = None  # Set after motion probe
        
        # Current session state
        self.resolution_class = self._classify_footage()
        self.footage_class = f"{self.resolution_class}_{footage_type}"
        self.current_settings: Dict = {}
        self.iteration = 0
        self.last_analysis: Optional[Dict] = None
        self.known_dead_zones: Set[str] = set()
        
        self.temporal_dead_zones: Dict[Tuple[int, int], Dict[str, int]] = {}
        
        self.refinement_iteration = 0
        self.best_solve_error = 999.0
        self.best_bundle_count = 0
        
        self.coverage_analyzer = CoverageAnalyzer(
            clip.frame_start,
            clip.frame_start + clip.frame_duration - 1
        )
        
        self.strategic_iteration = 0
        self.MAX_STRATEGIC_ITERATIONS = 5
        
        self.last_survival_rate: float = 1.0
        self.adaptation_count: int = 0
        self.MAX_ADAPTATIONS: int = 3
        
        self.MONITOR_INTERVAL = 10
        if self.robust_mode:
            self.MONITOR_INTERVAL = 5
            self.replenish_count = max(2, self.replenish_count)
        
        self.cached_motion_probe: Optional[Dict] = None
        
        self.region_confidence: Dict[str, float] = {r: 0.5 for r in REGIONS}
        
        self.enable_healing: bool = True
        self.healer = None
        
        self._last_reconstruction_velocity: Optional[Dict] = None
        
        self._try_load_cached_probe()
        self._load_initial_settings()
        
        print(f"AutoSolve: Quality={quality_preset} (targets={self.target_tracks}, "
              f"threshold={self.cleanup_threshold}px, iterations={self.MAX_ITERATIONS})")

    # ─────────────────────────────────────────────────────────────────────────
    # FRAME COORDINATE CONVERSION
    # ─────────────────────────────────────────────────────────────────────────
    
    def scene_to_clip_frame(self, scene_frame: int) -> int:
        """
        Convert scene frame to clip-relative frame number.
        """
        return scene_frame - self.clip.frame_start + 1
    
    def clip_to_scene_frame(self, clip_frame: int) -> int:
        """
        Convert clip-relative frame to scene frame number.
        """
        return clip_frame + self.clip.frame_start - 1

    # ─────────────────────────────────────────────────────────────────────────
    # CORE TRACKER ACTIONS & RUNTIME MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def configure_settings(self):
        """Apply current settings to Blender's tracker."""
        s = self.settings
        
        if hasattr(s, 'default_pattern_size'):
            s.default_pattern_size = self.current_settings.get('pattern_size', 15)
        if hasattr(s, 'default_search_size'):
            s.default_search_size = self.current_settings.get('search_size', 71)
        if hasattr(s, 'default_correlation_min'):
            s.default_correlation_min = self.current_settings.get('correlation', 0.7)
        if hasattr(s, 'default_motion_model'):
            s.default_motion_model = self.current_settings.get('motion_model', 'LocRot')
        if hasattr(s, 'use_default_normalization'):
            s.use_default_normalization = True
        if hasattr(s, 'use_default_brute'):
            s.use_default_brute = True
        
        print(f"AutoSolve: Configured - Pattern: {self.current_settings.get('pattern_size')}px, "
              f"Search: {self.current_settings.get('search_size')}px, "
              f"Corr: {self.current_settings.get('correlation'):.2f}")
    
    def clear_tracks(self):
        """Clear all tracks."""
        for track in self.tracking.tracks:
            track.select = True
        try:
            self._run_ops(bpy.ops.clip.delete_track)
        except Exception:
            pass

    def import_external_trajectories(self, trajectories, clip_meta):
        """
        Import CoTracker trajectories as native Blender tracking markers.
        """
        # Clear non-locked tracks to start fresh
        any_selected = False
        for track in self.tracking.tracks:
            if hasattr(track, 'lock') and track.lock:
                track.select = False
            else:
                track.select = True
                any_selected = True
        if any_selected:
            try:
                self._run_ops(bpy.ops.clip.delete_track)
            except Exception as e:
                print(f"AutoSolve: Error clearing tracks before import: {e}")
            
        print(f"AutoSolve: Importing {len(trajectories)} external trajectories...")
        
        for idx, traj in enumerate(trajectories):
            if not traj:
                continue
            
            # Create a new track
            track = self.tracking.tracks.new(name=f"AI_Track_{idx:03d}")
            track.lock = False
            
            # Add markers for each frame in the trajectory
            for f_idx, (nx, ny) in enumerate(traj):
                clip_frame = f_idx + 1
                
                # Check bounds
                if clip_frame < 1 or clip_frame > self.clip.frame_duration:
                    continue
                    
                # Create marker
                marker = track.markers.new(frame=clip_frame)
                marker.co = (nx, 1.0 - ny)
                # Keep it active
                marker.mute = False


    def count_active_tracks(self, frame: int) -> int:
        """Count tracks active at frame."""
        count = 0
        for track in self.tracking.tracks:
            marker = track.markers.find_frame(frame)
            if marker and not marker.mute:
                count += 1
        return count
    
    def detect_features(self, threshold: Optional[float] = None) -> int:
        """Detect features with current settings."""
        thresh = threshold or self.current_settings.get('threshold', 0.3)
        
        self._run_ops(
            bpy.ops.clip.detect_features,
            threshold=thresh,
            min_distance=50,
            margin=10,
            placement='FRAME'
        )
        
        for track in self.tracking.tracks:
            self._apply_track_settings(track)
        
        return len(self.tracking.tracks)
    
    def _apply_track_settings(self, track):
        """Apply settings to a track."""
        if hasattr(track, 'pattern_size'):
            track.pattern_size = self.current_settings.get('pattern_size', 15)
        if hasattr(track, 'search_size'):
            track.search_size = self.current_settings.get('search_size', 71)
        if hasattr(track, 'correlation_min'):
            track.correlation_min = self.current_settings.get('correlation', 0.7)
        if hasattr(track, 'motion_model'):
            track.motion_model = self.current_settings.get('motion_model', 'LocRot')
    
    def select_all_tracks(self):
        """Select all tracks."""
        for track in self.tracking.tracks:
            track.select = True
    
    def track_frame(self, backwards: bool = False) -> dict:
        """Track one frame."""
        self.select_all_tracks()
        
        frame = bpy.context.scene.frame_current
        clip_frame = self.scene_to_clip_frame(frame)
        selected_count = sum(1 for t in self.tracking.tracks if t.select)
        
        active_before = []
        for t in self.tracking.tracks:
            marker = t.markers.find_frame(clip_frame)
            if marker and not marker.mute:
                active_before.append(t.name)
        
        markers_at_frame_before = len(active_before)
        
        self._run_ops(bpy.ops.clip.track_markers, backwards=backwards, sequence=False)
        
        next_frame = frame - 1 if backwards else frame + 1
        next_clip_frame = self.scene_to_clip_frame(next_frame)
        
        active_after = []
        muted_markers = []
        for t in self.tracking.tracks:
            marker = t.markers.find_frame(next_clip_frame)
            if marker:
                if marker.mute:
                    muted_markers.append(t.name)
                else:
                    active_after.append(t.name)
        
        markers_at_next = len(active_after)
        lost_count = markers_at_frame_before - markers_at_next
        
        return {
            'lost_count': lost_count,
            'active_before': active_before,
            'active_after': active_after,
            'muted_markers': muted_markers
        }
    
    def track_sequence(self, start_frame: int, end_frame: int, backwards: bool = False) -> int:
        """
        Track a sequence of frames with per-frame processing.
        """
        if backwards:
            frame_range = range(start_frame, end_frame - 1, -1)
        else:
            frame_range = range(start_frame, end_frame + 1)
        
        frames_tracked = 0
        self.select_all_tracks()
        
        for frame in frame_range:
            bpy.context.scene.frame_set(frame)
            self._run_ops(bpy.ops.clip.track_markers, backwards=backwards, sequence=False)
            frames_tracked += 1
        
        return frames_tracked

    # Footage types that benefit from non-rigid motion filtering
    NON_RIGID_FOOTAGE_TYPES = {'DRONE', 'OUTDOOR', 'ACTION', 'HANDHELD'}
    
    def solve_camera(self, tripod_mode: bool = False) -> bool:
        """
        Solve camera with robustness and quality checks.
        """
        self.sanitize_tracks_before_solve()
        
        if hasattr(self.settings, 'use_tripod_solver'):
            self.settings.use_tripod_solver = tripod_mode
        
        track_count = len(self.tracking.tracks)
        
        def set_refinement(enable: bool):
            props = ['refine_focal_length', 'refine_principal_point', 'refine_k1', 'refine_k2']
            count = 0
            for p in props:
                if hasattr(self.settings, p):
                    setattr(self.settings, p, enable)
                    count += 1
            return count > 0

        original_refinement = {}
        for p in ['refine_focal_length', 'refine_principal_point', 'refine_k1', 'refine_k2']:
            if hasattr(self.settings, p):
                original_refinement[p] = getattr(self.settings, p)

        try:
            print("AutoSolve: Attempting initial camera solve...")
            self._run_ops(bpy.ops.clip.solve_camera)
            
            is_valid = self.tracking.reconstruction.is_valid
            bundle_count = self.get_bundle_count()
            bundle_ratio = bundle_count / max(track_count, 1)
            raw_error = self.tracking.reconstruction.average_error if is_valid else 999.0
            
            quality_fail = is_valid and bundle_ratio < 0.3
            
            if not is_valid or quality_fail or raw_error > 3.0:
                print(f"AutoSolve: Initial solve poor (valid={is_valid}, ratio={bundle_ratio:.0%}, err={raw_error:.2f})")
                print("AutoSolve: Retrying with FOCAL LENGTH REFINEMENT enabled...")
                
                set_refinement(True)
                self._run_ops(bpy.ops.clip.solve_camera)
                
                is_valid = self.tracking.reconstruction.is_valid
                bundle_count = self.get_bundle_count()
                bundle_ratio = bundle_count / max(track_count, 1)
                new_error = self.tracking.reconstruction.average_error if is_valid else 999.0
                
                print(f"AutoSolve: Refined solve result (valid={is_valid}, ratio={bundle_ratio:.0%}, err={new_error:.2f})")
                
                if is_valid and bundle_ratio >= 0.3:
                    print("AutoSolve: Refinement FIXED the solve!")
                else:
                    print("AutoSolve: Refinement failed to improve solve sufficienty.")
                    
                for p, val in original_refinement.items():
                    setattr(self.settings, p, val)

            if is_valid:
                bundle_count = self.get_bundle_count()
                bundle_ratio = bundle_count / max(track_count, 1)
                raw_error = self.tracking.reconstruction.average_error

                if bundle_ratio < 0.3:
                    print(f"AutoSolve WARNING: Low quality solve - only {bundle_count}/{track_count} tracks reconstructed")
                    self._solve_quality_failure = True
                    return False

                self._solve_quality_failure = False

                try:
                    poses = self.reconstruction_reader.read_camera_poses(self.clip)
                    if poses:
                        bad_frames = self.reconstruction_reader.detect_pose_jumps(poses)
                        vel_signal = self.reconstruction_reader.camera_velocity_signal(poses)

                        if bad_frames:
                            print(f"AutoSolve: Detected {len(bad_frames)} bad reconstruction frames: "
                                  f"{bad_frames[:5]}{'...' if len(bad_frames) > 5 else ''}")
                            self._mute_tracks_in_bad_frames(bad_frames)

                        self._last_reconstruction_velocity = vel_signal
                        print(f"AutoSolve: Camera velocity — "
                              f"angular={vel_signal['mean_angular_vel']:.2f}°/f, "
                              f"class={vel_signal['motion_class']}")

                except Exception as re_err:
                    print(f"AutoSolve: Reconstruction readback failed: {re_err}")

                return True
            else:
                print("AutoSolve WARNING: Solve failed (is_valid=False)")
                return False

        except Exception as e:
            print(f"AutoSolve: Solve camera failed with error: {e}")
            return False
    
    def get_solve_error(self) -> float:
        """Get solve error, accounting for quality issues."""
        if not self.tracking.reconstruction.is_valid:
            return 999.0
        
        track_count = len(self.tracking.tracks)
        bundle_count = self.get_bundle_count()
        bundle_ratio = bundle_count / max(track_count, 1)
        
        raw_error = self.tracking.reconstruction.average_error
        
        if bundle_ratio < 0.3:
            return 999.0
        elif bundle_ratio < 0.5:
            penalty = (0.5 - bundle_ratio) * 10
            return raw_error + penalty
        else:
            return raw_error
    
    def get_bundle_count(self) -> int:
        return len([t for t in self.tracking.tracks if t.has_bundle])

    def _mute_tracks_in_bad_frames(self, bad_frames: List[int]):
        """
        Neural Engine helper: mute markers on frames flagged as bad reconstruction frames.
        """
        if not bad_frames:
            return
        bad_set = set(bad_frames)
        muted_count = 0
        for track in self.tracking.tracks:
            for marker in track.markers:
                scene_frame = self.clip_to_scene_frame(marker.frame)
                if scene_frame in bad_set and not marker.mute:
                    marker.mute = True
                    muted_count += 1
        if muted_count:
            print(f"AutoSolve: Muted {muted_count} markers in {len(bad_frames)} bad frames")

    def analyze_and_learn(self) -> Dict:
        """Analyze tracks and learn from results."""
        min_life = max(5, self.min_lifespan // 2) if self.robust_mode else self.min_lifespan
        self.last_analysis = self.analyzer.analyze_tracks(self.tracking, min_life)
        self.analyzer.iteration = self.iteration
        
        success_rate = self.last_analysis['success_rate']
        print(f"AutoSolve: Analysis - {self.last_analysis['successful_tracks']}/{self.last_analysis['total_tracks']} "
              f"successful ({success_rate*100:.0f}%)")
        
        if self.last_analysis['dead_zones']:
            print(f"AutoSolve: Dead zones: {', '.join(self.last_analysis['dead_zones'])}")
        
        if self.last_analysis.get('region_stats'):
            self.update_region_confidence(self.last_analysis['region_stats'])
        
        return self.last_analysis
    
    def should_retry(self, analysis: Dict) -> bool:
        """Determine if retry is needed."""
        if self.iteration >= self.MAX_ITERATIONS:
            return False
        
        return analysis['success_rate'] < 0.35
    
    def prepare_retry(self, iteration: Optional[int] = None, keep_settings: bool = False):
        """Prepare for retry with adjusted settings."""
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1
        
        if not keep_settings:
            success_rate = self.last_analysis.get('success_rate', 0.5) if self.last_analysis else 0.5
            
            if success_rate < 0.15:
                tier = 'ultra_aggressive'
            elif success_rate < 0.25:
                tier = 'aggressive'
            elif success_rate < 0.40:
                tier = 'moderate'
            else:
                tier = 'balanced'
            
            self.current_settings = TIERED_SETTINGS[tier].copy()
            print(f"AutoSolve: Retry #{self.iteration} with '{tier}' settings")
        else:
            print(f"AutoSolve: Retry #{self.iteration} preserving diagnostic settings")
        
        self.clear_tracks()
        self.configure_settings()

    def _get_context_override(self):
        """Get context override for operators."""
        context = bpy.context
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'CLIP_EDITOR':
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            for space in area.spaces:
                                if space.type == 'CLIP_EDITOR':
                                    scene_frame = bpy.context.scene.frame_current
                                    space.clip_user.frame_current = scene_frame
                                    break
                            
                            return {
                                'window': window,
                                'screen': window.screen,
                                'area': area,
                                'region': region,
                                'scene': context.scene,
                                'workspace': context.workspace,
                            }
        return {}

    def _run_ops(self, op_func, **kwargs):
        """Run operator with context override."""
        override = self._get_context_override()
        if override:
            with bpy.context.temp_override(**override):
                op_func(**kwargs)
        else:
            op_func(**kwargs)


def sync_scene_to_clip(clip: bpy.types.MovieClip):
    """Sync scene settings to clip."""
    scene = bpy.context.scene
    scene.frame_start = clip.frame_start
    scene.frame_end = clip.frame_start + clip.frame_duration - 1
    
    if clip.fps > 0:
        # Preserve rational framerates (like 23.976, 29.97, 59.94)
        if abs(clip.fps - round(clip.fps)) < 0.001:
            scene.render.fps = round(clip.fps)
            scene.render.fps_base = 1.0
        else:
            fps_val = clip.fps
            if abs(fps_val - 23.976) < 0.01:
                scene.render.fps = 24000
                scene.render.fps_base = 1001.0
            elif abs(fps_val - 29.97) < 0.01:
                scene.render.fps = 30000
                scene.render.fps_base = 1001.0
            elif abs(fps_val - 59.94) < 0.01:
                scene.render.fps = 60000
                scene.render.fps_base = 1001.0
            else:
                # General fallback: round to 3 decimal places
                scene.render.fps = round(fps_val * 1000)
                scene.render.fps_base = 1000.0
    
    if clip.size[0] > 0:
        scene.render.resolution_x = clip.size[0]
        scene.render.resolution_y = clip.size[1]
        scene.render.resolution_percentage = 100
