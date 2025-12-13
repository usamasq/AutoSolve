# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
SmartTracker with Full Learning Integration.

Hybrid approach:
- Ships with pre-trained defaults (from developer training)
- Adapts to user's footage over time (local learning)
"""

import bpy
import json
import os
import hashlib
from mathutils import Vector
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Mixins containing extracted methods
from .validation import ValidationMixin
from .filtering import FilteringMixin
# Analyzer classes extracted to analyzers.py
from .analyzers import TrackStats, RegionStats, CoverageData, TrackAnalyzer, CoverageAnalyzer
# Constants needed for regions
from .constants import REGIONS
# Utility functions
from .utils import get_region, get_region_bounds


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

# These are the "shipped" defaults based on developer testing
# Users can override with local learning data


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
        'pattern_size': 61,      # 2x HD for proper visual scaling on 4K
        'search_size': 251,      # Proportionally larger search for 4K
        'correlation': 0.62,     # Slightly lower for larger patterns
        'threshold': 0.22,
        'motion_model': 'Affine',
    },
    '4K_30fps': {
        'pattern_size': 55,      # 2x HD for proper visual scaling on 4K
        'search_size': 231,      # Proportionally larger search for 4K
        'correlation': 0.62,     # Slightly lower for larger patterns
        'threshold': 0.22,
        'motion_model': 'LocRot',
    },
    '4K_60fps': {
        'pattern_size': 49,      # 2x HD for proper visual scaling on 4K
        'search_size': 201,      # Proportionally larger search for 4K
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

# Footage type specific adjustments (applied on top of resolution defaults)
FOOTAGE_TYPE_ADJUSTMENTS = {
    'AUTO': {
        # No adjustments - use pure resolution/fps defaults
    },
    'INDOOR': {
        # Indoor: usually good lighting, static features
        'correlation': 0.72,
        'threshold': 0.30,
    },
    'OUTDOOR': {
        # Outdoor: variable lighting, possible sky
        'dead_zones': ['top-center'],
        'threshold': 0.25,
    },
    'DRONE': {
        # Drone: lots of parallax, sky issues, fast motion
        'search_size_mult': 1.3,
        'pattern_size_mult': 1.2,
        'correlation': 0.60,
        'threshold': 0.20,
        'dead_zones': ['top-left', 'top-center', 'top-right'],
        'motion_model': 'Affine',
    },
    'HANDHELD': {
        # Handheld: camera shake, variable motion
        'search_size_mult': 1.2,
        'correlation': 0.65,
        'motion_model': 'LocRot',
    },
    'GIMBAL': {
        # Gimbal: smooth, predictable motion, often with dolly/push-in
        'search_size_mult': 0.9,
        'correlation': 0.72,
        'threshold': 0.32,
        'motion_model': 'LocRotScale',  # Handles zoom/dolly scale changes
    },
    'ACTION': {
        # Action: fast motion, motion blur
        'search_size_mult': 1.5,
        'pattern_size_mult': 1.3,
        'correlation': 0.50,
        'threshold': 0.15,
        'motion_model': 'Affine',
    },
    'VFX': {
        # VFX plate: typically well-shot, good markers, often dolly/crane
        'correlation': 0.75,
        'threshold': 0.35,
        'motion_model': 'LocRotScale',  # VFX plates often have camera moves
    },
}

# Known problematic regions (from developer testing)
PRETRAINED_DEAD_ZONES = {
    'DRONE': ['top-left', 'top-center', 'top-right'],
    'OUTDOOR': ['top-center'],
    'INDOOR': [],
    'AUTO': [],
}

# Tiered settings for iterative refinement
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

# Quality preset settings - different presets for different speed/accuracy tradeoffs
QUALITY_PRESET_SETTINGS = {
    'FAST': {
        'target_tracks': 60,           # 2x for averaging (was 30)
        'pattern_size_mult': 0.85,     # Smaller patterns = faster matching
        'search_size_mult': 0.9,       # Smaller search area = faster
        'correlation': 0.62,           # More lenient = fewer rejects
        'cleanup_threshold': 3.5,      # Higher error tolerance
        'min_lifespan': 8,             # Shorter track requirement
        'max_iterations': 2,           # Fewer retry iterations
        'replenish_count': 1,          # Fewer markers per replenish
        'motion_model': 'LocRot',      # Good balance for speed
    },
    'BALANCED': {
        'target_tracks': 100,          # 2x for averaging (was 50)
        'pattern_size_mult': 1.0,
        'search_size_mult': 1.0,
        'correlation': 0.70,
        'cleanup_threshold': 2.5,
        'min_lifespan': 8,
        'max_iterations': 3,
        'replenish_count': 1,
        'motion_model': 'LocRotScale', # Handles most camera moves
    },
    'QUALITY': {
        'target_tracks': 140,          # 2x for averaging (was 70)
        'pattern_size_mult': 1.25,     # Larger patterns = more accurate
        'search_size_mult': 1.15,      # Larger search = better tracking
        'correlation': 0.75,           # Stricter matching = cleaner tracks
        'cleanup_threshold': 1.5,      # Low error tolerance
        'min_lifespan': 10,            # Longer track requirement
        'max_iterations': 4,           # More retry iterations
        'replenish_count': 2,          # More markers per replenish
        'motion_model': 'LocRotScale', # Best quality - handles all transforms
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# SMART TRACKER (Main Class)
# ═══════════════════════════════════════════════════════════════════════════

class SmartTracker(ValidationMixin, FilteringMixin):
    """
    Adaptive Learning Tracker with Hybrid Model.
    
    Uses:
    1. Pre-trained defaults (shipped with addon)
    2. Local learning (adapts to user's footage)
    3. Per-session analysis (real-time adaptation)
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
        
        # Unified SettingsPredictor for all learning
        from .learning.settings_predictor import SettingsPredictor
        self.predictor = SettingsPredictor()
        
        # Session recorder for training data
        from .learning.session_recorder import SessionRecorder
        self.recorder = SessionRecorder()
        
        # Feature extractor for per-clip learning and motion classification
        from .learning.feature_extractor import FeatureExtractor
        self.feature_extractor = FeatureExtractor(clip)
        
        # Extract clip fingerprint immediately (cheap operation)
        self.clip_fingerprint = self.feature_extractor._generate_fingerprint()
        # Sync to feature_extractor for to_dict() export
        self.feature_extractor.features.clip_fingerprint = self.clip_fingerprint
        self.motion_class: Optional[str] = None  # Set after motion probe
        
        # Current session state
        self.resolution_class = self._classify_footage()
        self.footage_class = f"{self.resolution_class}_{footage_type}"
        self.current_settings: Dict = {}
        self.iteration = 0
        self.previous_session_id: str = ""  # For linking sessions across multi-attempts
        self.last_analysis: Optional[Dict] = None
        self.known_dead_zones: Set[str] = set()
        
        # Temporal dead zones: {frame_range: {region: failure_count}}
        # Frame ranges are tuples like (start, end) in 50-frame segments
        self.temporal_dead_zones: Dict[Tuple[int, int], Dict[str, int]] = {}
        
        # Failed tracks for learning (populated after solve attempts)
        self.failed_tracks: List[Dict] = []
        
        # Refinement state
        self.refinement_iteration = 0
        self.best_solve_error = 999.0
        self.best_bundle_count = 0
        
        # Coverage tracking for balanced distribution
        self.coverage_analyzer = CoverageAnalyzer(
            clip.frame_start,
            clip.frame_start + clip.frame_duration - 1
        )
        
        # Strategic tracking state
        self.strategic_iteration = 0
        self.MAX_STRATEGIC_ITERATIONS = 5
        
        # Mid-session adaptation state
        self.adaptation_history: List[Dict] = []
        self.last_survival_rate: float = 1.0
        self.adaptation_count: int = 0
        self.MAX_ADAPTATIONS: int = 3
        
        # Robust mode: more aggressive monitoring
        if self.robust_mode:
            self.MONITOR_INTERVAL = 5  # Check every 5 frames instead of 10
            self.replenish_count = max(2, self.replenish_count)  # At least 2 per region
        
        # Motion probe cache (persisted for session recording)
        self.cached_motion_probe: Optional[Dict] = None
        
        # Region confidence scores (probabilistic dead zones)
        self.region_confidence: Dict[str, float] = {r: 0.5 for r in REGIONS}
        
        # Track healing (enabled by default)
        self.enable_healing: bool = True
        self.healer = None  # Lazy init in heal_tracks()
        
        # Try to load cached probe from disk
        self._try_load_cached_probe()
        
        # Load initial settings
        self._load_initial_settings()
        
        # Log quality configuration
        print(f"AutoSolve: Quality={quality_preset} (targets={self.target_tracks}, "
              f"threshold={self.cleanup_threshold}px, iterations={self.MAX_ITERATIONS})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ANNOTATION PLACEMENT HELPER
    # ─────────────────────────────────────────────────────────────────────────
    
    def _get_feature_placement(self) -> str:
        """
        Get the placement mode for detect_features based on annotation_mode setting.
        
        Returns:
            'FRAME' - detect everywhere
            'INSIDE_GPENCIL' - detect only inside annotation
            'OUTSIDE_GPENCIL' - detect only outside annotation
        """
        try:
            annotation_mode = bpy.context.scene.autosolve.annotation_mode
            if annotation_mode == 'INCLUDE':
                return 'INSIDE_GPENCIL'
            elif annotation_mode == 'EXCLUDE':
                return 'OUTSIDE_GPENCIL'
            else:
                return 'FRAME'
        except Exception:
            return 'FRAME'
    
    def _has_active_annotation(self) -> bool:
        """
        Check if there are active annotations and annotation_mode is set.
        
        Returns True when:
        1. annotation_mode is INCLUDE or EXCLUDE (not NONE)
        2. There is actual gpencil/annotation data with strokes
        
        Used to decide between concentrated vs distributed detection.
        """
        try:
            annotation_mode = bpy.context.scene.autosolve.annotation_mode
            if annotation_mode == 'NONE':
                return False
            
            # Check for actual gpencil data
            gpd = bpy.context.annotation_data
            if gpd and gpd.layers:
                for layer in gpd.layers:
                    if layer.frames:
                        for frame in layer.frames:
                            if frame.strokes:
                                return True
            return False
        except Exception:
            return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # FRAME COORDINATE CONVERSION
    # ─────────────────────────────────────────────────────────────────────────
    
    def scene_to_clip_frame(self, scene_frame: int) -> int:
        """
        Convert scene frame to clip-relative frame number.
        
        Blender's marker API (track.markers.find_frame) uses clip-relative frames,
        where frame 1 is the first frame of the clip, regardless of where the
        clip is positioned in the timeline.
        
        Args:
            scene_frame: Frame number in the scene/timeline
            
        Returns:
            Frame number relative to the clip (1-indexed)
        """
        return scene_frame - self.clip.frame_start + 1
    
    def clip_to_scene_frame(self, clip_frame: int) -> int:
        """
        Convert clip-relative frame to scene frame number.
        
        Args:
            clip_frame: Frame number relative to the clip (1-indexed)
            
        Returns:
            Frame number in the scene/timeline
        """
        return clip_frame + self.clip.frame_start - 1

    # ─────────────────────────────────────────────────────────────────────────
    # PROBE CACHING (Per-Clip Persistence)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _get_probe_cache_path(self) -> Path:
        """Get path for cached probe data."""
        cache_dir = Path(bpy.utils.user_resource('SCRIPTS')) / 'autosolve' / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Hash clip filepath for unique identifier
        # For packed/embedded clips, use clip name + resolution as fallback
        filepath = self.clip.filepath or f"{self.clip.name}_{self.clip.size[0]}x{self.clip.size[1]}"
        clip_hash = hashlib.md5(filepath.encode()).hexdigest()[:8]
        return cache_dir / f"probe_{clip_hash}.json"
    
    def _try_load_cached_probe(self):
        """Try to load cached probe from disk if valid."""
        cache_path = self._get_probe_cache_path()
        if not cache_path.exists():
            return
        
        try:
            data = json.loads(cache_path.read_text())
            
            # Validate cache is for same clip version
            # Skip mtime check for packed/embedded clips (no filepath)
            if self.clip.filepath:
                try:
                    clip_mtime = os.path.getmtime(bpy.path.abspath(self.clip.filepath))
                    if abs(data.get('clip_mtime', 0) - clip_mtime) >= 1:
                        return  # Cache is stale
                except (FileNotFoundError, OSError):
                    pass  # Can't verify mtime, use cache anyway
            self.cached_motion_probe = data.get('probe_results')
            print(f"AutoSolve: Loaded cached probe for {self.clip.name}")
        except Exception as e:
            print(f"AutoSolve: Could not load cached probe: {e}")
    
    def _save_probe_to_cache(self, probe_results: Dict):
        """Save probe results to disk for reuse."""
        try:
            cache_path = self._get_probe_cache_path()
            # Get mtime for external clips, use 0 for packed/embedded
            clip_mtime = 0
            if self.clip.filepath:
                try:
                    clip_mtime = os.path.getmtime(bpy.path.abspath(self.clip.filepath))
                except (FileNotFoundError, OSError):
                    pass
            
            cache_data = {
                'clip_filepath': self.clip.filepath,
                'clip_mtime': clip_mtime,
                'footage_class': self.footage_class,
                'probe_results': probe_results,
            }
            
            cache_path.write_text(json.dumps(cache_data, indent=2))
            print(f"AutoSolve: Cached probe for {self.clip.name}")
        except Exception as e:
            print(f"AutoSolve: Could not cache probe: {e}")
    
    def _classify_footage(self) -> str:
        """Classify footage by resolution and fps."""
        width = self.clip.size[0]
        fps = self.clip.fps if (self.clip.fps is not None and self.clip.fps > 0) else 24
        
        if width >= 3840:
            res = '4K'
        elif width >= 1920:
            res = 'HD'
        else:
            res = 'SD'
        
        if fps >= 50:
            fps_class = '60fps'
        elif fps >= 28:
            fps_class = '30fps'
        else:
            fps_class = '24fps'
        
        return f"{res}_{fps_class}"
    
    def _load_initial_settings(self):
        """
        Load initial settings using the unified SettingsPredictor.
        
        Priority order:
        1. predict_settings (handles resolution, footage type, learned HER, motion, behavior)
        2. Quality preset multipliers (FAST/BALANCED/QUALITY)
        3. Robust mode adjustments (if enabled)
        4. Tripod mode adjustments (if enabled)
        5. Learned dead zones from tracking data
        """
        # Step 1: Use predict_settings as the PRIMARY source
        # This handles: resolution defaults, footage type, learned HER settings, motion, behavior
        # Also uses per-clip fingerprinting and motion sub-classification (if available)
        self.current_settings = self.predictor.predict_settings(
            self.clip,
            robust_mode=False,  # We apply robust mode separately in step 3
            footage_type=self.footage_type,
            motion_class=self.motion_class,  # May be None on first load, set after motion probe
            clip_fingerprint=self.clip_fingerprint
        )
        print(f"AutoSolve: Predicted settings for {self.footage_class}: "
              f"pattern={self.current_settings.get('pattern_size')}px, "
              f"search={self.current_settings.get('search_size')}px, "
              f"corr={self.current_settings.get('correlation', 0.7):.2f}")
        
        # Step 2: Apply quality preset multipliers
        quality_mult = self.quality_config
        self.current_settings['pattern_size'] = int(
            self.current_settings.get('pattern_size', 15) * quality_mult.get('pattern_size_mult', 1.0)
        )
        self.current_settings['search_size'] = int(
            self.current_settings.get('search_size', 71) * quality_mult.get('search_size_mult', 1.0)
        )
        # Quality preset can override correlation
        if 'correlation' in quality_mult:
            # Blend with existing - use max for QUALITY, min for FAST
            if self.quality_preset == 'QUALITY':
                self.current_settings['correlation'] = max(
                    self.current_settings.get('correlation', 0.7),
                    quality_mult['correlation']
                )
            elif self.quality_preset == 'FAST':
                self.current_settings['correlation'] = min(
                    self.current_settings.get('correlation', 0.7),
                    quality_mult['correlation']
                )
        
        # Apply motion_model from quality preset
        if 'motion_model' in quality_mult:
            self.current_settings['motion_model'] = quality_mult['motion_model']
        
        print(f"AutoSolve: After quality preset ({self.quality_preset}): "
              f"pattern={self.current_settings.get('pattern_size')}px, "
              f"search={self.current_settings.get('search_size')}px")
        
        # Step 3: Apply robust mode (more aggressive settings)
        if self.robust_mode:
            self.current_settings['pattern_size'] = int(self.current_settings.get('pattern_size', 15) * 1.4)
            self.current_settings['search_size'] = int(self.current_settings.get('search_size', 71) * 1.4)
            self.current_settings['correlation'] = max(0.45, self.current_settings.get('correlation', 0.7) - 0.15)
            self.current_settings['threshold'] = max(0.08, self.current_settings.get('threshold', 0.3) - 0.12)
            self.current_settings['motion_model'] = 'Affine'
            print(f"AutoSolve: Robust mode - enlarged search areas, lower thresholds")
        
        # Step 4: Apply tripod mode optimizations
        if self.tripod_mode:
            # Tripod shots: camera rotates on axis, minimal parallax
            # Use simpler motion model (Loc only, no rotation in-plane)
            self.current_settings['motion_model'] = 'Loc'
            # Tighter correlation - tripod features should be very stable
            self.current_settings['correlation'] = min(0.80, 
                self.current_settings.get('correlation', 0.7) + 0.08)
            # Prioritize center regions (tripod = pan/tilt from center)
            tripod_dead_zones = {'top-left', 'top-right', 'bottom-left', 'bottom-right'}
            self.known_dead_zones.update(tripod_dead_zones)
            print(f"AutoSolve: Tripod mode - Loc model, tighter correlation, avoiding corners")
        
        # Step 5: Get LEARNED dead zones
        learned_dead_zones = self.predictor.get_dead_zones_for_class(self.footage_class)
        if learned_dead_zones:
            self.known_dead_zones = learned_dead_zones
            print(f"AutoSolve: Using LEARNED dead zones: {', '.join(learned_dead_zones)}")


    
    # ═══════════════════════════════════════════════════════════════════════════
    # MID-SESSION ADAPTATION (Real-time Settings Adjustment)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def adapt_settings_mid_session(self, survival_rate: float) -> Dict:
        """
        Adapt settings based on current session track survival rate.
        
        CONSERVATIVE APPROACH: Never lower correlation threshold.
        Lowering correlation causes markers to accept worse matches and drift
        to incorrect features, making tracking worse not better.
        
        Instead, we only:
        - Increase search_size (helps find features faster)
        - Increase pattern_size (more distinctive patterns)
        - Tighten correlation when things are going well
        
        Adding new markers is handled by monitor_and_replenish(), not here.
        
        Args:
            survival_rate: Current track survival rate (0.0 to 1.0)
            
        Returns:
            Dict with adaptation details and new settings
        """
        if self.adaptation_count >= self.MAX_ADAPTATIONS:
            print(f"AutoSolve: Max adaptations reached ({self.MAX_ADAPTATIONS})")
            return {'adapted': False, 'reason': 'max_adaptations_reached'}
        
        old_settings = self.current_settings.copy()
        adapted = False
        changes = []
        
        # Determine adaptation based on survival rate
        if survival_rate < 0.3:
            # Critical survival - DON'T lower correlation, only increase search area
            # and pattern size to help remaining markers stay locked
            new_search = min(151, int(self.current_settings.get('search_size', 71) * 1.3))
            new_pattern = min(31, int(self.current_settings.get('pattern_size', 15) * 1.2))
            
            if new_search != self.current_settings.get('search_size'):
                self.current_settings['search_size'] = new_search
                changes.append(f"search_size: {old_settings.get('search_size')} → {new_search}")
                adapted = True
            if new_pattern != self.current_settings.get('pattern_size'):
                self.current_settings['pattern_size'] = new_pattern
                changes.append(f"pattern_size: {old_settings.get('pattern_size')} → {new_pattern}")
                adapted = True
            
            # Note: correlation is NOT lowered - that causes drift to wrong features
            # New markers are added by monitor_and_replenish() to compensate for losses
            
        elif survival_rate < 0.5:
            # Poor survival - modest search increase only
            new_search = min(121, int(self.current_settings.get('search_size', 71) * 1.15))
            
            if new_search != self.current_settings.get('search_size'):
                self.current_settings['search_size'] = new_search
                changes.append(f"search_size: {old_settings.get('search_size')} → {new_search}")
                adapted = True
            # Note: correlation is NOT lowered
            
        elif survival_rate > 0.85:
            # Excellent survival - can be more selective (tighten correlation)
            new_corr = min(0.85, self.current_settings.get('correlation', 0.7) + 0.05)
            if new_corr != self.current_settings.get('correlation'):
                self.current_settings['correlation'] = new_corr
                changes.append(f"correlation: {old_settings.get('correlation'):.2f} → {new_corr:.2f} (tighter)")
                adapted = True
        
        if adapted:
            self.adaptation_count += 1
            self.configure_settings()
            
            adaptation_record = {
                'iteration': self.adaptation_count,
                'survival_rate': survival_rate,
                'old_settings': old_settings,
                'new_settings': self.current_settings.copy(),
                'changes': changes,
            }
            self.adaptation_history.append(adaptation_record)
            
            print(f"AutoSolve: MID-SESSION ADAPTATION #{self.adaptation_count}")
            for change in changes:
                print(f"  → {change}")
            
            return {'adapted': True, 'changes': changes, 'new_settings': self.current_settings.copy()}
        
        return {'adapted': False, 'reason': 'survival_rate_acceptable'}
    
    def update_region_confidence(self, region_stats: Dict):
        """
        Update region confidence scores based on tracking results.
        
        Uses exponential moving average for smooth updates:
        new_confidence = 0.7 * old + 0.3 * current_success_rate
        
        Args:
            region_stats: Dict of {region: {total_tracks, successful_tracks}}
        """
        LEARNING_RATE = 0.3
        
        for region, stats in region_stats.items():
            total = stats.get('total_tracks', 0)
            successful = stats.get('successful_tracks', 0)
            
            if total < 2:
                continue  # Not enough data
            
            current_rate = successful / total
            old_confidence = self.region_confidence.get(region, 0.5)
            
            # Exponential moving average
            new_confidence = (1 - LEARNING_RATE) * old_confidence + LEARNING_RATE * current_rate
            self.region_confidence[region] = new_confidence
            
            # Update known_dead_zones based on confidence
            if new_confidence < 0.25:
                self.known_dead_zones.add(region)
            elif new_confidence > 0.4 and region in self.known_dead_zones:
                self.known_dead_zones.discard(region)
        
        # Log significant changes
        low_conf = [r for r, c in self.region_confidence.items() if c < 0.3]
        high_conf = [r for r, c in self.region_confidence.items() if c > 0.7]
        
        if low_conf:
            print(f"AutoSolve: Low confidence regions: {', '.join(low_conf)}")
        if high_conf:
            print(f"AutoSolve: High confidence regions: {', '.join(high_conf)}")
    
    def get_current_survival_rate(self, frame: Optional[int] = None) -> float:
        """
        Calculate current track survival rate.
        
        Args:
            frame: Optional specific frame to check. If None, uses current frame.
            
        Returns:
            Survival rate (0.0 to 1.0)
        """
        if frame is None:
            frame = bpy.context.scene.frame_current
        
        total_tracks = len(self.tracking.tracks)
        if total_tracks == 0:
            return 0.0
        
        active_at_frame = 0
        clip_frame = self.scene_to_clip_frame(frame)  # Convert scene frame to clip-relative
        for track in self.tracking.tracks:
            marker = track.markers.find_frame(clip_frame)
            if marker and not marker.mute:
                active_at_frame += 1
        
        rate = active_at_frame / total_tracks
        self.last_survival_rate = rate
        return rate
    
    # ═══════════════════════════════════════════════════════════════════════
    # ADAPTIVE MONITORING (Real-time health tracking)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Monitoring constants
    MONITOR_INTERVAL = 10  # Check every 10 frames
    SURVIVAL_THRESHOLD = 0.5  # Below 50% → add markers
    CRITICAL_THRESHOLD = 0.3  # Below 30% → adapt settings
    
    def monitor_and_replenish(self, frame: int, backwards: bool = False) -> Dict:
        """
        Real-time monitoring with surgical replenishment.
        
        Called every MONITOR_INTERVAL frames during tracking.
        
        - Adds markers surgically where survival is dropping
        - Adapts settings if survival is critical
        - Records samples for learning
        
        Args:
            frame: Current frame number
            backwards: Whether tracking is going backwards (affects new marker priming)
            
        Returns:
            Dict with monitoring results
        """
        result = {
            'frame': frame,
            'survival_rate': self.get_current_survival_rate(frame),
            'markers_added': 0,
            'adapted': False,
            'changes': [],
        }
        
        # 1. Check if we need to add markers
        if result['survival_rate'] < self.SURVIVAL_THRESHOLD:
            # Identify weak regions at current frame
            weak_regions = self._identify_weak_regions_at_frame(frame)
            
            # Track which markers existed before so we can identify new ones
            existing_tracks = set(self.tracking.tracks)
            
            for region in weak_regions[:3]:  # Max 3 regions per check
                added = self.detect_in_region(region, count=1)
                result['markers_added'] += added
                if added > 0:
                    result['changes'].append(f"+{added} in {region}")
            
            # Re-select all tracks immediately after adding new markers
            # The main tracking loop will track both existing and new markers together
            if result['markers_added'] > 0:
                self.select_all_tracks()
        
        # 2. Adapt settings if critical
        if result['survival_rate'] < self.CRITICAL_THRESHOLD:
            adaptation = self.adapt_settings_mid_session(result['survival_rate'])
            result['adapted'] = adaptation.get('adapted', False)
            if result['adapted']:
                result['changes'].extend(adaptation.get('changes', []))
        
        # 3. Log significant events
        if result['markers_added'] > 0 or result['adapted']:
            print(f"AutoSolve: Frame {frame} - survival: {result['survival_rate']:.0%}, "
                  f"added: {result['markers_added']}, adapted: {result['adapted']}")
        
        return result
    
    def _identify_weak_regions_at_frame(self, frame: int) -> List[str]:
        """
        Identify regions with low track coverage at a specific frame.
        
        Returns:
            List of region names needing more markers
        """
        all_regions = [
            'top-left', 'top-center', 'top-right',
            'mid-left', 'center', 'mid-right',
            'bottom-left', 'bottom-center', 'bottom-right'
        ]
        
        region_counts = {r: 0 for r in all_regions}
        
        for track in self.tracking.tracks:
            marker = track.markers.find_frame(frame)
            if marker and not marker.mute:
                # Determine which region this marker is in
                x, y = marker.co.x, marker.co.y
                region = self._get_region_for_position(x, y)
                if region:
                    region_counts[region] = region_counts.get(region, 0) + 1
        
        # Exclude known dead zones
        for dz in self.known_dead_zones:
            if dz in region_counts:
                del region_counts[dz]
        
        # Sort by count (ascending) and return regions with < 2 markers
        weak = [r for r, count in sorted(region_counts.items(), key=lambda x: x[1]) 
                if count < 2]
        
        return weak
    
    def _get_region_for_position(self, x: float, y: float) -> Optional[str]:
        """Map normalized x,y position to region name."""
        # x, y are 0-1 normalized (or may be absolute - handle both)
        if x > 1 or y > 1:
            # Probably absolute - normalize
            x = x / self.clip.size[0] if self.clip.size[0] else x
            y = y / self.clip.size[1] if self.clip.size[1] else y
        
        # Grid: 3x3
        col = 0 if x < 0.33 else (1 if x < 0.66 else 2)
        row = 2 if y < 0.33 else (1 if y < 0.66 else 0)  # y is inverted in Blender
        
        region_map = [
            ['top-left', 'top-center', 'top-right'],
            ['mid-left', 'center', 'mid-right'],
            ['bottom-left', 'bottom-center', 'bottom-right']
        ]
        
        return region_map[row][col]
    
    def get_adaptation_summary(self) -> Dict:
        """
        Get summary of all mid-session adaptations.
        
        Returns:
            Dict with adaptation history and current state
        """
        return {
            'adaptation_count': self.adaptation_count,
            'max_adaptations': self.MAX_ADAPTATIONS,
            'current_settings': self.current_settings.copy(),
            'region_confidence': self.region_confidence.copy(),
            'adaptation_history': self.adaptation_history,
            'known_dead_zones': list(self.known_dead_zones),
        }
    
    def _blend_settings(self, settings_a: Dict, settings_b: Dict, weight_a: float = 0.5) -> Dict:
        """
        Blend two settings dicts with weighted average.
        
        Args:
            settings_a: First settings dict (e.g., learned settings)
            settings_b: Second settings dict (e.g., current settings)
            weight_a: Weight for settings_a (0.0-1.0)
            
        Returns:
            Blended settings dict
        """
        weight_b = 1.0 - weight_a
        blended = {}
        
        # Numerical settings: weighted average
        for key in ['pattern_size', 'search_size']:
            val_a = settings_a.get(key, 15 if key == 'pattern_size' else 71)
            val_b = settings_b.get(key, 15 if key == 'pattern_size' else 71)
            blended[key] = int(val_a * weight_a + val_b * weight_b) | 1  # Ensure odd
        
        for key in ['correlation', 'threshold']:
            val_a = settings_a.get(key, 0.7 if key == 'correlation' else 0.3)
            val_b = settings_b.get(key, 0.7 if key == 'correlation' else 0.3)
            blended[key] = round(val_a * weight_a + val_b * weight_b, 2)
        
        # Non-numeric: prefer settings_a (assumed to be learned/better)
        blended['motion_model'] = settings_a.get('motion_model', settings_b.get('motion_model', 'LocRot'))
        
        return blended
    
    def _get_learned_skip_regions(self) -> Set[str]:
        """
        Get regions to skip based on historical learning data.
        
        This enables proactive region avoidance before detection.
        Uses SettingsPredictor's region_models for success rate analysis.
        
        Returns:
            Set of region names to skip
        """
        skip = set()
        MIN_SAMPLES = 20  # Need sufficient data to make decision
        SKIP_THRESHOLD = 0.25  # Below 25% success = skip
        
        region_models = self.predictor.model.get('region_models', {})
        
        for region, data in region_models.items():
            # Validate region name
            if region not in REGIONS:
                continue
            
            total = data.get('total_tracks', 0)
            successful = data.get('successful_tracks', 0)
            
            if total >= MIN_SAMPLES:
                rate = successful / total
                if rate < SKIP_THRESHOLD:
                    skip.add(region)
                    
        return skip
    
    
    def extract_training_data(self) -> Dict:
        """
        Extract patterns and data for training/learning.
        
        Captures:
        - Track success/failure rates by region
        - Velocity and jitter profiles
        - Settings that led to this result
        - Solve quality metrics
        
        Returns:
            Dict containing extracted training data
        """
        training_data = {
            'footage_class': self.footage_class,
            'settings_used': self.current_settings.copy(),
            'solve_success': self.tracking.reconstruction.is_valid,
            'solve_error': self.get_solve_error(),
            'track_count': self.get_bundle_count(),
            'region_stats': {},
            'velocity_stats': {},
            'iteration': self.iteration,
        }
        
        # Analyze by region
        region_tracks = {r: {'total': 0, 'success': 0, 'avg_lifespan': 0, 'lifespans': []} 
                        for r in REGIONS}
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            # Get region from average position
            avg_x = sum(m.co.x for m in markers) / len(markers)
            avg_y = sum(m.co.y for m in markers) / len(markers)
            region = get_region(avg_x, avg_y)
            
            markers_sorted = sorted(markers, key=lambda m: m.frame)
            lifespan = markers_sorted[-1].frame - markers_sorted[0].frame
            
            region_tracks[region]['total'] += 1
            region_tracks[region]['lifespans'].append(lifespan)
            if track.has_bundle:
                region_tracks[region]['success'] += 1
        
        # Compute averages
        for region, data in region_tracks.items():
            if data['total'] > 0:
                data['success_rate'] = data['success'] / data['total']
                if data['lifespans']:
                    data['avg_lifespan'] = sum(data['lifespans']) / len(data['lifespans'])
                del data['lifespans']  # Don't store raw data
            training_data['region_stats'][region] = data
        
        # Velocity statistics
        velocities = []
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            markers_sorted = sorted(markers, key=lambda m: m.frame)
            displacement = (Vector(markers_sorted[-1].co) - Vector(markers_sorted[0].co)).length
            duration = markers_sorted[-1].frame - markers_sorted[0].frame
            if duration > 0:
                velocities.append(displacement / duration)
        
        if velocities:
            training_data['velocity_stats'] = {
                'mean': sum(velocities) / len(velocities),
                'max': max(velocities),
                'min': min(velocities),
            }
        
        print(f"AutoSolve: Extracted training data - {training_data['track_count']} bundles, "
              f"{training_data['solve_error']:.2f}px error")
        
        return training_data

    # ═══════════════════════════════════════════════════════════════════════════
    # USER-GUIDED PRIORITY TRACKING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_user_priority_regions(self) -> Dict[str, List[str]]:
        """
        Extract priority regions from user-placed markers.
        
        User markers are detected by:
        - Having only 1-2 markers (just placed, not fully tracked)
        - OR already being fully tracked (user's existing work)
        
        Returns:
            Dict with 'high' and 'existing' priority region lists
        """
        priority = {
            'high': set(),      # Untracked user markers = high priority
            'existing': set(),  # Already tracked = preserve and enhance
        }
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if not markers:
                continue
            
            # Get region from first marker position
            region = get_region(
                markers[0].co.x, markers[0].co.y
            )
            
            if len(markers) <= 2:
                # Just placed, not tracked = high priority
                priority['high'].add(region)
            else:
                # Already tracked = existing work to preserve
                priority['existing'].add(region)
        
        return {k: list(v) for k, v in priority.items()}
    
    def extract_user_templates(self) -> List[Dict]:
        """
        Extract complete settings from user-placed markers.
        
        Extracts:
        - Pattern size, search size
        - Correlation threshold
        - Motion model
        - Region and frame info
        - For tracked markers: velocity, success metrics
        
        Returns:
            List of template dicts, one per user marker
        """
        templates = []
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if not markers:
                continue
            
            # Extract track settings with robust error handling
            pattern_size = (15, 15)  # Default
            search_size = (71, 71)   # Default
            correlation = 0.7
            motion_model = 'LOCATION'
            
            try:
                # Try to get pattern size from track
                if hasattr(track, 'pattern_bound_box'):
                    bb = track.pattern_bound_box
                    if bb and len(bb) >= 4:
                        w = abs(float(bb[0]) - float(bb[2]))
                        h = abs(float(bb[1]) - float(bb[3]))
                        if w > 0 and h > 0:
                            pattern_size = (int(w * self.clip.size[0]), int(h * self.clip.size[1]))
                
                # Try to get search size
                if hasattr(track, 'search_max') and hasattr(track, 'search_min'):
                    sm = track.search_max
                    sn = track.search_min
                    if sm and sn:
                        w = abs(float(sm[0]) - float(sn[0]))
                        h = abs(float(sm[1]) - float(sn[1]))
                        if w > 0 and h > 0:
                            search_size = (int(w * self.clip.size[0]), int(h * self.clip.size[1]))
                
                # Get correlation
                if hasattr(track, 'correlation_min'):
                    correlation = float(track.correlation_min)
                
                # Get motion model
                if hasattr(track, 'motion_model'):
                    motion_model = str(track.motion_model)
                    
            except (TypeError, ValueError, AttributeError) as e:
                # Keep defaults on any error
                pass
            
            template = {
                'name': track.name,
                'pattern_size': pattern_size,
                'search_size': search_size,
                'correlation': correlation,
                'motion_model': motion_model,
                'use_brute': getattr(track, 'use_brute', False),
                'use_normalization': getattr(track, 'use_normalization', False),
                'region': get_region(markers[0].co.x, markers[0].co.y),
                'is_tracked': len(markers) > 2,
            }
            
            # For tracked markers, add metrics
            if len(markers) >= 2:
                markers_sorted = sorted(markers, key=lambda m: m.frame)
                template['frame_start'] = markers_sorted[0].frame
                template['frame_end'] = markers_sorted[-1].frame
                template['lifespan'] = template['frame_end'] - template['frame_start']
                
                # Velocity (average motion per frame)
                total_motion = 0
                for i in range(1, len(markers_sorted)):
                    dx = markers_sorted[i].co.x - markers_sorted[i-1].co.x
                    dy = markers_sorted[i].co.y - markers_sorted[i-1].co.y
                    total_motion += (dx**2 + dy**2) ** 0.5
                template['avg_velocity'] = total_motion / max(len(markers_sorted) - 1, 1)
                
                # Success metrics (if have bundle)
                if track.has_bundle:
                    template['has_bundle'] = True
                    template['solve_error'] = track.average_error
                    template['success'] = track.average_error < 2.0
                else:
                    template['has_bundle'] = False
                    template['success'] = False
            else:
                template['lifespan'] = 0
                template['success'] = None  # Not tracked yet
            
            templates.append(template)
        
        return templates
    
    def learn_from_user_templates(self) -> Dict:
        """
        Analyze user templates and learn optimal settings.
        
        Computes:
        - Best settings by region
        - Success rates by setting combination
        - Recommended settings for each region
        
        Returns:
            Dict with learned settings
        """
        templates = self.extract_user_templates()
        if not templates:
            return {}
        
        # Group by region
        by_region: Dict[str, List[Dict]] = {}
        for t in templates:
            region = t['region']
            if region not in by_region:
                by_region[region] = []
            by_region[region].append(t)
        
        # Analyze each region
        learned = {
            'regions': {},
            'overall': {},
            'success_rate': 0,
            'total_templates': len(templates),
        }
        
        successful = [t for t in templates if t.get('success') is True]
        learned['success_rate'] = len(successful) / max(len([t for t in templates if t.get('success') is not None]), 1)
        
        # Learn from successful tracks
        if successful:
            learned['overall'] = {
                'avg_pattern_size': sum(t['pattern_size'][0] for t in successful) / len(successful),
                'avg_search_size': sum(t['search_size'][0] for t in successful) / len(successful),
                'avg_correlation': sum(t['correlation'] for t in successful) / len(successful),
                'avg_velocity': sum(t.get('avg_velocity', 0) for t in successful) / len(successful),
            }
        
        # Learn per region
        for region, region_templates in by_region.items():
            region_successful = [t for t in region_templates if t.get('success') is True]
            learned['regions'][region] = {
                'template_count': len(region_templates),
                'success_count': len(region_successful),
                'success_rate': len(region_successful) / max(len([t for t in region_templates if t.get('success') is not None]), 1),
            }
            
            if region_successful:
                learned['regions'][region]['recommended'] = {
                    'pattern_size': int(sum(t['pattern_size'][0] for t in region_successful) / len(region_successful)),
                    'search_size': int(sum(t['search_size'][0] for t in region_successful) / len(region_successful)),
                    'correlation': sum(t['correlation'] for t in region_successful) / len(region_successful),
                }
        
        print(f"AutoSolve: Learned from {len(templates)} user templates "
              f"({learned['success_rate']:.0%} success rate)")
        
        return learned
    
    def apply_user_template_settings(self, track, region: str, learned: Dict):
        """
        Apply learned settings to a new track based on region.
        
        Args:
            track: Blender track object
            region: Region name
            learned: Learned settings dict from learn_from_user_templates
        """
        settings = None
        
        # Try region-specific settings first
        if region in learned.get('regions', {}):
            settings = learned['regions'][region].get('recommended')
        
        # Fall back to overall settings
        if not settings and learned.get('overall'):
            settings = {
                'pattern_size': int(learned['overall'].get('avg_pattern_size', 15)),
                'search_size': int(learned['overall'].get('avg_search_size', 71)),
                'correlation': learned['overall'].get('avg_correlation', 0.7),
            }
        
        if settings:
            # Apply to track
            self._apply_track_settings(track)  # Base settings
            
            # Override with learned settings
            if hasattr(track, 'correlation_min'):
                track.correlation_min = settings.get('correlation', 0.7)
            
            # Pattern and search sizes applied at global level
            # Store for next detection
            self.current_settings['pattern_size'] = settings.get('pattern_size', 15)
            self.current_settings['search_size'] = settings.get('search_size', 71)
    
    def save_user_learning(self, learned: Dict):
        """
        Save learned user template data to local model.
        
        Args:
            learned: Learned settings from learn_from_user_templates
        """
        if not learned:
            return
        
        # Merge with existing local learning
        existing = self.predictor.get_data(self.footage_class) or {}
        
        # Update with user template learning
        existing['user_templates'] = {
            'last_updated': bpy.context.scene.frame_current,
            'success_rate': learned.get('success_rate', 0),
            'regions': learned.get('regions', {}),
            'overall': learned.get('overall', {}),
        }
        
        self.predictor.update(self.footage_class, existing)
        print(f"AutoSolve: Saved user template learning for {self.footage_class}")

    
    def preserve_existing_tracks(self) -> int:
        """
        Preserve user's existing tracked markers.
        
        Marks well-tracked existing markers as "protected" so they
        won't be deleted during filtering.
        
        Returns:
            Number of tracks preserved
        """
        preserved = 0
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 5:
                continue
            
            # Check if this is a good track (long lifespan)
            markers_sorted = sorted(markers, key=lambda m: m.frame)
            lifespan = markers_sorted[-1].frame - markers_sorted[0].frame
            
            # Good lifespan = preserve
            if lifespan >= 20:
                # Mark as locked (won't be deleted)
                if hasattr(track, 'lock'):
                    track.lock = True
                preserved += 1
        
        if preserved > 0:
            print(f"AutoSolve: Preserved {preserved} existing well-tracked markers")
        
        return preserved
    
    def enhance_priority_regions(self) -> int:
        """
        Add more markers to user-defined priority regions.
        
        Called when user has placed markers indicating important areas.
        
        Returns:
            Number of additional markers added
        """
        priority = self.get_user_priority_regions()
        added = 0
        
        # High priority regions get extra markers
        for region in priority['high']:
            count = self.detect_in_region(region, count=4)
            added += count
            print(f"AutoSolve: Priority region {region}: +{count} markers")
        
        # Existing tracked regions get maintenance (fill gaps)
        for region in priority['existing']:
            # Check if this region needs more
            self.coverage_analyzer.analyze_tracking(self.tracking)
            for seg, data in self.coverage_analyzer.coverage.get(region, {}).items():
                if data.successful_tracks < 3:
                    count = self.detect_in_region(region, count=2)
                    added += count
                    break
        
        return added

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGIC MARKER PLACEMENT (Industry-Standard Approach)
    # ═══════════════════════════════════════════════════════════════════════════
    

    
    def _is_non_rigid_region(self, region: str) -> bool:
        """
        Check if a region is likely to contain non-rigid objects (waves, water, foliage).
        
        This is a PRE-DETECTION check to avoid placing markers on problematic regions.
        Uses multiple signals:
        1. Low success rate in probe (<20%)
        2. High velocity (>3x average) - fast moving
        3. HIGH JITTER (NEW) - chaotic motion typical of water/waves
        
        Args:
            region: Region name like 'bottom-center'
            
        Returns:
            True if region should be skipped for non-rigid concerns
        """
        # Must have probe data to make this determination (no assumptions)
        if not hasattr(self, 'cached_motion_probe') or not self.cached_motion_probe:
            return False
        
        probe = self.cached_motion_probe
        region_success = probe.get('region_success', {})
        
        # Check if this region had very low success in the probe
        if region in region_success:
            region_data = region_success[region]
            # Handle both dict format (new) and float format (legacy)
            if isinstance(region_data, dict):
                success_rate = region_data.get('success_rate', 1.0)
                if region_data.get('total', 0) > 0:
                    success_rate = region_data.get('success', 0) / region_data['total']
                
                # NEW: Check for high jitter (chaotic motion like waves/water)
                jitters = region_data.get('jitters', [])
                if jitters:
                    avg_jitter = sum(jitters) / len(jitters)
                    # Compare to global average
                    all_jitters = []
                    for r, rd in region_success.items():
                        if isinstance(rd, dict):
                            all_jitters.extend(rd.get('jitters', []))
                    
                    global_avg_jitter = sum(all_jitters) / len(all_jitters) if all_jitters else 0.01
                    
                    # High jitter = chaotic motion (water/waves/foliage)
                    # Region jitter > 2x global average is suspicious
                    if global_avg_jitter > 0 and avg_jitter > global_avg_jitter * 2:
                        print(f"AutoSolve: Skipping {region} - high motion variance "
                              f"({avg_jitter:.4f} >> avg {global_avg_jitter:.4f}) - likely water/waves")
                        return True
                
                # Check velocity variance (inconsistent motion)
                velocities = region_data.get('velocities', [])
                if velocities and len(velocities) >= 2:
                    avg_v = sum(velocities) / len(velocities)
                    if avg_v > 0:
                        variance = sum((v - avg_v)**2 for v in velocities) / len(velocities)
                        coefficient_of_variation = (variance ** 0.5) / avg_v
                        # High CoV = erratic motion
                        if coefficient_of_variation > 0.8:
                            print(f"AutoSolve: Skipping {region} - erratic velocity (CoV={coefficient_of_variation:.2f})")
                            return True
            else:
                success_rate = region_data
            
            if success_rate < 0.2:
                # Probe showed this region is problematic
                print(f"AutoSolve: Skipping {region} - probe showed {success_rate:.0%} success")
                return True
        
        # Check if this region had extremely high velocity (likely non-rigid)
        velocities = probe.get('velocities', {})
        if region in velocities:
            region_velocity = velocities[region]
            avg_velocity = probe.get('avg_velocity', 0.01)
            if avg_velocity > 0 and region_velocity > avg_velocity * 3:
                # Region moves 3x faster than average - likely water/waves
                print(f"AutoSolve: Skipping {region} - velocity {region_velocity:.3f} >> avg {avg_velocity:.3f}")
                return True
        
        # Also skip if in known dead zones from learning
        if region in self.known_dead_zones:
            return True
        
        return False
    
    def detect_in_region(self, region: str, count: int = 3) -> int:
        """
        Detect features within a specific screen region.
        
        Approach: Detect globally with low threshold, then filter to keep
        only features in the target region (up to count).
        
        NOTE: Now includes non-rigid region check for DRONE footage to
        avoid placing markers on likely water/wave regions.
        
        NOTE: For detecting multiple regions, prefer detect_all_regions() 
        which is more efficient (single detection pass).
        
        Args:
            region: Region name (e.g., 'top-left', 'center')
            count: Target number of markers
            
        Returns:
            Number of features detected in this region
        """
        # Phase 6: Skip non-rigid regions (waves, water) during detection
        if self._is_non_rigid_region(region):
            print(f"AutoSolve: Skipping {region} - likely non-rigid (water/waves)")
            return 0
        
        bounds = get_region_bounds(region)
        x_min, y_min, x_max, y_max = bounds
        
        initial_count = len(self.tracking.tracks)
        
        # Detect globally with low threshold to get many candidates
        threshold = self.current_settings.get('threshold', 0.3) * DETECTION_THRESHOLD_MULTIPLIER
        
        try:
            self._run_ops(
                bpy.ops.clip.detect_features,
                threshold=threshold,
                min_distance=50,
                margin=20,
                placement=self._get_feature_placement()
            )
        except Exception as e:
            print(f"AutoSolve: detect_features failed: {e}")
            return 0
        
        # Filter: keep only tracks in target region, limit to count
        new_tracks = list(self.tracking.tracks)[initial_count:]
        
        if not new_tracks:
            return 0
        
        # Categorize tracks by region
        in_region = []
        outside = []
        
        current_frame = bpy.context.scene.frame_current
        clip_frame = self.scene_to_clip_frame(current_frame)  # Convert to clip-relative
        
        for track in new_tracks:
            # Try to find marker at current frame
            marker = track.markers.find_frame(clip_frame)
            
            # If no marker at exact frame, try to get any marker from this track
            if not marker and len(track.markers) > 0:
                marker = track.markers[0]
            
            if not marker:
                outside.append(track)
                continue
            
            # Check if in target region bounds
            x, y = marker.co.x, marker.co.y
            if x_min <= x <= x_max and y_min <= y <= y_max:
                in_region.append(track)
            else:
                outside.append(track)

        # Keep up to 'count' tracks in the region
        kept = 0
        for track in in_region[:count]:
            self._apply_track_settings(track)
            track.select = False
            kept += 1
        
        # Mark excess and outside tracks for deletion
        for track in in_region[count:] + outside:
            track.select = True
        
        # Delete marked tracks
        if in_region[count:] or outside:
            try:
                self._run_ops(bpy.ops.clip.delete_track)
            except:
                pass
        
        return kept
    
    def _detect_concentrated_in_annotation(self) -> Dict[str, int]:
        """
        Detect features concentrated in annotation region.
        
        When annotations are active (INCLUDE or EXCLUDE mode), ignore the
        9-region distribution and let Blender's placement filter handle
        concentration. Uses denser detection parameters for better coverage.
        
        Returns:
            Dict mapping region name to count of features (for logging only)
        """
        initial_count = len(self.tracking.tracks)
        
        # More aggressive detection for denser coverage
        base_threshold = self.current_settings.get('threshold', 0.3)
        threshold = base_threshold * DETECTION_THRESHOLD_MULTIPLIER
        
        placement = self._get_feature_placement()
        
        try:
            self._run_ops(
                bpy.ops.clip.detect_features,
                threshold=threshold,
                min_distance=15,  # Smaller = more dense coverage
                margin=16,
                placement=placement
            )
        except Exception as e:
            print(f"AutoSolve: Concentrated detection failed: {e}")
            return {r: 0 for r in REGIONS}
        
        new_tracks = list(self.tracking.tracks)[initial_count:]
        
        if not new_tracks:
            print("AutoSolve: Concentrated detection found no features")
            return {r: 0 for r in REGIONS}
        
        # Apply settings to all detected tracks (no region-based filtering)
        clip_frame = self.scene_to_clip_frame(bpy.context.scene.frame_current)
        result = {r: 0 for r in REGIONS}
        
        for track in new_tracks:
            self._apply_track_settings(track)
            track.select = False
            
            # Count by region for logging
            marker = track.markers.find_frame(clip_frame)
            if not marker and len(track.markers) > 0:
                marker = track.markers[0]
            if marker:
                region = get_region(marker.co.x, marker.co.y)
                result[region] = result.get(region, 0) + 1
        
        print(f"AutoSolve: Concentrated detection ({placement}): {len(new_tracks)} markers")
        
        # Store for feature extractor
        self._detected_feature_density = result
        
        return result
    
    def detect_all_regions(self, markers_per_region: int = 3, 
                          skip_regions: Optional[Set[str]] = None) -> Dict[str, int]:
        """
        OPTIMIZED: Detect features with single global pass, distribute to all regions.
        
        This is ~60% faster than calling detect_in_region 9 times because it:
        1. Runs detect_features ONCE globally
        2. Categorizes ALL features by region in one pass
        3. Sorts by QUALITY and keeps top N per region
        
        NOTE: When annotations are active, uses concentrated detection instead
        of distributing evenly across regions.
        
        Args:
            markers_per_region: Target markers per region (default 3)
            skip_regions: Optional set of regions to skip (dead zones, etc.)
            
        Returns:
            Dict mapping region name to count of features kept
        """
        # Annotation-aware: concentrate markers instead of distributing
        if self._has_active_annotation():
            return self._detect_concentrated_in_annotation()
        
        skip_regions = skip_regions or set()
        
        # Add non-rigid regions to skip list
        for region in REGIONS:
            if self._is_non_rigid_region(region):
                skip_regions.add(region)
        
        # Also skip known dead zones
        skip_regions.update(self.known_dead_zones)
        
        if skip_regions:
            print(f"AutoSolve: Skipping regions: {', '.join(skip_regions)}")
        
        initial_count = len(self.tracking.tracks)
        
        scene_frame = bpy.context.scene.frame_current
        
        # Use HIGHER threshold for better quality initial features
        # A higher threshold means only strong corners/features are detected
        base_threshold = self.current_settings.get('threshold', 0.3)
        threshold = max(0.4, base_threshold) * DETECTION_THRESHOLD_MULTIPLIER
        
        # Detect with smaller min_distance to get MORE candidates
        # We'll filter by quality later
        try:
            self._run_ops(
                bpy.ops.clip.detect_features,
                threshold=threshold,
                min_distance=25,  # Smaller = more candidates to choose from
                margin=20,        # Slightly larger margin to avoid edge issues
                placement=self._get_feature_placement()
            )
        except Exception as e:
            print(f"AutoSolve: detect_features failed: {e}")
            return {r: 0 for r in REGIONS}
        
        new_tracks = list(self.tracking.tracks)[initial_count:]
        
        # Verify detection was successful
        if not new_tracks:
            print("AutoSolve: No features detected")
            return {r: 0 for r in REGIONS}
        
        if not new_tracks:
            print("AutoSolve: No features detected")
            return {r: 0 for r in REGIONS}
        
        print(f"AutoSolve: Global detection found {len(new_tracks)} candidates")
        
        # Categorize all tracks by region WITH QUALITY SCORE
        tracks_by_region: Dict[str, List[Tuple[Any, float]]] = {r: [] for r in REGIONS}
        detected_per_region: Dict[str, int] = {r: 0 for r in REGIONS}  # For feature density
        no_marker_tracks = []
        
        current_frame = bpy.context.scene.frame_current
        clip_frame = self.scene_to_clip_frame(current_frame)  # Convert to clip-relative
        
        for track in new_tracks:
            marker = track.markers.find_frame(clip_frame)
            if not marker and len(track.markers) > 0:
                marker = track.markers[0]
            
            if not marker:
                no_marker_tracks.append(track)
                continue
            
            # Score the feature based on position quality
            quality = self._score_feature_quality(marker, track)
            
            region = get_region(marker.co.x, marker.co.y)
            tracks_by_region[region].append((track, quality))
            detected_per_region[region] += 1  # Count for feature density
        
        # Store detected counts for feature extractor (before filtering)
        self._detected_feature_density = detected_per_region
        
        # Process each region: SORT BY QUALITY, keep top N
        result: Dict[str, int] = {}
        to_delete = list(no_marker_tracks)  # Always delete tracks without markers
        
        for region in REGIONS:
            region_tracks = tracks_by_region[region]
            
            if region in skip_regions:
                # Skip this region entirely - delete all its tracks
                to_delete.extend([t for t, _ in region_tracks])
                result[region] = 0
                continue
            
            # SORT by quality score (highest first)
            region_tracks.sort(key=lambda x: x[1], reverse=True)
            
            # Keep up to markers_per_region of the BEST quality features
            keep_count = min(len(region_tracks), markers_per_region)
            
            for track, quality in region_tracks[:keep_count]:
                self._apply_track_settings(track)
                track.select = False
            
            # Mark excess for deletion
            to_delete.extend([t for t, _ in region_tracks[keep_count:]])
            result[region] = keep_count
        
        # Single batch deletion
        if to_delete:
            for track in to_delete:
                track.select = True
            try:
                self._run_ops(bpy.ops.clip.delete_track)
            except:
                pass
        
        total = sum(result.values())
        active_regions = sum(1 for c in result.values() if c > 0)
        print(f"AutoSolve: Distributed {total} quality-selected markers across {active_regions}/9 regions")
        
        return result
    
    def _score_feature_quality(self, marker, track) -> float:
        """
        Score a feature by its quality for tracking.
        
        Higher scores indicate better features:
        - Center of frame preferred (more stable tracking)
        - Avoid extreme edges
        - Pattern size affects tracking stability
        
        Returns a score from 0.0 to 1.0
        """
        x, y = marker.co.x, marker.co.y
        
        # Base score - start at 1.0
        score = 1.0
        
        # Penalty for extreme edges (features near edges are less stable)
        edge_margin = 0.08
        if x < edge_margin or x > (1.0 - edge_margin):
            score *= 0.7
        if y < edge_margin or y > (1.0 - edge_margin):
            score *= 0.7
        
        # Small bonus for center region (more parallax information)
        center_dist = ((x - 0.5) ** 2 + (y - 0.5) ** 2) ** 0.5
        if center_dist < 0.25:
            score *= 1.1
        
        # Prefer features not too close to other existing tracks
        # (spatial diversity)
        min_dist_to_existing = self._min_distance_to_existing_tracks(x, y)
        if min_dist_to_existing < 0.03:  # Too close to existing
            score *= 0.6
        elif min_dist_to_existing > 0.1:  # Good distance
            score *= 1.15
        
        return min(score, 1.0)
    
    def _min_distance_to_existing_tracks(self, x: float, y: float) -> float:
        """Calculate minimum distance to existing tracks (that we're keeping)."""
        min_dist = float('inf')
        
        current_frame = bpy.context.scene.frame_current
        clip_frame = self.scene_to_clip_frame(current_frame)
        
        for track in self.tracking.tracks:
            if not track.select:  # Only check tracks we're keeping
                marker = track.markers.find_frame(clip_frame)
                if marker:
                    dist = ((marker.co.x - x) ** 2 + (marker.co.y - y) ** 2) ** 0.5
                    min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 1.0
    

    # Exploratory settings variations for learning what works
    EXPLORATORY_SETTINGS = {
        'top-left': {'pattern_size': 11, 'search_size': 61, 'correlation': 0.75},
        'top-center': {'pattern_size': 15, 'search_size': 71, 'correlation': 0.70},
        'top-right': {'pattern_size': 19, 'search_size': 91, 'correlation': 0.65},
        'mid-left': {'pattern_size': 13, 'search_size': 81, 'correlation': 0.72},
        'center': {'pattern_size': 17, 'search_size': 71, 'correlation': 0.68},
        'mid-right': {'pattern_size': 21, 'search_size': 101, 'correlation': 0.60},
        'bottom-left': {'pattern_size': 15, 'search_size': 91, 'correlation': 0.65},
        'bottom-center': {'pattern_size': 19, 'search_size': 81, 'correlation': 0.70},
        'bottom-right': {'pattern_size': 13, 'search_size': 61, 'correlation': 0.75},
    }
    
    def _get_learned_region_settings(self) -> Dict[str, Dict]:
        """
        Get per-region settings from learning + exploratory baseline.
        
        Combines EXPLORATORY_SETTINGS baseline with learned adjustments
        based on historical region success rates.
        """
        region_settings = {}
        
        # Get region advice from predictor if available
        region_advice = {}
        if hasattr(self, 'predictor') and self.predictor:
            region_advice = self.predictor.get_region_advice()
        
        for region, base_settings in self.EXPLORATORY_SETTINGS.items():
            region_settings[region] = base_settings.copy()
            
            # Apply learned adjustments based on region success
            advice = region_advice.get(region, 'normal')
            
            if advice == 'avoid':
                # Bad region historically: increase search, lower correlation
                region_settings[region]['search_size'] = int(base_settings['search_size'] * 1.5)
                region_settings[region]['correlation'] = max(0.5, base_settings['correlation'] - 0.1)
                region_settings[region]['avoid'] = True
            elif advice == 'prioritize':
                # Good region: can be more selective
                region_settings[region]['correlation'] = min(0.8, base_settings['correlation'] + 0.05)
                region_settings[region]['prioritize'] = True
            # 'normal' or 'unknown' - use base settings
        
        return region_settings
    
    def detect_features_smart(self, markers_per_region: int = 3, use_cached_probe: bool = True) -> int:
        """
        SMART DETECTION
        
        Single entry point that combines the best of exploratory and strategic detection.
        Always uses motion-aware settings and leverages any learned region data.
        
        This replaces the separate detect_exploratory_features / detect_strategic_features
        with one smart approach that:
        1. Uses cached probe results if available
        2. Applies learned region settings when data exists
        3. Falls back to motion-class-based settings otherwise
        
        Args:
            markers_per_region: Target markers per region
            use_cached_probe: Whether to use cached probe results
            
        Returns:
            Total number of features detected
        """
        print(f"AutoSolve: Starting feature detection...")
        
        # Step 1: Get motion classification (use cache or run probe)
        if use_cached_probe and hasattr(self, 'cached_motion_probe') and self.cached_motion_probe:
            probe_results = self.cached_motion_probe
            print(f"AutoSolve: Using cached probe (motion: {probe_results.get('motion_class')})")
            
            # Fix: Ensure motion class is set on instance
            self.motion_class = probe_results.get('motion_class', 'MEDIUM')
            
            # Extract visual features from cached probe results
            try:
                if hasattr(self, 'feature_extractor'):
                    self.feature_extractor.extract_all(tracking_data=probe_results)
                    self.feature_extractor.features.motion_class = self.motion_class
            except Exception as e:
                print(f"AutoSolve: Error extracting features from cache: {e}")
        else:
            probe_results = self._run_motion_probe()
            self.cached_motion_probe = probe_results
            # Save to disk for future sessions
            self._save_probe_to_cache(probe_results)
        
        # Ensure we're at the optimal detection frame (middle of clip for bidirectional tracking)
        detection_frame = self.get_optimal_start_frame()
        bpy.context.scene.frame_set(detection_frame)
        
        motion_class = probe_results.get('motion_class', 'MEDIUM')
        texture_class = probe_results.get('texture_class', 'MEDIUM')
        best_regions = probe_results.get('best_regions', [])
        
        # Step 2: Check for learned region data
        learned_regions = self._get_learned_region_settings()
        has_learned_data = any(
            'prioritize' in v or 'avoid' in v 
            for v in learned_regions.values()
        )
        
        if has_learned_data:
            print(f"AutoSolve: Using learned region settings")
            # Detect using per-region learned settings
            total = self._detect_with_region_settings(learned_regions, markers_per_region, motion_class)
        else:
            # Fall back to motion-class-based detection
            target = markers_per_region if motion_class != 'HIGH' else max(1, markers_per_region - 1)
            total = self._detect_quality_markers(
                motion_class=motion_class,
                texture_class=texture_class,
                markers_per_region=target,
                priority_regions=best_regions
            )
        
        print(f"AutoSolve: Smart detection complete - {total} markers placed")
        
        # Minimum viable check
        if total < 8:
            print(f"AutoSolve: Only {total} markers, adding reinforcements...")
            extra = self._add_reinforcement_markers(total, motion_class)
            total += extra
        
        return total
    
    def _detect_with_region_settings(self, region_settings: Dict[str, Dict], 
                                     markers_per_region: int, motion_class: str) -> int:
        """
        Detect features using per-region learned settings.
        
        Uses EXPLORATORY_SETTINGS adjusted by learning data.
        """
        total = 0
        regions = list(region_settings.keys())
        
        # Sort: prioritized regions first, avoided last
        regions.sort(key=lambda r: (
            0 if region_settings[r].get('prioritize') else
            2 if region_settings[r].get('avoid') else 1
        ))
        
        for region in regions:
            if region in self.known_dead_zones:
                continue
            
            settings = region_settings[region]
            
            # Skip avoided regions in high motion (too risky)
            if settings.get('avoid') and motion_class == 'HIGH':
                continue
            
            # Prioritized regions get extra markers
            count = markers_per_region + 1 if settings.get('prioritize') else markers_per_region
            
            # Apply region-specific settings
            old_settings = self.current_settings.copy()
            self.current_settings.update({
                'pattern_size': settings.get('pattern_size', 15),
                'search_size': settings.get('search_size', 71),
                'correlation': settings.get('correlation', 0.70),
            })
            self.configure_settings()
            
            detected = self.detect_in_region(region, count)
            total += detected
            
            # Restore base settings
            self.current_settings = old_settings
            self.configure_settings()
            
            if detected > 0:
                print(f"AutoSolve: {region}: {detected} markers (learned settings)")
        
        return total
    

    
    def _estimate_motion_quick(self) -> str:
        """
        Quick motion estimate from clip metadata (no tracking needed).
        
        This avoids the expensive full motion probe for obvious cases.
        
        Returns:
            'LOW', 'MEDIUM', or 'HIGH' motion class estimate
        """
        fps = self.clip.fps if self.clip.fps > 0 else 24
        duration = self.clip.frame_duration
        
        # Higher FPS = less motion per frame (smoother footage)
        if fps >= 50:
            fps_class = 'LOW'
        elif fps >= 28:
            fps_class = 'MEDIUM'
        else:
            fps_class = 'HIGH'  # 24fps often has more apparent motion
        
        # Short clips often have dramatic motion
        if duration < 100:
            duration_class = 'HIGH'
        elif duration < 300:
            duration_class = 'MEDIUM'
        else:
            duration_class = 'LOW'
        
        # Footage type hints
        if self.footage_type in ['DRONE', 'ACTION', 'HANDHELD']:
            type_class = 'HIGH'
        elif self.footage_type in ['INDOOR', 'TRIPOD']:
            type_class = 'LOW'
        else:
            type_class = 'MEDIUM'
        
        # Combine: take highest motion estimate
        classes = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        max_class = max([fps_class, duration_class, type_class], key=lambda x: classes[x])
        
        return max_class
    
    def _run_motion_probe(self) -> dict:
        """
        Run a quick motion probe to analyze footage characteristics.
        
        OPTIMIZATION: Now checks quick estimate first and skips full probe
        when not needed (for LOW/MEDIUM motion without robust mode).
        
        Places 1 marker per region, tracks ~20 frames, measures:
        - Average motion velocity
        - Motion variance (jitter)
        - Region success rates
        
        Returns:
            Dict with motion_class, texture_class, best_regions
        """
        # Quick estimation first (no tracking needed)
        quick_class = self._estimate_motion_quick()
        
        # For low/medium motion and no robust mode, skip expensive full probe
        if quick_class != 'HIGH' and not self.robust_mode:
            print(f"AutoSolve: Quick motion estimate: {quick_class} (skipping full probe)")
            # Set motion_class for per-clip learning
            self.motion_class = quick_class
            quick_result = {
                'success': True,
                'motion_class': quick_class,
                'texture_class': 'MEDIUM',
                'best_regions': ['center', 'mid-left', 'mid-right', 'bottom-center'],
                'velocities': {},
                'region_success': {},
                'probe_type': 'quick_estimate'
            }
            self.cached_motion_probe = quick_result.copy()
            
            # Extract visual features for feature density
            try:
                if hasattr(self, 'feature_extractor'):
                    self.feature_extractor.extract_all(tracking_data=quick_result)
                    self.feature_extractor.features.motion_class = self.motion_class
                    print(f"AutoSolve: Visual features extracted (quick path)")
            except Exception as e:
                print(f"AutoSolve: Visual feature extraction skipped: {e}")
                
            return quick_result
        
        print(f"AutoSolve: Running full motion probe (quick estimate: {quick_class})")
        
        result = {
            'success': False,
            'motion_class': quick_class,  # Use quick estimate as baseline
            'texture_class': 'MEDIUM',
            'best_regions': [],
            'velocities': {},
            'region_success': {},
            'probe_type': 'full_probe'
        }
        
        # Save current frame
        original_frame = bpy.context.scene.frame_current
        probe_start = self.clip.frame_start + (self.clip.frame_duration // 4)  # Start at 25%
        
        # NOTE: Don't clear tracks here - let existing tracks be analyzed if any
        # This prevents wasting user-placed markers
        
        # Probe settings: very aggressive to catch motion
        probe_settings = {
            'pattern_size': 21,
            'search_size': 121,  # Large search for testing
            'correlation': 0.55,  # Low correlation to not lose tracks
            'threshold': 0.15,
        }
        
        # Place 1 probe marker per region
        regions = REGIONS.copy()
        import random
        random.shuffle(regions)
        
        probe_count = 0
        for region in regions[:5]:  # Only probe 5 regions for speed
            bpy.context.scene.frame_set(probe_start)
            
            # Apply probe settings
            self.current_settings = probe_settings.copy()
            self.configure_settings()
            
            # Try to detect 1 marker in this region
            detected = self.detect_in_region(region, count=1)
            if detected > 0:
                probe_count += 1
        
        if probe_count < 3:
            print(f"AutoSolve: Probe failed - only {probe_count} markers placed")
            self.clear_tracks()
            return result
        
        # Track forward for 20 frames
        print(f"AutoSolve: Probe tracking {probe_count} markers for 20 frames...")
        self.select_all_tracks()
        
        probe_frames = min(20, self.clip.frame_duration // 4)
        bpy.context.scene.frame_set(probe_start)
        
        for i in range(probe_frames):
            self.track_frame(backwards=False)
            bpy.context.scene.frame_set(probe_start + i + 1)
        
        # Analyze probe results
        velocities = []
        jitters = []
        region_success = {}
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 3:
                continue
            
            markers_sorted = sorted(markers, key=lambda m: m.frame)
            
            # Calculate velocity
            total_displacement = 0
            for i in range(1, len(markers_sorted)):
                dx = markers_sorted[i].co.x - markers_sorted[i-1].co.x
                dy = markers_sorted[i].co.y - markers_sorted[i-1].co.y
                total_displacement += (dx**2 + dy**2) ** 0.5
            
            avg_velocity = total_displacement / len(markers_sorted)
            velocities.append(avg_velocity)
            
            # Calculate jitter (variance in velocity)
            if len(markers_sorted) > 3:
                frame_velocities = []
                for i in range(1, len(markers_sorted)):
                    dx = markers_sorted[i].co.x - markers_sorted[i-1].co.x
                    dy = markers_sorted[i].co.y - markers_sorted[i-1].co.y
                    frame_velocities.append((dx**2 + dy**2) ** 0.5)
                
                if frame_velocities:
                    mean_v = sum(frame_velocities) / len(frame_velocities)
                    variance = sum((v - mean_v)**2 for v in frame_velocities) / len(frame_velocities)
                    jitter = variance ** 0.5
                    jitters.append(jitter)
                    
                    # Store per-region jitter (key for water/wave detection)
                    if region not in region_success:
                        region_success[region] = {'total': 0, 'success': 0, 'jitters': [], 'velocities': []}
                    region_success[region].setdefault('jitters', []).append(jitter)
                    region_success[region].setdefault('velocities', []).append(avg_velocity)
            
            # Track region success
            avg_x = sum(m.co.x for m in markers_sorted) / len(markers_sorted)
            avg_y = sum(m.co.y for m in markers_sorted) / len(markers_sorted)
            region = get_region(avg_x, avg_y)
            
            lifespan = len(markers_sorted)
            if region not in region_success:
                region_success[region] = {'total': 0, 'success': 0, 'jitters': [], 'velocities': []}
            region_success[region]['total'] += 1
            if lifespan >= probe_frames * 0.7:  # 70% survival
                region_success[region]['success'] += 1
        
        # Classify motion
        if velocities:
            avg_motion = sum(velocities) / len(velocities)
            if avg_motion > 0.03:
                result['motion_class'] = 'HIGH'
            elif avg_motion > 0.01:
                result['motion_class'] = 'MEDIUM'
            else:
                result['motion_class'] = 'LOW'
            
            result['velocities'] = {
                'avg': avg_motion,
                'max': max(velocities) if velocities else 0,
            }
        
        # Classify texture (based on how many features we could detect)
        if probe_count >= 4:
            result['texture_class'] = 'HIGH'
        elif probe_count >= 2:
            result['texture_class'] = 'MEDIUM'
        else:
            result['texture_class'] = 'LOW'
        
        # Find best regions
        best_regions = []
        for region, stats in region_success.items():
            if stats['total'] > 0:
                rate = stats['success'] / stats['total']
                if rate >= 0.5:
                    best_regions.append(region)
        
        result['best_regions'] = best_regions if best_regions else ['center']
        result['region_success'] = region_success
        result['success'] = True
        
        # Cache the probe results for session recording
        self.cached_motion_probe = result.copy()
        
        # Set motion_class for per-clip learning and sub-classification
        self.motion_class = result.get('motion_class', 'MEDIUM')
        print(f"AutoSolve: Motion class set to {self.motion_class}")
        
        # Extract visual features for ML training data
        try:
            if hasattr(self, 'feature_extractor'):
                # Extract all visual features for ML training
                override = self._get_context_override()
                if override:
                    with bpy.context.temp_override(**override):
                        self.feature_extractor.extract_all(tracking_data=result)
                else:
                    self.feature_extractor.extract_all(tracking_data=result)
                # Sync motion class to feature extractor
                self.feature_extractor.features.motion_class = self.motion_class
                print(f"AutoSolve: Visual features extracted")
        except Exception as e:
            print(f"AutoSolve: Visual feature extraction skipped: {e}")
        
        # Clear probe tracks
        self.clear_tracks()
        
        # Restore frame
        bpy.context.scene.frame_set(original_frame)
        
        return result
    
    def _detect_quality_markers(self, motion_class: str, texture_class: str,
                                markers_per_region: int, priority_regions: list = None) -> int:
        """
        Place quality markers based on motion/texture analysis.
        
        Uses appropriate settings based on motion class.
        OPTIMIZED: Now uses detect_all_regions for single-pass detection.
        """
        # Check if 4K resolution - need larger search areas
        is_4k = self.clip.size[0] >= 3840
        resolution_multiplier = 1.5 if is_4k else 1.0
        
        # Settings based on motion class
        if motion_class == 'HIGH':
            settings = {
                'pattern_size': int(25 * resolution_multiplier),  # Larger pattern for stability
                'search_size': int(141 * resolution_multiplier),  # Much larger search
                'correlation': 0.55,  # More lenient matching
                'threshold': 0.20,
                'motion_model': 'Affine',
            }
        elif motion_class == 'MEDIUM':
            settings = {
                'pattern_size': int(19 * resolution_multiplier),
                'search_size': int(101 * resolution_multiplier),
                'correlation': 0.65,
                'threshold': 0.25,
                'motion_model': 'LocRot',
            }
        else:  # LOW
            settings = {
                'pattern_size': int(15 * resolution_multiplier),
                'search_size': int(71 * resolution_multiplier),
                'correlation': 0.72,
                'threshold': 0.30,
                'motion_model': 'Loc',
            }
        
        # Adjust for low texture
        if texture_class == 'LOW':
            settings['threshold'] *= 0.6  # More sensitive detection
            settings['correlation'] -= 0.1  # More lenient matching
        
        # Apply settings
        self.current_settings = settings.copy()
        self.configure_settings()
        
        print(f"AutoSolve: Quality settings - Pattern:{settings['pattern_size']}, "
              f"Search:{settings['search_size']}, Corr:{settings['correlation']:.2f}"
              f"{' (4K scaled)' if is_4k else ''}")
        
        # OPTIMIZED: Use single-pass detection for all regions
        # Priority regions get +1 marker handled via per-region counts
        region_results = self.detect_all_regions(markers_per_region=markers_per_region)
        
        # Log priority regions
        if priority_regions:
            priority_found = sum(region_results.get(r, 0) for r in priority_regions)
            print(f"AutoSolve: Priority regions ({', '.join(priority_regions)}): {priority_found} markers")
        
        return sum(region_results.values())

    
    def _add_reinforcement_markers(self, current_count: int, motion_class: str) -> int:
        """
        Add reinforcement markers if we don't have enough.
        Focus on center regions which are usually most reliable.
        """
        needed = max(0, 12 - current_count)  # Aim for 12 total
        if needed == 0:
            return 0
        
        print(f"AutoSolve: Adding {needed} reinforcement markers...")
        
        # Focus on reliable regions
        reliable_regions = ['center', 'mid-left', 'mid-right', 'bottom-center']
        
        added = 0
        for region in reliable_regions:
            if added >= needed:
                break
            detected = self.detect_in_region(region, count=2)
            added += detected
        
        return added
    
    def _apply_exploratory_track_settings(self, track, region: str):
        """Apply region-specific exploratory settings to a track."""
        settings = self.EXPLORATORY_SETTINGS.get(region, self.current_settings)
        
        if hasattr(track, 'pattern_size'):
            track.pattern_size = settings.get('pattern_size', 15)
        if hasattr(track, 'search_size'):
            track.search_size = settings.get('search_size', 71)
        if hasattr(track, 'correlation_min'):
            track.correlation_min = settings.get('correlation', 0.7)
        if hasattr(track, 'motion_model'):
            track.motion_model = settings.get('motion_model', 'LocRot')
    
    def get_optimal_start_frame(self) -> int:
        """
        Get the optimal frame to start detection/tracking from.
        
        Starting from the middle allows bidirectional tracking,
        ensuring early frames get properly covered instead of
        only being covered during backfilling.
        
        Returns:
            Frame number to start from (typically middle of clip)
        """
        frame_start = self.clip.frame_start
        frame_end = frame_start + self.clip.frame_duration - 1
        
        # For very short clips (< 60 frames), start at beginning
        if self.clip.frame_duration < 60:
            return frame_start
        
        # For normal clips, start at the middle
        # This ensures both directions get equal attention
        middle_frame = frame_start + (self.clip.frame_duration // 2)
        
        print(f"AutoSolve: Optimal start frame: {middle_frame} "
              f"(range: {frame_start}-{frame_end})")
        
        return middle_frame
    
    def fill_coverage_gaps(self) -> Dict:
        """
        Fill gaps in coverage by adding markers to weak zones.
        
        Called after initial tracking pass to ensure balanced distribution.
        
        Returns:
            Dict with:
                - markers_added: Number of new markers added
                - detection_frames: List of frames where markers were detected
        """
        result = {
            'markers_added': 0,
            'detection_frames': [],
        }
        
        # Analyze current coverage
        self.coverage_analyzer.analyze_tracking(self.tracking)
        summary = self.coverage_analyzer.get_coverage_summary()
        
        if summary['is_balanced']:
            print(f"AutoSolve: Coverage is balanced ({summary['regions_with_tracks']}/9 regions)")
            return result
        
        # Get weak zones (regions needing more tracks)
        weak_zones = self.coverage_analyzer.get_weak_zones()
        if not weak_zones:
            print("AutoSolve: No weak zones identified")
            return result
        
        processed_regions = set()
        
        # Process weak zones, limiting by segment to target specific time ranges
        for region, segment in weak_zones[:5]:  # Limit to top 5 priorities
            if region in processed_regions:
                continue
            
            # Go to the segment's start frame
            target_frame = segment[0]
            bpy.context.scene.frame_set(target_frame)
            
            # Detect in this region
            added = self.detect_in_region(region, count=2)
            result['markers_added'] += added
            if added > 0:
                result['detection_frames'].append(target_frame)
            processed_regions.add(region)
            
            print(f"AutoSolve: Added {added} markers to {region} at frame {target_frame}")
        
        return result
    
    def get_coverage_analysis(self) -> Dict:
        """
        Analyze current coverage and return summary.
        
        Returns:
            Dict with coverage metrics
        """
        self.coverage_analyzer.analyze_tracking(self.tracking)
        return self.coverage_analyzer.get_coverage_summary()
    
    def is_coverage_balanced(self) -> bool:
        """Check if current tracking has balanced coverage."""
        summary = self.get_coverage_analysis()
        return summary['is_balanced']
    
    def strategic_track_iteration(self) -> Dict:
        """
        Perform one iteration of strategic tracking.
        
        1. Analyze current coverage
        2. Identify weak zones
        3. Add markers to weak zones
        4. Track those new markers
        
        Returns:
            Dict with iteration results including detection_frames for bidirectional tracking
        """
        self.strategic_iteration += 1
        print(f"AutoSolve: Strategic iteration {self.strategic_iteration}")
        
        # Analyze coverage
        summary = self.get_coverage_analysis()
        
        result = {
            'iteration': self.strategic_iteration,
            'coverage_before': summary.copy(),
            'markers_added': 0,
            'detection_frames': [],
            'coverage_after': None,
        }
        
        if summary['is_balanced']:
            print("AutoSolve: Coverage is balanced, no more iterations needed")
            return result
        
        # Fill gaps and get detection info
        gap_result = self.fill_coverage_gaps()
        result['markers_added'] = gap_result['markers_added']
        result['detection_frames'] = gap_result['detection_frames']
        
        # Re-analyze
        result['coverage_after'] = self.get_coverage_analysis()
        
        return result
    
    def verify_full_timeline_coverage(self) -> Dict:
        """
        Verify that all surviving tracks cover the full timeline.
        
        This is the key to ensuring no gaps remain at start/end frames.
        
        Returns:
            Dict with:
                - needs_backward_extension: Tracks missing early frames
                - needs_forward_extension: Tracks missing late frames
                - earliest_track_start: Earliest frame where a track starts
                - latest_track_end: Latest frame where a track ends
                - recommended_action: 'none', 'extend_backward', 'extend_forward', 'extend_both'
        """
        frame_start = self.clip.frame_start
        frame_end = frame_start + self.clip.frame_duration - 1
        
        result = {
            'needs_backward_extension': [],
            'needs_forward_extension': [],
            'earliest_track_start': frame_end,
            'latest_track_end': frame_start,
            'total_tracks': 0,
            'fully_covered_tracks': 0,
            'recommended_action': 'none',
        }
        
        # Tolerance: tracks don't need to reach exact frame_start/frame_end
        # Allow 5 frame margin at each end
        MARGIN = 5
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            result['total_tracks'] += 1
            
            markers_sorted = sorted(markers, key=lambda m: m.frame)
            track_start = markers_sorted[0].frame
            track_end = markers_sorted[-1].frame
            
            result['earliest_track_start'] = min(result['earliest_track_start'], track_start)
            result['latest_track_end'] = max(result['latest_track_end'], track_end)
            
            # Check if track needs extension
            needs_backward = track_start > frame_start + MARGIN
            needs_forward = track_end < frame_end - MARGIN
            
            if needs_backward:
                result['needs_backward_extension'].append({
                    'name': track.name,
                    'current_start': track_start,
                    'target_start': frame_start,
                })
            
            if needs_forward:
                result['needs_forward_extension'].append({
                    'name': track.name,
                    'current_end': track_end,
                    'target_end': frame_end,
                })
            
            if not needs_backward and not needs_forward:
                result['fully_covered_tracks'] += 1
        
        # Determine recommended action
        if result['needs_backward_extension'] and result['needs_forward_extension']:
            result['recommended_action'] = 'extend_both'
        elif result['needs_backward_extension']:
            result['recommended_action'] = 'extend_backward'
        elif result['needs_forward_extension']:
            result['recommended_action'] = 'extend_forward'
        else:
            result['recommended_action'] = 'none'
        
        coverage_pct = result['fully_covered_tracks'] / max(result['total_tracks'], 1) * 100
        print(f"AutoSolve: Timeline coverage: {result['fully_covered_tracks']}/{result['total_tracks']} tracks "
              f"({coverage_pct:.0f}%) cover full range")
        
        if result['needs_backward_extension']:
            print(f"AutoSolve: {len(result['needs_backward_extension'])} tracks need backward extension")
        if result['needs_forward_extension']:
            print(f"AutoSolve: {len(result['needs_forward_extension'])} tracks need forward extension")
        
        return result
    
    def should_continue_strategic(self) -> bool:
        """
        Determine if more strategic iterations are needed.
        
        Returns:
            True if more iterations needed
        """
        if self.strategic_iteration >= self.MAX_STRATEGIC_ITERATIONS:
            print(f"AutoSolve: Max strategic iterations reached ({self.MAX_STRATEGIC_ITERATIONS})")
            return False
        
        # Check coverage
        if self.is_coverage_balanced():
            print("AutoSolve: Coverage balanced, stopping strategic iterations")
            return False
        
        return True
    
    def remove_clustered_tracks(self) -> int:
        """
        Remove tracks from over-represented regions to improve balance.
        
        Called before final solve to ensure distribution requirements.
        
        Returns:
            Number of tracks removed
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
            
            # Find tracks in this region and remove excess
            tracks_in_region = []
            for track in self.tracking.tracks:
                markers = [m for m in track.markers if not m.mute]
                if len(markers) < 2:
                    continue
                
                avg_x = sum(m.co.x for m in markers) / len(markers)
                avg_y = sum(m.co.y for m in markers) / len(markers)
                if get_region(avg_x, avg_y) == region:
                    # Prioritize removing shorter tracks
                    lifespan = len(markers)
                    tracks_in_region.append((track.name, lifespan))
            
            # Sort by lifespan (shortest first)
            tracks_in_region.sort(key=lambda x: x[1])
            
            # Remove excess
            to_remove = set(name for name, _ in tracks_in_region[:excess])
            for track in self.tracking.tracks:
                track.select = track.name in to_remove
            
            if to_remove:
                try:
                    self._run_ops(bpy.ops.clip.delete_track)
                    removed += len(to_remove)
                    print(f"AutoSolve: Removed {len(to_remove)} excess tracks from {region}")
                except:
                    pass
        
        return removed

    # ═══════════════════════════════════════════════════════════════════════════
    # TRACK HEALING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def extend_lost_tracks(self, min_extension: int = 10) -> int:
        """
        Extend tracks that stopped tracking before the clip ends.
        
        This is simpler than full healing - it just re-tracks from where
        tracking was lost, using more tolerant settings.
        
        Args:
            min_extension: Minimum frames a track must be from clip edges
                          to be considered "lost" (default: 10 frames)
        
        Returns:
            Number of tracks successfully extended
        """
        if not self.clip or not self.tracking:
            return 0
        
        clip_start = self.clip.frame_start
        clip_end = clip_start + self.clip.frame_duration - 1
        
        # Find tracks that stopped early (didn't reach clip edges)
        lost_tracks = []
        
        for track in self.tracking.tracks:
            try:
                markers = [m for m in track.markers if not m.mute]
                if len(markers) < 3:
                    continue
                
                markers_sorted = sorted(markers, key=lambda m: m.frame)
                first_frame = markers_sorted[0].frame
                last_frame = markers_sorted[-1].frame
                
                # Track stopped before clip end?
                can_extend_forward = last_frame < (clip_end - min_extension)
                # Track started after clip start?
                can_extend_backward = first_frame > (clip_start + min_extension)
                
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
        
        # Prioritize shorter tracks (they need extension most)
        lost_tracks.sort(key=lambda t: t['lifespan'])
        lost_tracks = lost_tracks[:20]  # Limit for performance
        
        print(f"AutoSolve: Extending {len(lost_tracks)} tracks that stopped early...")
        
        # Save current settings
        orig_correlation = self.current_settings.get('correlation', 0.7)
        orig_search = self.current_settings.get('search_size', 71)
        
        # Apply tolerant settings
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
                
                # Deselect all, select this track
                for t in self.tracking.tracks:
                    t.select = False
                track.select = True
                
                markers_before = len([m for m in track.markers if not m.mute])
                
                # Try extending forward
                if track_info['extend_forward']:
                    bpy.context.scene.frame_set(track_info['last_frame'])
                    try:
                        self._run_ops(bpy.ops.clip.track_markers, backwards=False, sequence=True)
                    except:
                        pass
                
                # Try extending backward
                if track_info['extend_backward']:
                    bpy.context.scene.frame_set(track_info['first_frame'])
                    try:
                        self._run_ops(bpy.ops.clip.track_markers, backwards=True, sequence=True)
                    except:
                        pass
                
                markers_after = len([m for m in track.markers if not m.mute])
                if markers_after > markers_before:
                    extended += 1
        
        except Exception as e:
            print(f"AutoSolve: Track extension error: {e}")
        
        finally:
            # Restore settings
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
        
        Uses complete "anchor" tracks to estimate motion during gaps and
        reconnect broken tracks that likely represent the same real-world point.
        
        Returns:
            Number of gaps successfully healed
        """
        if not self.enable_healing:
            return 0
        
        # Lazy init healer
        if self.healer is None:
            from .learning.track_healer import TrackHealer
            self.healer = TrackHealer()
        
        # Find anchor tracks (complete, high-quality reference tracks)
        anchors = self.healer.find_anchor_tracks(self.tracking)
        
        if len(anchors) < self.healer.MIN_ANCHOR_TRACKS:
            print(f"AutoSolve: Only {len(anchors)} anchors found - need {self.healer.MIN_ANCHOR_TRACKS}+ for healing")
            return 0
        
        # Record anchors for session data
        if self.recorder and self.recorder.current_session:
            self.recorder.current_session.anchor_tracks = [
                {
                    'name': a.name,
                    'start_frame': a.start_frame,
                    'end_frame': a.end_frame,
                    'quality': round(a.quality_score, 3)
                }
                for a in anchors[:10]  # Top 10
            ]
        
        # Find healing candidates
        candidates = self.healer.find_healing_candidates(self.tracking)
        
        if not candidates:
            # Message already printed by healer
            return 0
        
        # Update healing stats
        if self.recorder and self.recorder.current_session:
            self.recorder.current_session.healing_stats['candidates_found'] = len(candidates)
        
        # Heal candidates above threshold
        healed = 0
        attempted = 0
        gap_frames_total = 0
        match_scores_total = 0.0
        below_threshold = 0
        
        for candidate in candidates:
            if candidate.match_score >= self.healer.MIN_MATCH_SCORE:
                attempted += 1
                
                # Interpolate positions
                positions = self.healer.interpolate_with_anchors(candidate, anchors)
                
                # Attempt to heal
                success = self.healer.heal_track(candidate, self.tracking, anchors)
                
                # Collect training data
                training_data = self.healer.collect_training_data(
                    candidate, anchors, positions, success
                )
                
                # Record for session
                if self.recorder and self.recorder.current_session:
                    self.recorder.current_session.healing_attempts.append(
                        training_data.to_dict()
                    )
                
                if success:
                    healed += 1
                    gap_frames_total += candidate.gap_frames
                    match_scores_total += candidate.match_score
            else:
                below_threshold += 1
        
        # Update healing stats
        if self.recorder and self.recorder.current_session and attempted > 0:
            stats = self.recorder.current_session.healing_stats
            stats['heals_attempted'] = attempted
            stats['heals_successful'] = healed
            stats['avg_gap_frames'] = round(gap_frames_total / healed, 1) if healed > 0 else 0.0
            stats['avg_match_score'] = round(match_scores_total / healed, 3) if healed > 0 else 0.0
        
        # Improved logging
        if healed > 0:
            print(f"AutoSolve: Healed {healed}/{attempted} track gaps "
                  f"({100*healed/attempted:.0f}% success rate)")
        elif attempted > 0:
            print(f"AutoSolve: Healing attempted {attempted} gaps but none succeeded")
        elif below_threshold > 0:
            print(f"AutoSolve: {below_threshold} candidates found but none met score threshold "
                  f"(need >= {self.healer.MIN_MATCH_SCORE}, best: {candidates[0].match_score:.2f})")
        
        # Also try to merge overlapping track segments via averaging
        merged = self.healer.merge_overlapping_segments(self.tracking)
        if merged > 0:
            print(f"AutoSolve: Merged {merged} overlapping track segments via averaging")
            healed += merged  # Count merges as heals
        
        return healed

    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPORAL DEAD ZONES AND ITERATIVE REFINEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _get_frame_segment(self, frame: int, segment_size: int = 50) -> Tuple[int, int]:
        """Get the segment (start, end) for a given frame."""
        segment_start = (frame // segment_size) * segment_size
        segment_end = segment_start + segment_size
        return (segment_start, segment_end)
    
    def learn_from_failed_tracks(self):
        """
        Analyze tracks that failed reconstruction and update temporal dead zones.
        
        Called after a solve attempt to learn which regions were problematic
        at which times. This allows the algorithm to avoid those regions
        in those specific frame ranges on retry.
        """
        # Find tracks that failed (no bundle or high error)
        failed = []
        for track in self.tracking.tracks:
            if not track.has_bundle:
                failed.append(track)
            elif track.has_bundle and track.average_error > 5.0:
                failed.append(track)
        
        if not failed:
            print("AutoSolve: No failed tracks to learn from")
            return
        
        # Analyze each failed track's temporal-spatial pattern
        for track in failed:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            # Get average position (region)
            avg_x = sum(m.co.x for m in markers) / len(markers)
            avg_y = sum(m.co.y for m in markers) / len(markers)
            region = get_region(avg_x, avg_y)
            
            # Get frame range as segments
            markers_sorted = sorted(markers, key=lambda m: m.frame)
            start_frame = markers_sorted[0].frame
            end_frame = markers_sorted[-1].frame
            
            # Update temporal dead zones for each segment this track spans
            for frame in range(start_frame, end_frame + 1, 50):
                segment = self._get_frame_segment(frame)
                if segment not in self.temporal_dead_zones:
                    self.temporal_dead_zones[segment] = {}
                
                if region not in self.temporal_dead_zones[segment]:
                    self.temporal_dead_zones[segment][region] = 0
                
                self.temporal_dead_zones[segment][region] += 1
            
            # Store for analysis
            self.failed_tracks.append({
                'name': track.name,
                'region': region,
                'frames': (start_frame, end_frame),
                'has_bundle': track.has_bundle,
                'error': track.average_error if track.has_bundle else None,
            })
        
        print(f"AutoSolve: Learned from {len(failed)} failed tracks")
        self._print_temporal_dead_zones()
    
    def _print_temporal_dead_zones(self):
        """Print summary of temporal dead zones."""
        if not self.temporal_dead_zones:
            return
        
        hot_zones = []
        for segment, regions in self.temporal_dead_zones.items():
            for region, count in regions.items():
                if count >= 3:  # Threshold for "hot" zone
                    hot_zones.append(f"{region}@{segment[0]}-{segment[1]}: {count} failures")
        
        if hot_zones:
            print(f"AutoSolve: Temporal hot zones: {', '.join(hot_zones[:5])}")
    
    def is_in_temporal_dead_zone(self, x: float, y: float, frame: int) -> bool:
        """
        Check if a position at a specific frame is in a known temporal dead zone.
        
        Returns True if this region has had 3+ failures in this frame segment.
        """
        segment = self._get_frame_segment(frame)
        if segment not in self.temporal_dead_zones:
            return False
        
        region = get_region(x, y)
        failure_count = self.temporal_dead_zones[segment].get(region, 0)
        
        return failure_count >= 3
    
    def remove_worst_tracks(self, percentage: float = 0.15) -> int:
        """
        Remove the worst-performing tracks for iterative refinement.
        
        This is a gradual cleanup - not aggressive, just removes the worst
        performers to allow re-solving with better data.
        
        Args:
            percentage: Fraction of tracks to remove (0.15 = 15%)
            
        Returns:
            Number of tracks removed
        """
        # Get tracks with errors
        tracks_with_error = []
        for track in self.tracking.tracks:
            if track.has_bundle:
                tracks_with_error.append((track.name, track.average_error))
        
        if len(tracks_with_error) < self.SAFE_MIN_TRACKS:
            print("AutoSolve: Not enough tracks for removal")
            return 0
        
        # Sort by error (worst first)
        tracks_with_error.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate how many to remove
        num_to_remove = max(1, int(len(tracks_with_error) * percentage))
        # Don't remove too many
        num_to_remove = min(num_to_remove, len(tracks_with_error) - self.SAFE_MIN_TRACKS)
        
        if num_to_remove <= 0:
            return 0
        
        # Remove worst tracks
        to_remove = set(name for name, _ in tracks_with_error[:num_to_remove])
        
        for track in self.tracking.tracks:
            track.select = track.name in to_remove
        
        try:
            self._run_ops(bpy.ops.clip.delete_track)
            print(f"AutoSolve: Removed {num_to_remove} worst tracks (errors: "
                  f"{tracks_with_error[0][1]:.2f} - {tracks_with_error[num_to_remove-1][1]:.2f}px)")
        except:
            return 0
        
        return num_to_remove
    
    def should_continue_refinement(self) -> bool:
        """
        Determine if another refinement iteration is needed.
        
        Checks:
        - Current error vs target
        - Improvement from last iteration
        - Max refinement iterations
        """
        MAX_REFINEMENT_ITERATIONS = 5
        TARGET_ERROR = 2.0  # px
        
        if self.refinement_iteration >= MAX_REFINEMENT_ITERATIONS:
            print(f"AutoSolve: Max refinement iterations reached ({MAX_REFINEMENT_ITERATIONS})")
            return False
        
        current_error = self.get_solve_error()
        
        if current_error < TARGET_ERROR:
            print(f"AutoSolve: Target error achieved ({current_error:.2f}px < {TARGET_ERROR}px)")
            return False
        
        # Check if we're improving
        if current_error < self.best_solve_error:
            improvement = self.best_solve_error - current_error
            self.best_solve_error = current_error
            self.best_bundle_count = self.get_bundle_count()
            
            # If improvement is tiny, stop
            if improvement < 0.1 and self.refinement_iteration > 1:
                print(f"AutoSolve: Diminishing returns (improvement: {improvement:.2f}px)")
                return False
            
            return True
        else:
            # No improvement, stop refinement
            print(f"AutoSolve: No improvement from last iteration")
            return False
    
    def refine_solve(self) -> bool:
        """
        Perform one iteration of solve refinement.
        
        1. Learn from failed tracks
        2. Remove worst performers
        3. Re-solve camera
        
        Returns:
            True if solve succeeded, False otherwise
        """
        self.refinement_iteration += 1
        print(f"AutoSolve: Refinement iteration {self.refinement_iteration}")
        
        # Learn from failures
        self.learn_from_failed_tracks()
        
        # Remove worst tracks
        removed = self.remove_worst_tracks(percentage=0.15)
        if removed == 0:
            print("AutoSolve: Cannot remove more tracks")
            return False
        
        # Re-solve
        success = self.solve_camera(tripod_mode=False)
        
        if success:
            new_error = self.get_solve_error()
            new_bundles = self.get_bundle_count()
            print(f"AutoSolve: Refinement result - {new_bundles} bundles, {new_error:.2f}px error")
        
        return success
    def configure_settings(self):
        """Apply current settings to Blender's tracker."""
        # Start recording session for ML data collection
        if hasattr(self, 'recorder') and self.recorder:
            if not self.recorder.current_session:
                self.recorder.start_session(self.clip, self.current_settings)
        
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
        except:
            pass
    
    def identify_user_tracks(self) -> List[str]:
        """
        Identify tracks that appear to be user-placed.
        
        User-placed tracks typically have:
        - Few markers (1-5) - just placed, not fully tracked yet
        - OR are marked as locked
        
        Returns:
            List of track names that are user-placed
        """
        user_tracks = []
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            
            # Tracks with few markers (1-5) are likely user-placed
            if 1 <= len(markers) <= 5:
                user_tracks.append(track.name)
            # Locked tracks should be preserved
            elif hasattr(track, 'lock') and track.lock:
                user_tracks.append(track.name)
        
        if user_tracks:
            print(f"AutoSolve: Identified {len(user_tracks)} user-placed tracks (will protect)")
        
        return user_tracks
    
    def refine_struggling_tracks(self, user_tracks: set = None) -> int:
        """
        Attempt to re-track struggling tracks with more tolerant settings.
        
        This is called before deletion to give tracks a second chance.
        Struggling tracks are those with:
        - Short lifespan (< min_lifespan)
        - High error (> 5px if solve exists)
        
        Args:
            user_tracks: Optional pre-computed set of user track names
        
        Returns:
            Number of tracks that were successfully extended
        """
        # Use provided user_tracks or compute (avoids duplicate calls)
        if user_tracks is None:
            user_tracks = set(self.identify_user_tracks())
        
        # Safety check
        if not self.tracking or len(self.tracking.tracks) == 0:
            return 0
        
        has_solve = self.tracking.reconstruction.is_valid
        min_lifespan = max(3, self.min_lifespan // 2)
        
        # Identify struggling tracks
        struggling = []
        for track in self.tracking.tracks:
            try:
                markers = [m for m in track.markers if not m.mute]
                if len(markers) < 2:
                    continue
                
                markers_sorted = sorted(markers, key=lambda m: m.frame)
                lifespan = markers_sorted[-1].frame - markers_sorted[0].frame
                
                is_struggling = False
                
                # Short lifespan
                if lifespan < min_lifespan:
                    is_struggling = True
                
                # High error (only if solve exists)
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
                # Track may have been deleted
                continue
        
        if not struggling:
            return 0
        
        # Prioritize user tracks, limit to prevent slowdown
        struggling.sort(key=lambda t: (not t['is_user_track'], -t['lifespan']))
        struggling = struggling[:15]  # Limit for UX
        
        user_count = sum(1 for t in struggling if t['is_user_track'])
        print(f"AutoSolve: Refining {len(struggling)} struggling tracks ({user_count} user-placed)...")
        
        # Save current settings
        orig_correlation = self.current_settings.get('correlation', 0.7)
        orig_search = self.current_settings.get('search_size', 71)
        
        # Apply more tolerant settings for re-tracking
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
        
        # Batch refine: select all struggling tracks, then track as batch
        try:
            # Deselect all
            for t in self.tracking.tracks:
                t.select = False
            
            # Select struggling tracks
            struggling_names = {t['name'] for t in struggling}
            for t in self.tracking.tracks:
                if t.name in struggling_names:
                    t.select = True
            
            # Get markers before
            markers_before = {}
            for t in self.tracking.tracks:
                if t.name in struggling_names:
                    markers_before[t.name] = len([m for m in t.markers if not m.mute])
            
            # Find optimal frame range for batch tracking
            min_frame = min(t['first_frame'] for t in struggling)
            max_frame = max(t['last_frame'] for t in struggling)
            
            # Track forward from middle
            mid_frame = (min_frame + max_frame) // 2
            bpy.context.scene.frame_set(mid_frame)
            
            # Use sequence tracking (more efficient)
            self._run_ops(bpy.ops.clip.track_markers, backwards=False, sequence=True)
            
            # Track backward
            bpy.context.scene.frame_set(mid_frame)
            self._run_ops(bpy.ops.clip.track_markers, backwards=True, sequence=True)
            
            # Count extensions
            user_extended = []
            for t in self.tracking.tracks:
                if t.name in struggling_names:
                    markers_after = len([m for m in t.markers if not m.mute])
                    if markers_after > markers_before.get(t.name, 0):
                        extended += 1
                        # Track user track improvements for logging
                        info = next((s for s in struggling if s['name'] == t.name), None)
                        if info and info['is_user_track']:
                            user_extended.append(f"'{t.name}' ({markers_before[t.name]}→{markers_after})")
            
            # Log user track improvements (limited to prevent spam)
            if user_extended:
                if len(user_extended) <= 3:
                    print(f"AutoSolve: Extended user tracks: {', '.join(user_extended)}")
                else:
                    print(f"AutoSolve: Extended {len(user_extended)} user tracks")
            
        except Exception as e:
            print(f"AutoSolve: Track refinement error (continuing): {e}")
        finally:
            # Always restore settings and frame
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
        
        On retry or new autotrack, this preserves investment in good tracks
        while clearing tracks that didn't contribute to the solve.
        
        Args:
            min_lifespan: Minimum frames for track to be considered good.
                         Defaults to self.min_lifespan // 2 (lenient).
            max_error: Maximum reprojection error to keep (only applies if solve exists)
            refine: If True, attempt to refine struggling tracks before deleting
            
        Returns:
            Number of tracks preserved
        """
        if min_lifespan is None:
            min_lifespan = max(3, self.min_lifespan // 2)
        
        # STEP 1: Identify user tracks first (shared with refinement)
        user_tracks = set(self.identify_user_tracks())
        
        # STEP 2: Try to refine struggling tracks (give them a chance)
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
            
            # User-placed tracks are always protected
            if track.name in user_tracks:
                protected_tracks.append(track.name)
                continue
            
            # Criteria for keeping a track:
            # 1. Has sufficient lifespan
            # 2. If solve exists, has acceptable error OR hasn't been solved yet
            is_good = lifespan >= min_lifespan
            
            if has_solve and track.has_bundle:
                if track.average_error > max_error:
                    is_good = False
            
            if is_good and len(markers) >= 2:
                good_tracks.append(track.name)
            else:
                bad_tracks.append(track.name)
        
        # Delete bad tracks (but NEVER user tracks)
        if bad_tracks:
            for track in self.tracking.tracks:
                track.select = track.name in bad_tracks
            
            try:
                self._run_ops(bpy.ops.clip.delete_track)
                msg = f"AutoSolve: Removed {len(bad_tracks)} poor tracks, preserved {len(good_tracks)} good tracks"
                if protected_tracks:
                    msg += f", protected {len(protected_tracks)} user tracks"
                print(msg)
            except:
                pass
        else:
            msg = f"AutoSolve: Preserved all {len(good_tracks)} existing tracks"
            if protected_tracks:
                msg += f" + {len(protected_tracks)} user tracks"
            print(msg)
        
        return len(good_tracks) + len(protected_tracks)

    
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
        
        # Apply per-track settings
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
    
    def track_frame(self, backwards: bool = False):
        """Track one frame."""
        self.select_all_tracks()
        
        # Count markers BEFORE tracking
        frame = bpy.context.scene.frame_current
        clip_frame = self.scene_to_clip_frame(frame)  # Convert to clip-relative
        selected_count = sum(1 for t in self.tracking.tracks if t.select)
        
        # Get detailed marker status before tracking
        active_before = []
        for t in self.tracking.tracks:
            marker = t.markers.find_frame(clip_frame)
            if marker and not marker.mute:
                active_before.append(t.name)
        
        markers_at_frame_before = len(active_before)
        

        

        
        self._run_ops(bpy.ops.clip.track_markers, backwards=backwards, sequence=False)
        
        # Count markers AFTER tracking at next frame
        next_frame = frame - 1 if backwards else frame + 1
        next_clip_frame = self.scene_to_clip_frame(next_frame)
        
        # Get detailed marker status after tracking
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
        
        # Track marker loss for adaptive replenishment
        lost_count = markers_at_frame_before - markers_at_next
        

    
    def track_sequence(self, start_frame: int, end_frame: int, backwards: bool = False) -> int:
        """
        Track a sequence of frames with per-frame processing.
        
        Note: Uses frame-by-frame tracking to allow per-frame validation.
        For pure batch tracking, use bpy.ops.clip.track_markers with sequence=True.
        
        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number
            backwards: Track in reverse direction
            
        Returns:
            Number of frames tracked
        """
        if backwards:
            frame_range = range(start_frame, end_frame, -1)
        else:
            frame_range = range(start_frame, end_frame)
        
        frames_tracked = 0
        prev_active_count = 0
        self.select_all_tracks()
        
        for frame in frame_range:
            bpy.context.scene.frame_set(frame)
            self._run_ops(bpy.ops.clip.track_markers, backwards=backwards, sequence=False)
            frames_tracked += 1
            
            # Record frame sample every 10 frames for ML temporal analysis
            if hasattr(self, 'recorder') and self.recorder and frames_tracked % 10 == 0:
                prev_active_count = self.recorder.record_frame_sample(
                    frame, self.tracking, prev_active_count
                )
        
        return frames_tracked

    
    # Footage types that benefit from non-rigid motion filtering
    NON_RIGID_FOOTAGE_TYPES = {'DRONE', 'OUTDOOR', 'ACTION', 'HANDHELD'}
    
    
    def select_optimal_keyframes(self) -> bool:
        """
        Select optimal keyframes for camera solve based on parallax.
        
        The solver requires keyframes with:
        1. Maximum average track displacement (parallax)
        2. At least 8 common tracks on both frames
        3. Sufficient temporal separation (20% of clip duration)
        
        This fixes "POINT BEHIND CAMERA" errors caused by poor keyframe selection.
        
        Returns:
            True if keyframes were updated, False if kept defaults
        """
        camera = self.clip.tracking.camera
        clip_start = 1
        clip_end = self.clip.frame_duration
        min_separation = max(10, int(self.clip.frame_duration * 0.2))  # 20% of clip
        
        # Collect all frames with track counts
        frame_tracks = {}  # frame -> list of (track_name, x, y)
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            for marker in markers:
                if marker.frame not in frame_tracks:
                    frame_tracks[marker.frame] = []
                frame_tracks[marker.frame].append((track.name, marker.co.x, marker.co.y))
        
        if len(frame_tracks) < 2:
            print("AutoSolve: Not enough frames with tracks for keyframe selection")
            return False
        
        # Find frames with at least 8 tracks
        valid_frames = [f for f, tracks in frame_tracks.items() if len(tracks) >= 8]
        if len(valid_frames) < 2:
            print("AutoSolve: Not enough frames with 8+ tracks")
            return False
        
        valid_frames.sort()
        
        # Find best pair with maximum parallax
        best_parallax = 0
        best_pair = (valid_frames[0], valid_frames[-1])
        best_common_count = 0
        
        # Sample frames efficiently (every 10% of clip)
        sample_step = max(1, len(valid_frames) // 10)
        sample_frames = valid_frames[::sample_step]
        if valid_frames[-1] not in sample_frames:
            sample_frames.append(valid_frames[-1])
        
        for i, frame_a in enumerate(sample_frames):
            for frame_b in sample_frames[i+1:]:
                # Check separation
                if frame_b - frame_a < min_separation:
                    continue
                
                # Find common tracks
                tracks_a = {t[0]: (t[1], t[2]) for t in frame_tracks[frame_a]}
                tracks_b = {t[0]: (t[1], t[2]) for t in frame_tracks[frame_b]}
                common_tracks = set(tracks_a.keys()) & set(tracks_b.keys())
                
                if len(common_tracks) < 8:
                    continue
                
                # Calculate average displacement (parallax)
                total_disp = 0
                for track_name in common_tracks:
                    xa, ya = tracks_a[track_name]
                    xb, yb = tracks_b[track_name]
                    disp = ((xb - xa)**2 + (yb - ya)**2)**0.5
                    total_disp += disp
                
                avg_parallax = total_disp / len(common_tracks)
                
                # Prefer more parallax AND more common tracks
                score = avg_parallax * (1 + len(common_tracks) / 50)
                
                if score > best_parallax:
                    best_parallax = score
                    best_pair = (frame_a, frame_b)
                    best_common_count = len(common_tracks)
        
        # Apply best keyframes if they're different from defaults
        keyframe_a, keyframe_b = best_pair
        avg_parallax_percent = best_parallax * 100 if best_parallax < 1 else best_parallax
        
        # Blender 5.0+ removed keyframe_a/keyframe_b from MovieTrackingCamera
        # The solver now handles keyframe selection automatically
        if hasattr(camera, 'keyframe_a') and hasattr(camera, 'keyframe_b'):
            current_a = getattr(camera, 'keyframe_a', 1)
            current_b = getattr(camera, 'keyframe_b', clip_end)
            
            if keyframe_a != current_a or keyframe_b != current_b:
                camera.keyframe_a = keyframe_a
                camera.keyframe_b = keyframe_b
                print(f"AutoSolve: Selected keyframes {keyframe_a} and {keyframe_b} "
                      f"({best_common_count} common tracks, {avg_parallax_percent:.1f}% avg parallax)")
                return True
        else:
            # Blender 5.0+: keyframe selection is automatic, just log for info
            print(f"AutoSolve: Optimal keyframes analysis: frames {keyframe_a} and {keyframe_b} "
                  f"({best_common_count} common tracks, {avg_parallax_percent:.1f}% avg parallax)")
            return True  # Analysis completed successfully
        
        return False
    
    def solve_camera(self, tripod_mode: bool = False) -> bool:
        """
        Solve camera with robustness and quality checks.
        
        Attempts to solve with current settings. If that fails or yields poor quality
        (low track reconstruction), it re-tries with focal length refinement enabled.
        """
        if hasattr(self.settings, 'use_tripod_solver'):
            self.settings.use_tripod_solver = tripod_mode
        
        # Count tracks before solve
        track_count = len(self.tracking.tracks)
        tracks_with_markers = sum(1 for t in self.tracking.tracks if len(t.markers) > 0)
        
        # Check marker distribution across frame range
        frame_coverage = {}
        for t in self.tracking.tracks:
            for m in t.markers:
                if not m.mute:
                    frame_coverage[m.frame] = frame_coverage.get(m.frame, 0) + 1
        
        min_markers = min(frame_coverage.values()) if frame_coverage else 0
        max_markers = max(frame_coverage.values()) if frame_coverage else 0
        

        
        # Helper to toggle refinement options safely across Blender versions
        def set_refinement(enable: bool):
            # refine_focal_length, refine_principal_point, refine_k1, refine_k2 are standard properties
            # Some versions might bundle them differently, but these are top-level on settings
            props = ['refine_focal_length', 'refine_principal_point', 'refine_k1', 'refine_k2']
            count = 0
            for p in props:
                if hasattr(self.settings, p):
                    setattr(self.settings, p, enable)
                    count += 1
            return count > 0

        # Store original refinement state
        original_refinement = {}
        for p in ['refine_focal_length', 'refine_principal_point', 'refine_k1', 'refine_k2']:
            if hasattr(self.settings, p):
                original_refinement[p] = getattr(self.settings, p)

        try:
            # ATTEMPT 1: Initial solve (usually without refinement unless user enabled it)
            print("AutoSolve: Attempting initial camera solve...")
            self._run_ops(bpy.ops.clip.solve_camera)
            
            # Check results
            is_valid = self.tracking.reconstruction.is_valid
            bundle_count = self.get_bundle_count()
            bundle_ratio = bundle_count / max(track_count, 1)
            raw_error = self.tracking.reconstruction.average_error if is_valid else 999.0
            
            # Define failure (invalid or < 30% reconstruction is poor)
            quality_fail = is_valid and bundle_ratio < 0.3
            
            # ATTEMPT 2: Refine Intrinsics (if Attempt 1 failed or was poor)
            # This fixes "POINT BEHIND CAMERA" errors due to wrong focal length
            if not is_valid or quality_fail or raw_error > 3.0:
                print(f"AutoSolve: Initial solve poor (valid={is_valid}, ratio={bundle_ratio:.0%}, err={raw_error:.2f})")
                print("AutoSolve: Retrying with FOCAL LENGTH REFINEMENT enabled...")
                
                # Enable refinement
                set_refinement(True)
                
                # Solve again
                self._run_ops(bpy.ops.clip.solve_camera)
                
                # Check new results
                is_valid = self.tracking.reconstruction.is_valid
                bundle_count = self.get_bundle_count()
                bundle_ratio = bundle_count / max(track_count, 1)
                new_error = self.tracking.reconstruction.average_error if is_valid else 999.0
                
                print(f"AutoSolve: Refined solve result (valid={is_valid}, ratio={bundle_ratio:.0%}, err={new_error:.2f})")
                
                # If this worked better, keep it!
                # If it's still bad, we proceed to report failure
                if is_valid and bundle_ratio >= 0.3:
                    print("AutoSolve: Refinement FIXED the solve!")
                else:
                    print("AutoSolve: Refinement failed to improve solve sufficienty.")
                    
                # Restore original refinement state (so we don't accidentally leave it on forever)
                # But wait, if it fixed it, maybe we should keep the refined intrinsics?
                # The solver updates the camera intrinsics (FL, k1, k2). The 'refine' flags just tell it to DO so.
                # Once done, the FL is updated. We can turn the flags off.
                for p, val in original_refinement.items():
                    setattr(self.settings, p, val)

            # Final check before return
            if is_valid:
                # Re-calculate final stats
                bundle_count = self.get_bundle_count()
                bundle_ratio = bundle_count / max(track_count, 1)
                raw_error = self.tracking.reconstruction.average_error

                
                if bundle_ratio < 0.3:
                    print(f"AutoSolve WARNING: Low quality solve - only {bundle_count}/{track_count} tracks reconstructed")
                    print(f"AutoSolve WARNING: This usually indicates incorrect focal length or missing lens distortion")
                    self._solve_quality_failure = True
                    return False
                elif bundle_ratio < 0.5:
                    print(f"AutoSolve NOTICE: Moderate quality solve - {bundle_count}/{track_count} tracks reconstructed")
                
                self._solve_quality_failure = False
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
        
        # If most tracks failed to reconstruct, the error is misleading
        track_count = len(self.tracking.tracks)
        bundle_count = self.get_bundle_count()
        bundle_ratio = bundle_count / max(track_count, 1)
        
        raw_error = self.tracking.reconstruction.average_error
        
        # Penalize low bundle ratio - error can't be trusted
        if bundle_ratio < 0.3:
            # Very low ratio - report as failure
            return 999.0
        elif bundle_ratio < 0.5:
            # Low ratio - add penalty to error
            penalty = (0.5 - bundle_ratio) * 10  # Up to 5px penalty
            return raw_error + penalty
        else:
            return raw_error
    
    def get_bundle_count(self) -> int:
        return len([t for t in self.tracking.tracks if t.has_bundle])
    
    def analyze_and_learn(self) -> Dict:
        """Analyze tracks and learn from results."""
        # Use same min_lifespan as cleanup_tracks for consistency
        # In robust mode, use a lower threshold (half of normal)
        min_life = max(5, self.min_lifespan // 2) if self.robust_mode else self.min_lifespan
        self.last_analysis = self.analyzer.analyze_tracks(self.tracking, min_life)
        self.analyzer.iteration = self.iteration
        
        success_rate = self.last_analysis['success_rate']
        print(f"AutoSolve: Analysis - {self.last_analysis['successful_tracks']}/{self.last_analysis['total_tracks']} "
              f"successful ({success_rate*100:.0f}%)")
        
        if self.last_analysis['dead_zones']:
            print(f"AutoSolve: Dead zones: {', '.join(self.last_analysis['dead_zones'])}")
        
        # Update region confidence scores (probabilistic dead zones)
        if self.last_analysis.get('region_stats'):
            self.update_region_confidence(self.last_analysis['region_stats'])
        
        return self.last_analysis
    
    def should_retry(self, analysis: Dict) -> bool:
        """Determine if retry is needed."""
        if self.iteration >= self.MAX_ITERATIONS:
            return False
        
        return analysis['success_rate'] < 0.35
    
    def prepare_retry(self):
        """Prepare for retry with adjusted settings."""
        self.iteration += 1
        
        # Determine new tier based on previous success rate
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
        
        self.clear_tracks()
        self.configure_settings()
    
    def save_session_results(self, success: bool, solve_error: float):
        """Save session results for future learning."""
        bundle_count = self.get_bundle_count()
        
        # Always update model, even without full analysis (for early failures)
        region_stats = self.last_analysis.get('region_stats', {}) if self.last_analysis else {}
        
        self.predictor.update_from_session(
            footage_class=self.footage_class,
            success=success,
            settings=self.current_settings,
            error=solve_error,
            region_stats=region_stats,
            bundle_count=bundle_count  # For HER reward computation
        )
        print(f"AutoSolve: Updated model - success={success}, error={solve_error:.2f}")
        
        # Save clip-specific settings for per-clip learning
        if hasattr(self, 'clip_fingerprint') and self.clip_fingerprint:
            self.predictor.save_clip_specific_settings(
                self.clip_fingerprint, 
                self.current_settings, 
                success, 
                solve_error
            )
            
        # Record session data
        if hasattr(self, 'recorder') and self.recorder:
            try:
                # 1. Start Session (if not already aligned)
                if not self.recorder.current_session:
                    self.recorder.start_session(self.clip, self.current_settings)
                else:
                    # CRITICAL FIX: Update settings to match final state (in case of adaptation)
                    self.recorder.current_session.settings = self.current_settings.copy()
                    print(f"AutoSolve: Synced final settings to session record")
                
                # 2. Record Motion Probe Results (if available)
                if hasattr(self, 'cached_motion_probe') and self.cached_motion_probe:
                    self.recorder.record_motion_probe(self.cached_motion_probe)
                
                # 3. Record Clip fingerprint and motion class
                if hasattr(self, 'clip_fingerprint') and self.clip_fingerprint:
                    self.recorder.record_clip_fingerprint(self.clip_fingerprint)
                if hasattr(self, 'motion_class') and self.motion_class:
                    self.recorder.record_motion_class(self.motion_class)
                
                # 4. Record session linkage for multi-attempt analysis
                # Uses iteration count and previous_session_id if available
                previous_session_id = getattr(self, 'previous_session_id', "")
                iteration = getattr(self, 'iteration', 1)
                self.recorder.record_session_linkage(previous_session_id, iteration)
                
                # 5. Record contributor ID for multi-user data distinction
                from .utils import get_contributor_id
                self.recorder.record_contributor_id(get_contributor_id())
                
                # 4. Extract and Record Visual Features (feature density, motion, etc.)
                if hasattr(self, 'feature_extractor') and self.feature_extractor:
                    # Build tracking_data with pre-computed feature density
                    tracking_data = self.last_analysis.copy() if self.last_analysis else {}
                    
                    # Add detected feature density from smart detection (avoids duplicate detect_features call)
                    if hasattr(self, '_detected_feature_density') and self._detected_feature_density:
                        tracking_data['detected_feature_density'] = self._detected_feature_density
                    
                    # Extract features (feature density via detect_features if not pre-computed, motion, edge density)
                    try:
                        # CRITICAL: Force recompute to ensure we get full 9-frame timeline (not just detection frame shortcut)
                        self.feature_extractor.extract_all(
                            clip=self.clip,
                            tracking_data=tracking_data,
                            force_recompute=True
                        )
                        
                        # CRITICAL FIX: Explicitly compute post-tracking features
                        # valid tracking data is now available, so we must update edge density and histograms
                        if tracking_data:
                            self.feature_extractor.compute_from_tracking(tracking_data)
                            
                            # Build list of motion vectors from tracks for histogram
                            vectors = []
                            for track in self.tracking.tracks:
                                if len(track.markers) >= 2:
                                    markers = sorted(track.markers, key=lambda m: m.frame)
                                    dx = markers[-1].co.x - markers[0].co.x
                                    dy = markers[-1].co.y - markers[0].co.y
                                    vectors.append((dx, dy))
                            
                            self.feature_extractor.compute_flow_histograms(vectors)
                            print("AutoSolve: Re-computed visual features from final tracking data")
                            
                    except Exception as fe:
                        print(f"AutoSolve: Feature extraction failed: {fe}")
                    self.recorder.record_visual_features(self.feature_extractor.to_dict())
                    
                # 5. Record Adaptation History
                if hasattr(self, 'adaptation_history') and self.adaptation_history:
                    summary = self.get_adaptation_summary()
                    self.recorder.record_adaptation_history(summary)
                
                # 6. Record Tracks & Solve metrics
                self.recorder.record_tracks(self.tracking)
                self.recorder.finalize_session(
                    success=success,
                    solve_error=solve_error,
                    bundle_count=self.get_bundle_count()
                )
                
            except Exception as e:
                import traceback
                print(f"AutoSolve: Error recording session: {e}")
                traceback.print_exc()
    
    def _get_context_override(self):
        """Get context override for operators."""
        context = bpy.context
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'CLIP_EDITOR':
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            # CRITICAL: Sync clip editor's frame with scene frame
                            # The clip editor has its own frame cursor separate from scene
                            for space in area.spaces:
                                if space.type == 'CLIP_EDITOR':
                                    # Set clip space frame to match scene frame
                                    # This ensures detect_features places markers at the correct frame
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
        scene.render.fps = round(clip.fps)
        scene.render.fps_base = 1.0
    
    if clip.size[0] > 0:
        scene.render.resolution_x = clip.size[0]
        scene.render.resolution_y = clip.size[1]
        scene.render.resolution_percentage = 100
