# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
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
        'pattern_size': 31,      # Larger for 4K (was 25)
        'search_size': 151,      # Doubled for 4K resolution (was 111)
        'correlation': 0.65,
        'threshold': 0.25,
        'motion_model': 'Affine',
    },
    '4K_30fps': {
        'pattern_size': 29,      # Larger for 4K (was 23)
        'search_size': 141,      # ~2x HD's 71px to match relative coverage
        'correlation': 0.65,     # Slightly lower for larger search (was 0.68)
        'threshold': 0.25,       # Lower threshold for 4K detail (was 0.28)
        'motion_model': 'LocRot',
    },
    '4K_60fps': {
        'pattern_size': 25,      # Larger for 4K (was 21)
        'search_size': 121,      # ~2x HD's 51px (was 71)
        'correlation': 0.68,     # (was 0.70)
        'threshold': 0.28,       # (was 0.30)
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
        # Gimbal: smooth, predictable motion
        'search_size_mult': 0.9,
        'correlation': 0.72,
        'threshold': 0.32,
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
        # VFX plate: typically well-shot, good markers
        'correlation': 0.75,
        'threshold': 0.35,
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


# ═══════════════════════════════════════════════════════════════════════════
# TRACK ANALYZER (Learning Component)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrackStats:
    """Statistics for a single track."""
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
class RegionStats:
    """Statistics for a screen region."""
    name: str
    total_tracks: int = 0
    successful_tracks: int = 0
    avg_lifespan: float = 0.0
    success_rate: float = 0.0


class TrackAnalyzer:
    """Analyzes tracking patterns and learns from them."""
    
    REGIONS = [
        'top-left', 'top-center', 'top-right',
        'mid-left', 'center', 'mid-right',
        'bottom-left', 'bottom-center', 'bottom-right'
    ]
    
    def __init__(self):
        self.track_stats: List[TrackStats] = []
        self.region_stats: Dict[str, RegionStats] = {
            r: RegionStats(name=r) for r in self.REGIONS
        }
        self.dead_zones: Set[str] = set()
        self.sweet_spots: Set[str] = set()
        self.iteration: int = 0
    
    def get_region(self, x: float, y: float) -> str:
        """Get region name from normalized coordinates (0-1)."""
        col = 0 if x < 0.33 else (1 if x < 0.66 else 2)
        row = 2 if y < 0.33 else (1 if y < 0.66 else 0)
        
        region_map = [
            ['top-left', 'top-center', 'top-right'],
            ['mid-left', 'center', 'mid-right'],
            ['bottom-left', 'bottom-center', 'bottom-right']
        ]
        return region_map[row][col]
    
    def analyze_tracks(self, tracking, min_lifespan: int = 5) -> Dict:
        """Analyze all tracks and calculate statistics."""
        self.track_stats.clear()
        
        for region in self.region_stats.values():
            region.total_tracks = 0
            region.successful_tracks = 0
            region.avg_lifespan = 0.0
        
        region_lifespans = {r: [] for r in self.REGIONS}
        
        for track in tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            markers.sort(key=lambda x: x.frame)
            
            lifespan = markers[-1].frame - markers[0].frame
            
            avg_x = sum(m.co.x for m in markers) / len(markers)
            avg_y = sum(m.co.y for m in markers) / len(markers)
            region = self.get_region(avg_x, avg_y)
            
            displacement = (Vector(markers[-1].co) - Vector(markers[0].co)).length
            avg_velocity = displacement / max(lifespan, 1)
            
            jitter = self._calculate_jitter(markers)
            success = lifespan >= min_lifespan
            
            stats = TrackStats(
                name=track.name,
                lifespan=lifespan,
                start_frame=markers[0].frame,
                end_frame=markers[-1].frame,
                region=region,
                avg_velocity=avg_velocity,
                jitter_score=jitter,
                success=success,
                contributed_to_solve=track.has_bundle,
                reprojection_error=track.average_error if track.has_bundle else 0.0,
            )
            self.track_stats.append(stats)
            
            self.region_stats[region].total_tracks += 1
            if success:
                self.region_stats[region].successful_tracks += 1
            region_lifespans[region].append(lifespan)
        
        for region, stats in self.region_stats.items():
            if stats.total_tracks > 0:
                stats.success_rate = stats.successful_tracks / stats.total_tracks
                if region_lifespans[region]:
                    stats.avg_lifespan = sum(region_lifespans[region]) / len(region_lifespans[region])
        
        self._identify_zones()
        return self._get_summary()
    
    def _calculate_jitter(self, markers) -> float:
        """Calculate jitter score (variance in velocity)."""
        if len(markers) < 3:
            return 0.0
        
        velocities = []
        for i in range(1, len(markers)):
            v = (Vector(markers[i].co) - Vector(markers[i-1].co)).length
            velocities.append(v)
        
        if not velocities:
            return 0.0
        
        avg_v = sum(velocities) / len(velocities)
        if avg_v == 0:
            return 0.0
        
        variance = sum((v - avg_v) ** 2 for v in velocities) / len(velocities)
        return (variance ** 0.5) / avg_v
    
    def _identify_zones(self):
        """Identify dead zones and sweet spots."""
        self.dead_zones.clear()
        self.sweet_spots.clear()
        
        for region, stats in self.region_stats.items():
            if stats.total_tracks < 3:
                continue
            
            if stats.success_rate < 0.3:
                self.dead_zones.add(region)
            elif stats.success_rate > 0.7:
                self.sweet_spots.add(region)
    
    def _get_summary(self) -> Dict:
        """Get analysis summary."""
        total = len(self.track_stats)
        successful = sum(1 for t in self.track_stats if t.success)
        
        return {
            'total_tracks': total,
            'successful_tracks': successful,
            'success_rate': successful / max(total, 1),
            'dead_zones': list(self.dead_zones),
            'sweet_spots': list(self.sweet_spots),
            'region_stats': {r: asdict(s) for r, s in self.region_stats.items()},
            'avg_lifespan': sum(t.lifespan for t in self.track_stats) / max(total, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════
# COVERAGE ANALYZER (Industry-Standard Distribution Tracking)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CoverageData:
    """Coverage data for a region-time segment."""
    region: str
    segment: Tuple[int, int]
    track_count: int = 0
    successful_tracks: int = 0
    avg_lifespan: float = 0.0
    needs_more: bool = False


class CoverageAnalyzer:
    """
    Tracks spatial and temporal distribution of markers.
    
    Industry standard: Good camera solves require:
    - Tracks distributed across the frame (not clustered)
    - Tracks spanning the full timeline (not just parts)
    - Minimum parallax requirements met
    """
    
    REGIONS = [
        'top-left', 'top-center', 'top-right',
        'mid-left', 'center', 'mid-right',
        'bottom-left', 'bottom-center', 'bottom-right'
    ]
    
    # Industry thresholds
    MIN_TRACKS_PER_REGION = 3
    MIN_REGIONS_WITH_TRACKS = 6  # At least 6/9 regions
    MAX_TRACKS_PER_REGION_PERCENT = 0.30  # No region > 30%
    MIN_TEMPORAL_COVERAGE = 0.80  # Tracks should span 80% of frames
    
    def __init__(self, clip_frame_start: int, clip_frame_end: int, segment_size: int = 50):
        self.frame_start = clip_frame_start
        self.frame_end = clip_frame_end
        self.segment_size = segment_size
        
        # Coverage grid: region -> segment -> CoverageData
        self.coverage: Dict[str, Dict[Tuple[int, int], CoverageData]] = {}
        self._init_coverage_grid()
    
    def _init_coverage_grid(self):
        """Initialize empty coverage grid."""
        for region in self.REGIONS:
            self.coverage[region] = {}
            for frame in range(self.frame_start, self.frame_end + 1, self.segment_size):
                segment = self._get_segment(frame)
                self.coverage[region][segment] = CoverageData(
                    region=region,
                    segment=segment
                )
    
    def _get_segment(self, frame: int) -> Tuple[int, int]:
        """Get segment tuple for a frame."""
        seg_start = ((frame - self.frame_start) // self.segment_size) * self.segment_size + self.frame_start
        seg_end = min(seg_start + self.segment_size, self.frame_end)
        return (seg_start, seg_end)
    
    def get_region(self, x: float, y: float) -> str:
        """Get region name from normalized coordinates (0-1)."""
        col = 0 if x < 0.33 else (1 if x < 0.66 else 2)
        row = 2 if y < 0.33 else (1 if y < 0.66 else 0)
        
        region_map = [
            ['top-left', 'top-center', 'top-right'],
            ['mid-left', 'center', 'mid-right'],
            ['bottom-left', 'bottom-center', 'bottom-right']
        ]
        return region_map[row][col]
    
    def get_region_bounds(self, region: str) -> Tuple[float, float, float, float]:
        """Get (x_min, y_min, x_max, y_max) for a region in normalized coords."""
        region_bounds = {
            'top-left': (0.0, 0.66, 0.33, 1.0),
            'top-center': (0.33, 0.66, 0.66, 1.0),
            'top-right': (0.66, 0.66, 1.0, 1.0),
            'mid-left': (0.0, 0.33, 0.33, 0.66),
            'center': (0.33, 0.33, 0.66, 0.66),
            'mid-right': (0.66, 0.33, 1.0, 0.66),
            'bottom-left': (0.0, 0.0, 0.33, 0.33),
            'bottom-center': (0.33, 0.0, 0.66, 0.33),
            'bottom-right': (0.66, 0.0, 1.0, 0.33),
        }
        return region_bounds.get(region, (0.0, 0.0, 1.0, 1.0))
    
    def analyze_tracking(self, tracking, min_lifespan: int = 5):
        """
        Analyze current tracking data for coverage.
        
        Updates the coverage grid with actual track distribution.
        """
        # Reset counts
        self._init_coverage_grid()
        
        region_lifespans: Dict[str, Dict[Tuple, List[int]]] = {
            r: {s: [] for s in self.coverage[r].keys()} for r in self.REGIONS
        }
        
        for track in tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            markers_sorted = sorted(markers, key=lambda m: m.frame)
            lifespan = markers_sorted[-1].frame - markers_sorted[0].frame
            
            # Get region from average position
            avg_x = sum(m.co.x for m in markers) / len(markers)
            avg_y = sum(m.co.y for m in markers) / len(markers)
            region = self.get_region(avg_x, avg_y)
            
            # Update coverage for each segment this track spans
            for marker in markers:
                segment = self._get_segment(marker.frame)
                if segment in self.coverage[region]:
                    self.coverage[region][segment].track_count += 1
                    if lifespan >= min_lifespan:
                        self.coverage[region][segment].successful_tracks += 1
                    if segment in region_lifespans[region]:
                        region_lifespans[region][segment].append(lifespan)
        
        # Calculate averages and mark gaps
        for region in self.REGIONS:
            for segment, data in self.coverage[region].items():
                lifespans = region_lifespans[region].get(segment, [])
                if lifespans:
                    data.avg_lifespan = sum(lifespans) / len(lifespans)
                data.needs_more = data.successful_tracks < self.MIN_TRACKS_PER_REGION
    
    def get_coverage_summary(self) -> Dict:
        """
        Get comprehensive coverage analysis.
        
        Returns industry-standard metrics for solve quality prediction.
        """
        total_tracks = 0
        regions_with_tracks = 0
        region_counts = {}
        temporal_coverage = set()
        gaps = []
        
        for region in self.REGIONS:
            region_total = 0
            for segment, data in self.coverage[region].items():
                region_total += data.successful_tracks
                if data.successful_tracks > 0:
                    temporal_coverage.add(segment)
                if data.needs_more:
                    gaps.append((region, segment))
            
            region_counts[region] = region_total
            total_tracks += region_total
            if region_total >= self.MIN_TRACKS_PER_REGION:
                regions_with_tracks += 1
        
        # Calculate balance score
        max_region = max(region_counts.values()) if region_counts else 0
        balance_score = 1.0 - (max_region / max(total_tracks, 1))
        
        # Calculate temporal coverage
        total_segments = len(list(self.coverage[self.REGIONS[0]].keys()))
        temporal_percent = len(temporal_coverage) / max(total_segments, 1)
        
        return {
            'total_tracks': total_tracks,
            'regions_with_tracks': regions_with_tracks,
            'region_counts': region_counts,
            'balance_score': balance_score,  # 1.0 = perfectly balanced
            'temporal_coverage': temporal_percent,
            'gaps': gaps,  # List of (region, segment) needing more tracks
            'is_balanced': (
                regions_with_tracks >= self.MIN_REGIONS_WITH_TRACKS and
                balance_score >= (1.0 - self.MAX_TRACKS_PER_REGION_PERCENT) and
                temporal_percent >= self.MIN_TEMPORAL_COVERAGE
            ),
        }
    
    def get_weak_zones(self) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Get regions and time segments that need more tracks.
        
        Returns list of (region, segment) tuples sorted by priority.
        """
        weak = []
        for region in self.REGIONS:
            for segment, data in self.coverage[region].items():
                if data.needs_more:
                    weak.append((region, segment, data.successful_tracks))
        
        # Sort by track count (lowest first = highest priority)
        weak.sort(key=lambda x: x[2])
        return [(r, s) for r, s, _ in weak]
    
    def get_clustered_regions(self) -> List[str]:
        """
        Get regions with too many tracks (clustering).
        
        These should be deprioritized for new marker placement.
        """
        summary = self.get_coverage_summary()
        total = summary['total_tracks']
        if total == 0:
            return []
        
        clustered = []
        for region, count in summary['region_counts'].items():
            if count / total > self.MAX_TRACKS_PER_REGION_PERCENT:
                clustered.append(region)
        
        return clustered



# ═══════════════════════════════════════════════════════════════════════════
# SMART TRACKER (Main Class)
# ═══════════════════════════════════════════════════════════════════════════

class SmartTracker:
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
                 footage_type: str = 'AUTO'):
        self.clip = clip
        self.tracking = clip.tracking
        self.settings = clip.tracking.settings
        self.robust_mode = robust_mode
        self.footage_type = footage_type
        
        # Learning components
        self.analyzer = TrackAnalyzer()
        
        # Unified SettingsPredictor for all learning
        from .learning.settings_predictor import SettingsPredictor
        self.predictor = SettingsPredictor()
        
        # Session recorder for training data
        from .learning.session_recorder import SessionRecorder
        self.recorder = SessionRecorder()
        
        # Current session state
        self.resolution_class = self._classify_footage()
        self.footage_class = f"{self.resolution_class}_{footage_type}"
        self.current_settings: Dict = {}
        self.iteration = 0
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
        
        # Motion probe cache (persisted for session recording)
        self.cached_motion_probe: Optional[Dict] = None
        
        # Region confidence scores (probabilistic dead zones)
        self.region_confidence: Dict[str, float] = {r: 0.5 for r in TrackAnalyzer.REGIONS}
        
        # Try to load cached probe from disk
        self._try_load_cached_probe()
        
        # Load initial settings
        self._load_initial_settings()
    
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
        clip_hash = hashlib.md5(self.clip.filepath.encode()).hexdigest()[:8]
        return cache_dir / f"probe_{clip_hash}.json"
    
    def _try_load_cached_probe(self):
        """Try to load cached probe from disk if valid."""
        cache_path = self._get_probe_cache_path()
        if not cache_path.exists():
            return
        
        try:
            data = json.loads(cache_path.read_text())
            
            # Validate cache is for same clip version
            clip_mtime = os.path.getmtime(bpy.path.abspath(self.clip.filepath))
            if abs(data.get('clip_mtime', 0) - clip_mtime) < 1:  # Within 1 second
                self.cached_motion_probe = data.get('probe_results')
                print(f"AutoSolve: Loaded cached probe for {self.clip.name}")
        except Exception as e:
            print(f"AutoSolve: Could not load cached probe: {e}")
    
    def _save_probe_to_cache(self, probe_results: Dict):
        """Save probe results to disk for reuse."""
        try:
            cache_path = self._get_probe_cache_path()
            clip_mtime = os.path.getmtime(bpy.path.abspath(self.clip.filepath))
            
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
        fps = self.clip.fps if self.clip.fps > 0 else 24
        
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
        Load initial settings using hybrid approach with footage type.
        
        Priority order:
        1. Resolution-based defaults (HD_30fps, 4K_24fps, etc.)
        2. Footage type adjustments (DRONE, HANDHELD, etc.) - hints only
        3. Learned settings from local model (overrides if 2+ sessions)
        4. Learned dead zones from actual tracking data
        5. Robust mode adjustments (always on top if enabled)
        
        Dead zones from footage type are just HINTS - actual dead zones
        are learned from tracking data and override predictions.
        """
        # Step 1: Start with resolution-based defaults
        if self.resolution_class in PRETRAINED_DEFAULTS:
            self.current_settings = PRETRAINED_DEFAULTS[self.resolution_class].copy()
            print(f"AutoSolve: Base settings for {self.resolution_class}")
        else:
            self.current_settings = TIERED_SETTINGS['balanced'].copy()
            print(f"AutoSolve: Using balanced defaults")
        
        # Step 2: Apply footage type adjustments (HINTS, not hard rules)
        predicted_dead_zones = set()
        if self.footage_type != 'AUTO' and self.footage_type in FOOTAGE_TYPE_ADJUSTMENTS:
            adjustments = FOOTAGE_TYPE_ADJUSTMENTS[self.footage_type]
            
            # Apply multipliers
            if 'pattern_size_mult' in adjustments:
                self.current_settings['pattern_size'] = int(
                    self.current_settings.get('pattern_size', 15) * adjustments['pattern_size_mult']
                )
            if 'search_size_mult' in adjustments:
                self.current_settings['search_size'] = int(
                    self.current_settings.get('search_size', 71) * adjustments['search_size_mult']
                )
            
            # Apply direct overrides
            for key in ['correlation', 'threshold', 'motion_model']:
                if key in adjustments:
                    self.current_settings[key] = adjustments[key]
            
            # Store predicted dead zones (just hints, will be verified)
            if 'dead_zones' in adjustments:
                predicted_dead_zones = set(adjustments['dead_zones'])
            
            print(f"AutoSolve: Applied {self.footage_type} adjustments")
        
        # Step 3: Check for learned settings (overrides if available)
        local_settings = self.predictor.get_settings_for_class(self.footage_class)
        if local_settings:
            self.current_settings = local_settings.copy()
            print(f"AutoSolve: Using LEARNED settings for {self.footage_class}")
        
        # Step 4: Get LEARNED dead zones (overrides predictions if we have data)
        learned_dead_zones = self.predictor.get_dead_zones_for_class(self.footage_class)
        if learned_dead_zones:
            # Use learned data, ignore predictions
            self.known_dead_zones = learned_dead_zones
            print(f"AutoSolve: Using LEARNED dead zones: {', '.join(learned_dead_zones)}")
        else:
            # No learned data yet, use predictions as hints (but don't block)
            # Note: These are just informational, we don't block detection
            self.known_dead_zones = set()  # Don't use predictions to block
            if predicted_dead_zones:
                print(f"AutoSolve: Predicted dead zones (will verify): {', '.join(predicted_dead_zones)}")
        
        # Step 4: Apply robust mode (always on top)
        if self.robust_mode:
            self.current_settings['pattern_size'] = int(self.current_settings.get('pattern_size', 15) * 1.4)
            self.current_settings['search_size'] = int(self.current_settings.get('search_size', 71) * 1.4)
            self.current_settings['correlation'] = max(0.45, self.current_settings.get('correlation', 0.7) - 0.15)
            self.current_settings['threshold'] = max(0.08, self.current_settings.get('threshold', 0.3) - 0.12)
            self.current_settings['motion_model'] = 'Affine'

    def load_learning(self, clip_name: str):
        """
        Load learning data for the specific clip.
        
        Currently a placeholder or wrapper for ensure settings are consistent
        with what might be loaded from a more specific per-clip file if implemented later.
        For now, we rely on the class-based local learning loaded in __init__.
        """
        print(f"AutoSolve: Loading learning data for clip '{clip_name}'...")
        # In the future, per-clip specific learning could be loaded here.
        # For now, we are good with the class-based learning loaded in init.
        pass

    def analyze_footage(self):
        """
        Analyze footage characteristics.
        
        This method inspects the clip to determine its properties and
        adjust settings accordingly. Currently a placeholder that logs
        the classification already done in __init__.
        """
        print(f"AutoSolve: Analyzing footage - {self.footage_class}")
        # Footage analysis is already done in __init__ via _classify_footage
        # and _load_initial_settings. This method exists for explicit calls.
    
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
            if region not in TrackAnalyzer.REGIONS:
                continue
            
            total = data.get('total_tracks', 0)
            successful = data.get('successful_tracks', 0)
            
            if total >= MIN_SAMPLES:
                rate = successful / total
                if rate < SKIP_THRESHOLD:
                    skip.add(region)
                    
        return skip
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDATION METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def validate_pre_tracking(self) -> Tuple[bool, List[str]]:
        """
        Validate conditions before tracking begins.
        
        Checks:
        - Clip has sufficient frames
        - Markers have valid initial positions
        - No NaN/Inf values in existing markers
        
        Returns:
            Tuple of (is_valid, list of issues found)
        """
        issues = []
        
        # Check clip duration
        if self.clip.frame_duration < 10:
            issues.append(f"Clip too short: {self.clip.frame_duration} frames (need 10+)")
        
        # Check existing markers for NaN/Inf
        nan_tracks = []
        for track in self.tracking.tracks:
            for marker in track.markers:
                if marker.mute:
                    continue
                # Check for invalid coordinates
                import math
                if math.isnan(marker.co.x) or math.isnan(marker.co.y):
                    nan_tracks.append(track.name)
                    break
                if math.isinf(marker.co.x) or math.isinf(marker.co.y):
                    nan_tracks.append(track.name)
                    break
                # Check bounds (normalized 0-1)
                if not (0 <= marker.co.x <= 1) or not (0 <= marker.co.y <= 1):
                    nan_tracks.append(track.name)
                    break
        
        if nan_tracks:
            issues.append(f"Invalid marker data in {len(nan_tracks)} tracks: {', '.join(nan_tracks[:5])}")
        
        is_valid = len(issues) == 0
        if is_valid:
            print("AutoSolve: Pre-tracking validation passed")
        else:
            print(f"AutoSolve: Pre-tracking validation failed: {'; '.join(issues)}")
        
        return is_valid, issues
    
    def validate_track_quality(self, frame: int) -> Dict:
        """
        Validate track quality at a specific frame during tracking.
        
        Checks for:
        - Out-of-bounds markers
        - Velocity spikes (sudden jumps)
        - Tracks that should be muted
        
        Returns:
            Dict with validation results and tracks to mute
        """
        import math
        
        result = {
            'frame': frame,
            'active_tracks': 0,
            'out_of_bounds': [],
            'velocity_spikes': [],
            'tracks_to_mute': [],
        }
        
        for track in self.tracking.tracks:
            marker = track.markers.find_frame(frame)
            if not marker or marker.mute:
                continue
            
            result['active_tracks'] += 1
            
            # Check bounds
            if not (0 <= marker.co.x <= 1) or not (0 <= marker.co.y <= 1):
                result['out_of_bounds'].append(track.name)
                result['tracks_to_mute'].append(track.name)
                continue
            
            # Check for NaN
            if math.isnan(marker.co.x) or math.isnan(marker.co.y):
                result['tracks_to_mute'].append(track.name)
                continue
            
            # Check velocity spike (compare to previous frame)
            prev_marker = track.markers.find_frame(frame - 1)
            if prev_marker and not prev_marker.mute:
                dx = abs(marker.co.x - prev_marker.co.x)
                dy = abs(marker.co.y - prev_marker.co.y)
                displacement = (dx**2 + dy**2) ** 0.5
                
                # If displacement > 10% of frame in one step, likely a spike
                if displacement > 0.1:
                    result['velocity_spikes'].append(track.name)
                    result['tracks_to_mute'].append(track.name)
        
        # Mute problematic tracks
        for track in self.tracking.tracks:
            if track.name in result['tracks_to_mute']:
                marker = track.markers.find_frame(frame)
                if marker:
                    marker.mute = True
        
        if result['tracks_to_mute']:
            print(f"AutoSolve: Frame {frame} - Muted {len(result['tracks_to_mute'])} bad tracks")
        
        return result
    
    def validate_pre_solve(self) -> Tuple[bool, List[str]]:
        """
        Validate track data before camera solve.
        
        Ensures:
        - Sufficient valid tracks
        - No NaN/Inf in marker data
        - Adequate track coverage
        - Minimum lifespan requirements
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        import math
        issues = []
        
        # Count valid tracks
        valid_tracks = 0
        total_markers = 0
        nan_count = 0
        short_tracks = 0
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            # Check for NaN/Inf
            has_nan = False
            for m in markers:
                if math.isnan(m.co.x) or math.isnan(m.co.y):
                    has_nan = True
                    nan_count += 1
                    break
                if math.isinf(m.co.x) or math.isinf(m.co.y):
                    has_nan = True
                    nan_count += 1
                    break
            
            if has_nan:
                continue
            
            # Check lifespan
            markers_sorted = sorted(markers, key=lambda m: m.frame)
            lifespan = markers_sorted[-1].frame - markers_sorted[0].frame
            if lifespan < 5:
                short_tracks += 1
                continue
            
            valid_tracks += 1
            total_markers += len(markers)
        
        # Validation checks
        if valid_tracks < self.ABSOLUTE_MIN_TRACKS:
            issues.append(f"Too few valid tracks: {valid_tracks} (need {self.ABSOLUTE_MIN_TRACKS}+)")
        
        if nan_count > 0:
            issues.append(f"Found {nan_count} tracks with NaN/Inf values")
        
        if short_tracks > valid_tracks:
            issues.append(f"Too many short-lived tracks: {short_tracks}")
        
        # Check frame coverage
        frame_coverage = {}
        for track in self.tracking.tracks:
            for marker in track.markers:
                if not marker.mute:
                    frame_coverage[marker.frame] = frame_coverage.get(marker.frame, 0) + 1
        
        if frame_coverage:
            avg_tracks_per_frame = sum(frame_coverage.values()) / len(frame_coverage)
            if avg_tracks_per_frame < 8:
                issues.append(f"Low average track coverage: {avg_tracks_per_frame:.1f} tracks/frame")
        
        is_valid = len(issues) == 0
        if is_valid:
            print(f"AutoSolve: Pre-solve validation passed ({valid_tracks} valid tracks)")
        else:
            print(f"AutoSolve: Pre-solve validation failed: {'; '.join(issues)}")
        
        return is_valid, issues
    
    def compute_pre_solve_confidence(self) -> Dict:
        """
        Estimate solve quality before running the solver.
        
        Uses track features to predict likelihood of successful solve.
        Records result for ML training correlation with actual solve error.
        
        Returns:
            Dict with:
                - confidence: 0-1 overall estimate (1 = likely good solve)
                - parallax_score: 0-1 depth variation detected
                - track_distribution_score: 0-1 how well tracks cover the frame
                - warnings: List of potential issues
        """
        confidence = 1.0
        warnings = []
        
        # 1. Check track count
        track_count = len(self.tracking.tracks)
        if track_count < 15:
            confidence *= 0.6
            warnings.append(f"Low track count ({track_count})")
        elif track_count < 25:
            confidence *= 0.85
        
        # 2. Compute parallax score from motion vectors
        motion_vectors = []
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            markers.sort(key=lambda x: x.frame)
            dx = markers[-1].co.x - markers[0].co.x
            dy = markers[-1].co.y - markers[0].co.y
            motion_vectors.append((dx, dy))
        
        parallax_score = 0.0
        if len(motion_vectors) >= 3:
            # Compute variance in motion directions
            avg_dx = sum(v[0] for v in motion_vectors) / len(motion_vectors)
            avg_dy = sum(v[1] for v in motion_vectors) / len(motion_vectors)
            
            variance = sum(
                ((v[0] - avg_dx)**2 + (v[1] - avg_dy)**2) 
                for v in motion_vectors
            ) / len(motion_vectors)
            
            avg_magnitude = (avg_dx**2 + avg_dy**2)**0.5
            if avg_magnitude > 0.0001:
                parallax_score = min(1.0, (variance**0.5) / avg_magnitude)
        
        if parallax_score < 0.05:
            confidence *= 0.5
            warnings.append("Very low parallax - consider tripod mode")
        elif parallax_score < 0.15:
            confidence *= 0.75
            warnings.append("Low parallax - perspective solve may struggle")
        
        # 3. Check track distribution (coverage of 9 regions)
        regions_covered = set()
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if markers:
                avg_x = sum(m.co.x for m in markers) / len(markers)
                avg_y = sum(m.co.y for m in markers) / len(markers)
                col = 0 if avg_x < 0.33 else (1 if avg_x < 0.66 else 2)
                row = 2 if avg_y < 0.33 else (1 if avg_y < 0.66 else 0)
                regions_covered.add((row, col))
        
        distribution_score = len(regions_covered) / 9.0
        
        if distribution_score < 0.5:
            confidence *= 0.7
            warnings.append(f"Poor track distribution ({len(regions_covered)}/9 regions)")
        elif distribution_score < 0.7:
            confidence *= 0.9
        
        # 4. Check track lifespan consistency
        lifespans = []
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) >= 2:
                markers.sort(key=lambda x: x.frame)
                lifespans.append(markers[-1].frame - markers[0].frame)
        
        if lifespans:
            avg_lifespan = sum(lifespans) / len(lifespans)
            if avg_lifespan < 30:
                confidence *= 0.8
                warnings.append(f"Short average track lifespan ({avg_lifespan:.0f} frames)")
        
        result = {
            'confidence': round(confidence, 3),
            'parallax_score': round(parallax_score, 3),
            'track_distribution_score': round(distribution_score, 3),
            'track_count': track_count,
            'warnings': warnings,
        }
        
        # Log and record for ML training
        print(f"AutoSolve: Pre-solve confidence: {confidence:.0%} "
              f"(parallax: {parallax_score:.2f}, distribution: {distribution_score:.2f})")
        if warnings:
            print(f"  Warnings: {', '.join(warnings)}")
        
        # Record to session for correlation with actual solve error
        if hasattr(self, 'recorder') and self.recorder and self.recorder.current_session:
            self.recorder.record_pre_solve_confidence(result)
        
        return result
    
    def sanitize_tracks_before_solve(self) -> int:
        """
        Actively clean up problematic tracks before camera solve.
        
        This method REMOVES tracks that would cause Ceres solver errors:
        - Tracks with NaN/Inf values
        - Tracks with out-of-bounds markers
        - Tracks with too few markers
        - Tracks with impossible velocity spikes
        
        Returns:
            Number of tracks removed
        """
        import math
        
        tracks_to_remove = []
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            
            # Remove tracks with too few markers
            if len(markers) < 3:
                tracks_to_remove.append(track.name)
                continue
            
            # Check for bad data
            is_bad = False
            prev_pos = None
            
            for marker in markers:
                x, y = marker.co.x, marker.co.y
                
                # Check NaN/Inf
                if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                    is_bad = True
                    break
                
                # Check out of bounds (with margin)
                if x < -0.1 or x > 1.1 or y < -0.1 or y > 1.1:
                    is_bad = True
                    break
                
                # Check velocity spike
                if prev_pos is not None:
                    dx = abs(x - prev_pos[0])
                    dy = abs(y - prev_pos[1])
                    if dx > 0.2 or dy > 0.2:  # 20% of frame in one step
                        is_bad = True
                        break
                
                prev_pos = (x, y)
            
            if is_bad:
                tracks_to_remove.append(track.name)
        
        # Remove bad tracks
        if tracks_to_remove:
            for track in self.tracking.tracks:
                track.select = track.name in tracks_to_remove
            
            try:
                self._run_ops(bpy.ops.clip.delete_track)
                print(f"AutoSolve: Sanitized {len(tracks_to_remove)} bad tracks before solve")
            except:
                pass
        
        return len(tracks_to_remove)
    
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
                        for r in self.analyzer.REGIONS}
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            # Get region from average position
            avg_x = sum(m.co.x for m in markers) / len(markers)
            avg_y = sum(m.co.y for m in markers) / len(markers)
            region = self.analyzer.get_region(avg_x, avg_y)
            
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
            region = self.coverage_analyzer.get_region(
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
                'region': self.coverage_analyzer.get_region(markers[0].co.x, markers[0].co.y),
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
    
    def detect_strategic_features(self, markers_per_region: int = 3) -> int:
        """
        DEPRECATED: Use detect_features_smart() instead.
        
        This method is maintained for backwards compatibility but will be removed
        in a future version. detect_features_smart() provides the same functionality
        with better performance using single-pass detection.
        
        Args:
            markers_per_region: Target markers per screen region (default 3)
            
        Returns:
            Total number of features detected
        """
        import warnings
        warnings.warn(
            "detect_strategic_features is deprecated, use detect_features_smart instead",
            DeprecationWarning,
            stacklevel=2
        )
        total_detected = 0
        regions = CoverageAnalyzer.REGIONS.copy()
        
        # Get user priority regions
        priority = self.get_user_priority_regions()
        priority_regions = set(priority['high']) | set(priority['existing'])
        
        # Preserve existing good tracks
        self.preserve_existing_tracks()
        
        # Shuffle regions to avoid bias (randomize placement order)
        import random
        random.shuffle(regions)
        
        # Exclude known temporal dead zones for current frame
        current_frame = bpy.context.scene.frame_current
        active_dead_zones = set()
        segment = self._get_frame_segment(current_frame)
        if segment in self.temporal_dead_zones:
            for region, count in self.temporal_dead_zones[segment].items():
                if count >= 3:
                    active_dead_zones.add(region)
        
        for region in regions:
            if region in active_dead_zones:
                print(f"AutoSolve: Skipping {region} (temporal dead zone)")
                continue
            
            # Double markers in user-priority regions
            target_count = markers_per_region
            if region in priority_regions:
                target_count = markers_per_region * 2
                print(f"AutoSolve: Priority region {region}: targeting {target_count} markers")
            
            detected = self.detect_in_region(region, target_count)
            total_detected += detected
        
        # If strategic detection failed, fallback to standard detection
        if total_detected < 8:
            print(f"AutoSolve: Strategic detection got only {total_detected}, using standard detection")
            threshold = self.current_settings.get('threshold', 0.3)
            standard_count = self.detect_features(threshold)
            print(f"AutoSolve: Standard detection: {standard_count} markers")
            return standard_count
        
        print(f"AutoSolve: Strategic detection - {total_detected} markers "
              f"({len(priority_regions)} priority regions)")
        return total_detected
    
    def _is_non_rigid_region(self, region: str) -> bool:
        """
        Check if a region is likely to contain non-rigid objects (waves, water, foliage).
        
        This is a PRE-DETECTION check to avoid placing markers on problematic regions.
        ONLY skips regions with actual probe evidence, not just based on footage type.
        
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
        velocities = probe.get('velocities', {})
        
        # Check if this region had very low success in the probe
        if region in region_success:
            region_data = region_success[region]
            # Handle both dict format (new) and float format (legacy)
            if isinstance(region_data, dict):
                success_rate = region_data.get('success_rate', 1.0)
            else:
                success_rate = region_data
            
            if success_rate < 0.2:
                # Probe showed this region is problematic
                print(f"AutoSolve: Skipping {region} - probe showed {success_rate:.0%} success")
                return True
        
        # Check if this region had extremely high velocity (likely non-rigid)
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
        
        bounds = self.coverage_analyzer.get_region_bounds(region)
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
                placement='FRAME'
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
    
    def detect_all_regions(self, markers_per_region: int = 3, 
                          skip_regions: Optional[Set[str]] = None) -> Dict[str, int]:
        """
        OPTIMIZED: Detect features with single global pass, distribute to all regions.
        
        This is ~60% faster than calling detect_in_region 9 times because it:
        1. Runs detect_features ONCE globally
        2. Categorizes ALL features by region in one pass
        3. Sorts by QUALITY and keeps top N per region
        
        Args:
            markers_per_region: Target markers per region (default 3)
            skip_regions: Optional set of regions to skip (dead zones, etc.)
            
        Returns:
            Dict mapping region name to count of features kept
        """
        skip_regions = skip_regions or set()
        
        # Add non-rigid regions to skip list
        for region in CoverageAnalyzer.REGIONS:
            if self._is_non_rigid_region(region):
                skip_regions.add(region)
        
        # Also skip known dead zones
        skip_regions.update(self.known_dead_zones)
        
        if skip_regions:
            print(f"AutoSolve: Skipping regions: {', '.join(skip_regions)}")
        
        initial_count = len(self.tracking.tracks)
        
        # DEBUG: Log frame context before detection
        scene_frame = bpy.context.scene.frame_current
        print(f"AutoSolve DEBUG: detect_all_regions at scene_frame={scene_frame}, clip.frame_start={self.clip.frame_start}")
        print(f"AutoSolve DEBUG: Expected clip_frame = {self.scene_to_clip_frame(scene_frame)}")
        
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
                placement='FRAME'
            )
        except Exception as e:
            print(f"AutoSolve: detect_features failed: {e}")
            return {r: 0 for r in CoverageAnalyzer.REGIONS}
        
        new_tracks = list(self.tracking.tracks)[initial_count:]
        
        # DEBUG: Show where markers were actually placed
        if new_tracks:
            sample_marker = new_tracks[0].markers[0] if len(new_tracks[0].markers) > 0 else None
            if sample_marker:
                print(f"AutoSolve DEBUG: First marker placed at clip frame {sample_marker.frame} (expected {self.scene_to_clip_frame(scene_frame)})")
        
        if not new_tracks:
            print("AutoSolve: No features detected")
            return {r: 0 for r in CoverageAnalyzer.REGIONS}
        
        print(f"AutoSolve: Global detection found {len(new_tracks)} candidates")
        
        # Categorize all tracks by region WITH QUALITY SCORE
        tracks_by_region: Dict[str, List[Tuple[Any, float]]] = {r: [] for r in CoverageAnalyzer.REGIONS}
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
            
            region = self.coverage_analyzer.get_region(marker.co.x, marker.co.y)
            tracks_by_region[region].append((track, quality))
        
        # Process each region: SORT BY QUALITY, keep top N
        result: Dict[str, int] = {}
        to_delete = list(no_marker_tracks)  # Always delete tracks without markers
        
        for region in CoverageAnalyzer.REGIONS:
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
        UNIFIED SMART DETECTION
        
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
        print(f"AutoSolve: ═══════════════════════════════════════════════")
        print(f"AutoSolve: SMART FEATURE DETECTION")
        print(f"AutoSolve: ═══════════════════════════════════════════════")
        
        # Step 1: Get motion classification (use cache or run probe)
        if use_cached_probe and hasattr(self, 'cached_motion_probe') and self.cached_motion_probe:
            probe_results = self.cached_motion_probe
            print(f"AutoSolve: Using cached probe (motion: {probe_results.get('motion_class')})")
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
    
    def detect_exploratory_features(self, markers_per_region: int = 3) -> int:
        """
        DEPRECATED: Use detect_features_smart() instead.
        
        This method is maintained for backwards compatibility but will be removed
        in a future version. detect_features_smart() provides the same functionality
        with better performance.
        
        Args:
            markers_per_region: Target markers per region (default 3)
            
        Returns:
            Total number of quality features detected
        """
        import warnings
        warnings.warn(
            "detect_exploratory_features is deprecated, use detect_features_smart instead",
            DeprecationWarning,
            stacklevel=2
        )
        print(f"AutoSolve: ═══════════════════════════════════════════════")
        print(f"AutoSolve: PHASED EXPLORATORY DETECTION")
        print(f"AutoSolve: ═══════════════════════════════════════════════")
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: MOTION PROBE
        # Place few markers with very aggressive settings to measure motion
        # ═══════════════════════════════════════════════════════════════
        print(f"AutoSolve: Phase 1 - MOTION PROBE")
        
        probe_results = self._run_motion_probe()
        
        if not probe_results['success']:
            print(f"AutoSolve: Probe failed, using default aggressive settings")
            return self._detect_quality_markers(
                motion_class='HIGH',
                texture_class='LOW',
                markers_per_region=2
            )
        
        motion_class = probe_results['motion_class']  # LOW, MEDIUM, HIGH
        texture_class = probe_results['texture_class']  # LOW, MEDIUM, HIGH
        best_regions = probe_results['best_regions']
        
        print(f"AutoSolve: Phase 1 complete - Motion: {motion_class}, Texture: {texture_class}")
        print(f"AutoSolve: Best regions: {', '.join(best_regions)}")
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: QUALITY DETECTION
        # Use probe results to place fewer, better markers
        # ═══════════════════════════════════════════════════════════════
        print(f"AutoSolve: Phase 2 - QUALITY DETECTION")
        
        # Determine markers per region based on motion (fewer for high motion)
        if motion_class == 'HIGH':
            target_per_region = 1  # Few but robust
        elif motion_class == 'MEDIUM':
            target_per_region = 2
        else:
            target_per_region = 2  # Low motion can handle more
        
        total = self._detect_quality_markers(
            motion_class=motion_class,
            texture_class=texture_class,
            markers_per_region=target_per_region,
            priority_regions=best_regions
        )
        
        print(f"AutoSolve: Phase 2 complete - {total} quality markers placed")
        
        # Minimum viable check
        if total < 8:
            print(f"AutoSolve: Only {total} markers, adding reinforcements...")
            extra = self._add_reinforcement_markers(total, motion_class)
            total += extra
        
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
            return {
                'success': True,
                'motion_class': quick_class,
                'texture_class': 'MEDIUM',
                'best_regions': ['center', 'mid-left', 'mid-right', 'bottom-center'],
                'velocities': {},
                'region_success': {},
                'probe_type': 'quick_estimate'
            }
        
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
        regions = CoverageAnalyzer.REGIONS.copy()
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
                    jitters.append(variance ** 0.5)
            
            # Track region success
            avg_x = sum(m.co.x for m in markers_sorted) / len(markers_sorted)
            avg_y = sum(m.co.y for m in markers_sorted) / len(markers_sorted)
            region = self.analyzer.get_region(avg_x, avg_y)
            
            lifespan = len(markers_sorted)
            if region not in region_success:
                region_success[region] = {'total': 0, 'success': 0}
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
                if self.coverage_analyzer.get_region(avg_x, avg_y) == region:
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
            region = self.analyzer.get_region(avg_x, avg_y)
            
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
        
        region = self.analyzer.get_region(x, y)
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
        
        # ALWAYS log first few frames to confirm tracking runs
        if frame <= 190 or frame % 50 == 0:
            print(f"AutoSolve DEBUG: track_frame({backwards=}) frame={frame} selected={selected_count} markers_here={markers_at_frame_before}")
            # Inspect first track to see where its markers are
            if selected_count > 0 and markers_at_frame_before == 0:
                for track in self.tracking.tracks:
                    if track.select:
                        print(f"  - Track {track.name} has {len(track.markers)} markers at frames: {[m.frame for m in track.markers]}")
                        break
        
        # DEBUG: Check clip_user frame vs scene frame before tracking
        clip_user_frame = None
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'CLIP_EDITOR':
                    for space in area.spaces:
                        if space.type == 'CLIP_EDITOR':
                            clip_user_frame = space.clip_user.frame_current
                            break
        
        if frame <= 125:
            print(f"AutoSolve DEBUG: Before track_markers: scene_frame={frame}, clip_user_frame={clip_user_frame}, clip.frame_start={self.clip.frame_start}")
        
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
        
        # Log if we lost markers this frame
        lost_count = markers_at_frame_before - markers_at_next
        if lost_count > 0 and frame <= 130:
            lost = set(active_before) - set(active_after)
            print(f"AutoSolve DEBUG: Frame {frame}→{next_frame} LOST {lost_count} markers: {list(lost)[:5]}")
            # Show settings of first lost track
            for t in self.tracking.tracks:
                if t.name in lost:
                    print(f"  - {t.name}: correlation_min={t.correlation_min:.2f}")
                    break
        
        # Log if tracking completely failed
        if markers_at_next == 0 and markers_at_frame_before > 0:
            print(f"AutoSolve DEBUG: TRACKING FAILED at frame {frame} → {next_frame}")
            print(f"  - Had {markers_at_frame_before} markers, now {markers_at_next}")
    
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
    
    def cleanup_tracks(self, min_frames: int = 5, spike_multiplier: float = 8.0,
                       jitter_threshold: float = 0.6, coherence_threshold: float = 0.4) -> int:
        """
        Unified track cleanup - all filters in one pass.
        
        Combines:
        1. Short track filtering
        2. Velocity spike removal
        3. Non-rigid motion filtering (ONLY for outdoor/drone footage)
        
        Args:
            min_frames: Minimum frames for a track to survive
            spike_multiplier: Velocity spike threshold multiplier
            jitter_threshold: Jitter score for non-rigid detection
            coherence_threshold: Motion coherence for non-rigid detection
            
        Returns:
            Total number of tracks removed
        """
        initial = len(self.tracking.tracks)
        
        # 1. Short tracks (using Blender's built-in)
        self.filter_short_tracks(min_frames=min_frames)
        after_short = len(self.tracking.tracks)
        
        # 2. Velocity spikes
        self.filter_spikes(limit_multiplier=spike_multiplier)
        after_spikes = len(self.tracking.tracks)
        
        # 3. Non-rigid motion - ONLY for footage types that may have waves/water/foliage
        non_rigid_removed = 0
        if self.footage_type in self.NON_RIGID_FOOTAGE_TYPES:
            self.filter_non_rigid_motion(jitter_threshold, coherence_threshold)
            non_rigid_removed = after_spikes - len(self.tracking.tracks)
        
        after_non_rigid = len(self.tracking.tracks)
        
        # 4. Coverage-aware deduplication (removes close tracks only in saturated regions)
        self.deduplicate_tracks(min_distance_px=30)
        duplicates_removed = after_non_rigid - len(self.tracking.tracks)
        
        final = len(self.tracking.tracks)
        
        removed = initial - final
        non_rigid_msg = f", non-rigid:{non_rigid_removed}" if non_rigid_removed > 0 else ""
        dedup_msg = f", duplicates:{duplicates_removed}" if duplicates_removed > 0 else ""
        print(f"AutoSolve: Cleanup removed {removed} tracks "
              f"(short:{initial-after_short}, spikes:{after_short-after_spikes}{non_rigid_msg}{dedup_msg}) → {final} remaining")
        
        return removed
    
    def analyze_and_learn(self) -> Dict:
        """Analyze tracks and learn from results."""
        min_life = 5 if self.robust_mode else 8
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
        
        if self.last_analysis:
            # Extract region_stats from last_analysis properly
            region_stats = self.last_analysis.get('region_stats', {})
            
            self.predictor.update_from_session(
                footage_class=self.footage_class,
                success=success,
                settings=self.current_settings,
                error=solve_error,
                region_stats=region_stats,
                bundle_count=bundle_count  # NEW: For HER reward computation
            )
            
        # ═══════════════════════════════════════════════════════════════
        # RECORD SESSION DATA
        # ═══════════════════════════════════════════════════════════════
        if hasattr(self, 'recorder') and self.recorder:
            try:
                # 1. Start Session (if not already aligned)
                if not self.recorder.current_session:
                    self.recorder.start_session(self.clip, self.current_settings)
                
                # 2. Record Motion Probe Results (if available)
                if hasattr(self, 'cached_motion_probe') and self.cached_motion_probe:
                    self.recorder.record_motion_probe(self.cached_motion_probe)
                    
                # 3. Record Adaptation History
                if hasattr(self, 'adaptation_history') and self.adaptation_history:
                    # Summarize adaptation history
                    summary = self.get_adaptation_summary()
                    self.recorder.record_adaptation_history(summary)
                
                # 4. Record Tracks & Solve metrics
                self.recorder.record_tracks(self.tracking)
                self.recorder.finalize_session(
                    success=success,
                    solve_error=solve_error,
                    bundle_count=self.get_bundle_count()
                )
                
                # 5. Record Failure Diagnostics (stored in analyzer)
                # We need to extract this from the last failure analysis ideally
                # For now, we'll rely on the recorder picking up what it can or wait for v2
                
            except Exception as e:
                print(f"AutoSolve: Error recording session: {e}")
    
    def filter_short_tracks(self, min_frames: int = 5):
        """Filter short tracks with safeguards."""
        current = len(self.tracking.tracks)
        
        survivors = sum(1 for t in self.tracking.tracks
                       if len([m for m in t.markers if not m.mute]) >= min_frames)
        
        if survivors < self.SAFE_MIN_TRACKS:
            print(f"AutoSolve: Skipping filter (would leave {survivors})")
            return
        
        self.select_all_tracks()
        try:
            self._run_ops(bpy.ops.clip.clean_tracks, frames=min_frames, error=999, action='DELETE_TRACK')
        except TypeError:
            self._run_ops(bpy.ops.clip.clean_tracks, frames=min_frames, error=999, action='DELETE')
        
        print(f"AutoSolve: After filter: {len(self.tracking.tracks)} tracks")
    
    def filter_spikes(self, limit_multiplier: float = 8.0):
        """Filter velocity outliers."""
        current = len(self.tracking.tracks)
        
        if current < self.SAFE_MIN_TRACKS:
            return
        
        track_speeds = {}
        total_speed = 0.0
        count = 0
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            markers.sort(key=lambda x: x.frame)
            displacement = (Vector(markers[-1].co) - Vector(markers[0].co)).length
            duration = abs(markers[-1].frame - markers[0].frame)
            
            if duration > 0:
                speed = displacement / duration
                track_speeds[track.name] = speed
                total_speed += speed
                count += 1
        
        if count == 0:
            return
        
        avg = max(total_speed / count, 0.001)
        limit = avg * limit_multiplier
        
        to_delete = [n for n, s in track_speeds.items() if s > limit]
        max_del = min(len(to_delete), current - self.ABSOLUTE_MIN_TRACKS)
        
        if max_del <= 0:
            return
        
        sorted_tracks = sorted(track_speeds.items(), key=lambda x: x[1], reverse=True)
        to_delete = set(n for n, _ in sorted_tracks[:max_del] if track_speeds[n] > limit)
        
        for track in self.tracking.tracks:
            track.select = track.name in to_delete
        
        if to_delete:
            print(f"AutoSolve: Removing {len(to_delete)} outliers")
            try:
                self._run_ops(bpy.ops.clip.delete_track)
            except:
                pass
            self.select_all_tracks()
    
    def deduplicate_tracks(self, min_distance_px: int = 30):
        """
        Coverage-aware track deduplication.
        
        Only removes close tracks when overall coverage is good.
        Keeps duplicates in sparse regions to maintain solve quality.
        
        Args:
            min_distance_px: Minimum pixel distance between tracks (in normalized coords)
        """
        current = len(self.tracking.tracks)
        if current < self.SAFE_MIN_TRACKS:
            return
        
        # Get current coverage assessment
        self.coverage_analyzer.analyze_tracking(self.tracking)
        summary = self.coverage_analyzer.get_coverage_summary()
        regions_with_tracks = summary.get('regions_with_tracks', 0)
        region_counts = summary.get('region_counts', {})
        
        # Only deduplicate if we have good coverage (6+ of 9 regions)
        if regions_with_tracks < 6:
            print(f"AutoSolve: Skipping dedup (only {regions_with_tracks}/9 regions have tracks)")
            return
        
        # Find saturated regions (> 35% of total tracks - very conservative)
        total = max(sum(region_counts.values()), 1)
        saturated_regions = {r for r, c in region_counts.items() if c / total > 0.35}
        
        if not saturated_regions:
            return
        
        print(f"AutoSolve: Deduplicating in saturated regions: {saturated_regions}")
        
        # Convert min_distance to normalized coords - very conservative (15px in HD)
        width = self.clip.size[0]
        min_dist_norm = 15 / width  # Only remove truly overlapping tracks
        
        # Collect track positions (first marker position)
        track_positions = {}
        current_frame = bpy.context.scene.frame_current
        clip_frame = self.scene_to_clip_frame(current_frame)
        
        for track in self.tracking.tracks:
            marker = track.markers.find_frame(clip_frame)
            if not marker:
                markers = [m for m in track.markers if not m.mute]
                if markers:
                    marker = markers[0]
            
            if marker:
                track_positions[track.name] = (marker.co.x, marker.co.y)
        
        # Find duplicates to remove (within min_dist in saturated regions)
        to_delete = set()
        track_list = list(track_positions.items())
        
        for i, (name1, pos1) in enumerate(track_list):
            if name1 in to_delete:
                continue
            
            # Check region of this track
            region1 = self._get_region_for_pos(pos1[0], pos1[1])
            if region1 not in saturated_regions:
                continue
            
            for j, (name2, pos2) in enumerate(track_list[i+1:], i+1):
                if name2 in to_delete:
                    continue
                
                # Check if same region
                region2 = self._get_region_for_pos(pos2[0], pos2[1])
                if region2 != region1:
                    continue
                
                # Check distance
                dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) ** 0.5
                if dist < min_dist_norm:
                    # Keep track with longer lifespan
                    track1 = self.tracking.tracks.get(name1)
                    track2 = self.tracking.tracks.get(name2)
                    
                    if track1 and track2:
                        len1 = len([m for m in track1.markers if not m.mute])
                        len2 = len([m for m in track2.markers if not m.mute])
                        
                        # Delete shorter track
                        to_delete.add(name1 if len1 < len2 else name2)
        
        # Safety check - very conservative: max 10% of tracks or keep SAFE_MIN
        max_delete = min(len(to_delete), current // 10, current - self.SAFE_MIN_TRACKS)
        if max_delete <= 0:
            return
        to_delete = set(list(to_delete)[:max_delete])
        
        if not to_delete:
            return
        
        # Select and delete
        for track in self.tracking.tracks:
            track.select = track.name in to_delete
        
        print(f"AutoSolve: Removing {len(to_delete)} duplicate tracks")
        try:
            self._run_ops(bpy.ops.clip.delete_track)
        except:
            pass
        self.select_all_tracks()
    
    def _get_region_for_pos(self, x: float, y: float) -> str:
        """Get region name for normalized coordinates."""
        col = 0 if x < 0.33 else (1 if x < 0.66 else 2)
        row = 2 if y < 0.33 else (1 if y < 0.66 else 0)
        regions = [
            ['top-left', 'top-center', 'top-right'],
            ['mid-left', 'center', 'mid-right'],
            ['bottom-left', 'bottom-center', 'bottom-right']
        ]
        return regions[row][col]
    
    def filter_non_rigid_motion(self, jitter_threshold: float = 0.6, coherence_threshold: float = 0.4):
        """
        Filter tracks on non-rigid moving objects like waves, water, foliage.
        
        This filter detects tracks that:
        1. Have high jitter (erratic motion, not smooth camera movement)
        2. Move differently from the global camera motion pattern
        
        Non-rigid objects like waves create tracks that oscillate randomly,
        while camera motion produces coherent, parallel track movements.
        
        Args:
            jitter_threshold: Normalized jitter score above which tracks are suspect (default 0.6)
            coherence_threshold: Motion coherence below which tracks are filtered (default 0.4)
        """
        current = len(self.tracking.tracks)
        if current < self.SAFE_MIN_TRACKS:
            return
        
        # Step 1: Compute per-track motion vectors and jitter scores
        track_data = {}
        motion_vectors = []  # For computing median camera motion
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 5:  # Need enough markers to analyze motion
                continue
            
            markers.sort(key=lambda x: x.frame)
            
            # Compute frame-to-frame velocities
            velocities = []
            for i in range(1, len(markers)):
                dx = markers[i].co.x - markers[i-1].co.x
                dy = markers[i].co.y - markers[i-1].co.y
                velocities.append((dx, dy))
            
            if not velocities:
                continue
            
            # Average motion vector (overall direction)
            avg_dx = sum(v[0] for v in velocities) / len(velocities)
            avg_dy = sum(v[1] for v in velocities) / len(velocities)
            motion_vec = (avg_dx, avg_dy)
            motion_vectors.append(motion_vec)
            
            # Jitter: how much does velocity change frame to frame?
            # High jitter = erratic motion (waves, leaves moving)
            jitter = 0.0
            if len(velocities) >= 2:
                vel_changes = []
                for i in range(1, len(velocities)):
                    change_x = abs(velocities[i][0] - velocities[i-1][0])
                    change_y = abs(velocities[i][1] - velocities[i-1][1])
                    vel_changes.append((change_x**2 + change_y**2)**0.5)
                
                if vel_changes:
                    avg_mag = (avg_dx**2 + avg_dy**2)**0.5
                    if avg_mag > 0.0001:
                        jitter = (sum(vel_changes) / len(vel_changes)) / avg_mag
                    else:
                        jitter = sum(vel_changes) / len(vel_changes) * 100  # Scale for near-static
            
            track_data[track.name] = {
                'motion_vec': motion_vec,
                'jitter': jitter,
                'coherence': 0.0,  # Will be computed below
            }
        
        if len(motion_vectors) < 5:
            print("AutoSolve: Not enough tracks for non-rigid filter")
            return
        
        # Step 2: Compute median camera motion direction
        # Sort by angle to find the dominant motion direction
        import math
        angles = [math.atan2(v[1], v[0]) for v in motion_vectors]
        angles.sort()
        median_angle = angles[len(angles) // 2]
        
        # Compute median magnitude
        magnitudes = [(v[0]**2 + v[1]**2)**0.5 for v in motion_vectors]
        magnitudes.sort()
        median_mag = magnitudes[len(magnitudes) // 2]
        
        # Median motion vector (represents camera motion)
        camera_motion = (math.cos(median_angle) * median_mag, math.sin(median_angle) * median_mag)
        camera_mag = (camera_motion[0]**2 + camera_motion[1]**2)**0.5
        
        # Step 3: Compute coherence for each track (how well it matches camera motion)
        for name, data in track_data.items():
            mv = data['motion_vec']
            mv_mag = (mv[0]**2 + mv[1]**2)**0.5
            
            if camera_mag < 0.0001 or mv_mag < 0.0001:
                # Very low motion - assume coherent
                data['coherence'] = 1.0
            else:
                # Dot product normalized = cosine similarity
                dot = mv[0] * camera_motion[0] + mv[1] * camera_motion[1]
                coherence = dot / (mv_mag * camera_mag)
                # Clamp to [0, 1] (negative = opposite direction, still incoherent)
                data['coherence'] = max(0.0, coherence)
        
        # Step 4: Identify non-rigid tracks (high jitter OR low coherence)
        non_rigid = []
        for name, data in track_data.items():
            is_jittery = data['jitter'] > jitter_threshold
            is_incoherent = data['coherence'] < coherence_threshold
            
            if is_jittery or is_incoherent:
                non_rigid.append((name, data['jitter'], data['coherence']))
        
        # Safety: don't delete too many
        max_can_delete = current - self.ABSOLUTE_MIN_TRACKS
        if len(non_rigid) > max_can_delete:
            # Prioritize removing the most jittery/incoherent
            non_rigid.sort(key=lambda x: (x[1] - x[2]), reverse=True)
            non_rigid = non_rigid[:max_can_delete]
        
        to_delete = set(n for n, _, _ in non_rigid)
        
        for track in self.tracking.tracks:
            track.select = track.name in to_delete
        
        if to_delete:
            print(f"AutoSolve: Removing {len(to_delete)} non-rigid tracks (waves/water/foliage)")
            for name, jitter, coherence in non_rigid[:5]:  # Log first 5
                print(f"  → {name}: jitter={jitter:.2f}, coherence={coherence:.2f}")
            try:
                self._run_ops(bpy.ops.clip.delete_track)
            except:
                pass
            self.select_all_tracks()

    
    def filter_high_error(self, max_error: float = 3.0):
        """Filter high error tracks."""
        if not self.tracking.reconstruction.is_valid:
            return
        
        current = len(self.tracking.tracks)
        if current < self.SAFE_MIN_TRACKS:
            return
        
        to_delete = [t.name for t in self.tracking.tracks
                    if t.has_bundle and t.average_error > max_error]
        
        max_can = current - self.ABSOLUTE_MIN_TRACKS
        if len(to_delete) > max_can:
            errors = [(t.name, t.average_error) for t in self.tracking.tracks if t.has_bundle]
            errors.sort(key=lambda x: x[1], reverse=True)
            to_delete = [n for n, _ in errors[:max_can]]
        
        for track in self.tracking.tracks:
            track.select = track.name in to_delete
        
        if to_delete:
            print(f"AutoSolve: Removing {len(to_delete)} high-error tracks")
            try:
                self._run_ops(bpy.ops.clip.delete_track)
            except:
                pass
    
    def solve_camera(self, tripod_mode: bool = False) -> bool:
        """Solve camera."""
        if hasattr(self.settings, 'use_tripod_solver'):
            self.settings.use_tripod_solver = tripod_mode
        
        # DEBUG: Log track status before solve
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
        
        print(f"AutoSolve DEBUG: solve_camera - {track_count} tracks, {tracks_with_markers} with markers")
        print(f"AutoSolve DEBUG: Frame coverage - min {min_markers} markers, max {max_markers} markers")
        
        try:
            self._run_ops(bpy.ops.clip.solve_camera)
            is_valid = self.tracking.reconstruction.is_valid
            if is_valid:
                print(f"AutoSolve DEBUG: Solve SUCCESS - error {self.tracking.reconstruction.average_error:.2f}")
            else:
                print(f"AutoSolve DEBUG: Solve FAILED - reconstruction not valid")
            return is_valid
        except RuntimeError as e:
            print(f"AutoSolve DEBUG: Solve EXCEPTION - {e}")
            return False
    
    def get_solve_error(self) -> float:
        if self.tracking.reconstruction.is_valid:
            return self.tracking.reconstruction.average_error
        return 999.0
    
    def get_bundle_count(self) -> int:
        return len([t for t in self.tracking.tracks if t.has_bundle])
    
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
