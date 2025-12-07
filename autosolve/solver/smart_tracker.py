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
from mathutils import Vector
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path


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
        'pattern_size': 25,
        'search_size': 111,
        'correlation': 0.65,
        'threshold': 0.25,
        'motion_model': 'Affine',
    },
    '4K_30fps': {
        'pattern_size': 23,
        'search_size': 91,
        'correlation': 0.68,
        'threshold': 0.28,
        'motion_model': 'LocRot',
    },
    '4K_60fps': {
        'pattern_size': 21,
        'search_size': 71,
        'correlation': 0.70,
        'threshold': 0.30,
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
# LOCAL LEARNING MODEL
# ═══════════════════════════════════════════════════════════════════════════

class LocalLearningModel:
    """
    Stores and retrieves local learning data.
    
    The model starts with pre-trained defaults and adapts
    based on user's footage over time.
    """
    
    def __init__(self):
        self.data_dir = self._get_data_dir()
        self.model_path = self.data_dir / 'model.json'
        self.model = self._load_or_create()
    
    def _get_data_dir(self) -> Path:
        """Get the data directory for storing learning data."""
        try:
            data_dir = Path(bpy.utils.user_resource('DATAFILES')) / 'eztrack'
        except:
            data_dir = Path(bpy.app.tempdir) / 'eztrack'
        
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def _load_or_create(self) -> Dict:
        """Load existing model or create new one with defaults."""
        if self.model_path.exists():
            try:
                with open(self.model_path) as f:
                    model = json.load(f)
                print(f"AutoSolve: Loaded local model ({model.get('session_count', 0)} sessions)")
                return model
            except Exception as e:
                print(f"AutoSolve: Error loading model: {e}")
        
        # Create new model with pre-trained defaults
        return {
            'version': 2,
            'session_count': 0,
            'footage_classes': {},
            'region_models': {},
            # Pre-trained preferences
            'pretrained_used': True,
        }
    
    def save(self):
        """Save model to disk."""
        with open(self.model_path, 'w') as f:
            json.dump(self.model, f, indent=2)
    
    def get_settings_for_class(self, footage_class: str) -> Optional[Dict]:
        """Get learned settings for a footage class."""
        class_data = self.model['footage_classes'].get(footage_class)
        
        if class_data and class_data.get('sample_count', 0) >= 2:
            return class_data.get('best_settings')
        
        return None
    
    def get_dead_zones_for_class(self, footage_class: str) -> Set[str]:
        """Get learned dead zones for a specific footage class."""
        class_data = self.model['footage_classes'].get(footage_class, {})
        region_data = class_data.get('region_stats', {})
        
        dead = set()
        for region, stats in region_data.items():
            total = stats.get('total', 0)
            successful = stats.get('successful', 0)
            if total >= 5:  # Need enough samples
                rate = successful / total
                if rate < 0.25:
                    dead.add(region)
        
        return dead
    
    def update_from_session(self, footage_class: str, settings: Dict, 
                            success: bool, solve_error: float, 
                            analysis: Dict):
        """
        Update model with session results.
        
        This is where the LOCAL LEARNING happens:
        1. Track count per footage class increases
        2. Best settings are saved if this solve is better
        3. Region success rates are updated per footage class
        4. Dead zones are computed from actual region data
        
        After 2+ sessions, the learned settings override defaults.
        """
        self.model['session_count'] += 1
        
        # Initialize footage class data if new
        if footage_class not in self.model['footage_classes']:
            self.model['footage_classes'][footage_class] = {
                'sample_count': 0,
                'success_count': 0,
                'best_settings': None,
                'best_error': 999.0,
                'region_stats': {},  # Per-class region tracking
            }
        
        fc = self.model['footage_classes'][footage_class]
        fc['sample_count'] += 1
        
        if success:
            fc['success_count'] += 1
            
            # If this is the best result so far, save settings
            if solve_error < fc.get('best_error', 999.0):
                fc['best_error'] = solve_error
                fc['best_settings'] = settings.copy()
                print(f"AutoSolve: New best settings for {footage_class}! (error: {solve_error:.2f}px)")
        
        # Update region data FOR THIS FOOTAGE CLASS
        if 'region_stats' not in fc:
            fc['region_stats'] = {}
        
        for region, stats in analysis.get('region_stats', {}).items():
            if region not in fc['region_stats']:
                fc['region_stats'][region] = {'total': 0, 'successful': 0}
            
            fc['region_stats'][region]['total'] += stats.get('total_tracks', 0)
            fc['region_stats'][region]['successful'] += stats.get('successful_tracks', 0)
        
        # Also update global region models for cross-footage-class learning
        for region, stats in analysis.get('region_stats', {}).items():
            if region not in self.model['region_models']:
                self.model['region_models'][region] = {
                    'total': 0,
                    'successful': 0,
                }
            
            rm = self.model['region_models'][region]
            rm['total'] += stats.get('total_tracks', 0)
            rm['successful'] += stats.get('successful_tracks', 0)
        
        self.save()
        
        # Debug output showing what was learned
        print(f"AutoSolve: Model updated - {fc['sample_count']} sessions for {footage_class}")
        
        # Show learned dead zones if any
        dead_zones = self.get_dead_zones_for_class(footage_class)
        if dead_zones:
            print(f"AutoSolve: Learned dead zones for {footage_class}: {', '.join(dead_zones)}")
    
    def get_dead_zones(self) -> Set[str]:
        """Get regions with historically low success rates (global)."""
        dead = set()
        
        for region, data in self.model.get('region_models', {}).items():
            total = data.get('total', 0)
            successful = data.get('successful', 0)
            
            if total >= 10:  # Need enough samples
                rate = successful / total
                if rate < 0.25:
                    dead.add(region)
        
        return dead


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
        self.local_model = LocalLearningModel()
        
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
        
        # Load initial settings
        self._load_initial_settings()
    
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
        local_settings = self.local_model.get_settings_for_class(self.footage_class)
        if local_settings:
            self.current_settings = local_settings.copy()
            print(f"AutoSolve: Using LEARNED settings for {self.footage_class}")
        
        # Step 4: Get LEARNED dead zones (overrides predictions if we have data)
        learned_dead_zones = self.local_model.get_dead_zones_for_class(self.footage_class)
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
            'region_performance': {},
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
            training_data['region_performance'][region] = data
        
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
        Detect features with balanced distribution across screen regions.
        
        Industry standard: Instead of carpet-bombing with 200+ markers,
        strategically place 2-4 markers per region for even coverage.
        
        Respects user-placed markers as priority hints.
        
        Args:
            markers_per_region: Target markers per screen region (default 3)
            
        Returns:
            Total number of features detected
        """
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
        
        print(f"AutoSolve: Strategic detection - {total_detected} markers "
              f"({len(priority_regions)} priority regions)")
        return total_detected
    
    def detect_in_region(self, region: str, count: int = 3) -> int:
        """
        Detect features within a specific screen region.
        
        Uses Blender's detect_features with constrained margin to target
        a specific area of the frame.
        
        Args:
            region: Region name (e.g., 'top-left', 'center')
            count: Target number of markers
            
        Returns:
            Number of features detected
        """
        bounds = self.coverage_analyzer.get_region_bounds(region)
        x_min, y_min, x_max, y_max = bounds
        
        # Convert normalized to pixel coordinates
        width = self.clip.size[0]
        height = self.clip.size[1]
        
        # Calculate pixel margins to constrain detection to this region
        margin_left = int(x_min * width)
        margin_right = int((1.0 - x_max) * width)
        margin_bottom = int(y_min * height)
        margin_top = int((1.0 - y_max) * height)
        
        # Use smallest margin (Blender uses uniform margin)
        # So we'll detect in the region then filter to our bounds
        initial_count = len(self.tracking.tracks)
        
        # Detect with low threshold to get candidates
        threshold = self.current_settings.get('threshold', 0.3) * 0.7  # Lower for more candidates
        
        self._run_ops(
            bpy.ops.clip.detect_features,
            threshold=threshold,
            min_distance=100,  # Larger distance to avoid clustering
            margin=max(10, min(margin_left, margin_right, margin_bottom, margin_top) // 2),
            placement='FRAME'
        )
        
        # Filter: keep only tracks in target region, limit count
        new_tracks = list(self.tracking.tracks)[initial_count:]
        kept = 0
        
        for track in new_tracks:
            marker = track.markers.find_frame(bpy.context.scene.frame_current)
            if not marker:
                track.select = True  # Mark for deletion
                continue
            
            # Check if in target region
            track_region = self.coverage_analyzer.get_region(marker.co.x, marker.co.y)
            if track_region == region and kept < count:
                track.select = False
                self._apply_track_settings(track)
                kept += 1
            else:
                track.select = True  # Mark for deletion
        
        # Delete tracks outside region or over count
        try:
            self._run_ops(bpy.ops.clip.delete_track)
        except:
            pass
        
        return kept
    
    def fill_coverage_gaps(self) -> int:
        """
        Fill gaps in coverage by adding markers to weak zones.
        
        Called after initial tracking pass to ensure balanced distribution.
        
        Returns:
            Number of new markers added
        """
        # Analyze current coverage
        self.coverage_analyzer.analyze_tracking(self.tracking)
        summary = self.coverage_analyzer.get_coverage_summary()
        
        if summary['is_balanced']:
            print(f"AutoSolve: Coverage is balanced ({summary['regions_with_tracks']}/9 regions)")
            return 0
        
        # Get weak zones (regions needing more tracks)
        weak_zones = self.coverage_analyzer.get_weak_zones()
        if not weak_zones:
            print("AutoSolve: No weak zones identified")
            return 0
        
        total_added = 0
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
            total_added += added
            processed_regions.add(region)
            
            print(f"AutoSolve: Added {added} markers to {region} at frame {target_frame}")
        
        return total_added
    
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
            Dict with iteration results
        """
        self.strategic_iteration += 1
        print(f"AutoSolve: Strategic iteration {self.strategic_iteration}")
        
        # Analyze coverage
        summary = self.get_coverage_analysis()
        
        result = {
            'iteration': self.strategic_iteration,
            'coverage_before': summary.copy(),
            'markers_added': 0,
            'coverage_after': None,
        }
        
        if summary['is_balanced']:
            print("AutoSolve: Coverage is balanced, no more iterations needed")
            return result
        
        # Fill gaps
        result['markers_added'] = self.fill_coverage_gaps()
        
        # Re-analyze
        result['coverage_after'] = self.get_coverage_analysis()
        
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
        self._run_ops(bpy.ops.clip.track_markers, backwards=backwards, sequence=False)
    
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
        if self.last_analysis:
            self.local_model.update_from_session(
                self.footage_class,
                self.current_settings,
                success,
                solve_error,
                self.last_analysis
            )
    
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
        
        try:
            self._run_ops(bpy.ops.clip.solve_camera)
            return self.tracking.reconstruction.is_valid
        except RuntimeError:
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
