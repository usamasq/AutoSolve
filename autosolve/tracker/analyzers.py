# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Analyzer classes for tracking pattern analysis and coverage analysis.

Extracted from smart_tracker.py for better modularity.
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
from mathutils import Vector

from .constants import REGIONS
from .utils import get_region, get_region_bounds, calculate_jitter


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
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


@dataclass
class CoverageData:
    """Coverage data for a region-time segment."""
    region: str
    segment: Tuple[int, int]
    track_count: int = 0
    successful_tracks: int = 0
    avg_lifespan: float = 0.0
    needs_more: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# TRACK ANALYZER (Learning Component)
# ═══════════════════════════════════════════════════════════════════════════

class TrackAnalyzer:
    """Analyzes tracking patterns and learns from them."""
    
    def __init__(self):
        self.track_stats: List[TrackStats] = []
        self.region_stats: Dict[str, RegionStats] = {
            r: RegionStats(name=r) for r in REGIONS
        }
        self.dead_zones: Set[str] = set()
        self.sweet_spots: Set[str] = set()
        self.iteration: int = 0
    
    def analyze_tracks(self, tracking, min_lifespan: int = 5) -> Dict:
        """Analyze all tracks and calculate statistics."""
        self.track_stats.clear()
        
        for region in self.region_stats.values():
            region.total_tracks = 0
            region.successful_tracks = 0
            region.avg_lifespan = 0.0
        
        region_lifespans = {r: [] for r in REGIONS}
        
        for track in tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            markers.sort(key=lambda x: x.frame)
            
            lifespan = markers[-1].frame - markers[0].frame
            
            avg_x = sum(m.co.x for m in markers) / len(markers)
            avg_y = sum(m.co.y for m in markers) / len(markers)
            region = get_region(avg_x, avg_y)
            
            displacement = (Vector(markers[-1].co) - Vector(markers[0].co)).length
            avg_velocity = displacement / max(lifespan, 1)
            
            jitter = calculate_jitter(markers)
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

class CoverageAnalyzer:
    """
    Tracks spatial and temporal distribution of markers.
    
    Industry standard: Good camera solves require:
    - Tracks distributed across the frame (not clustered)
    - Tracks spanning the full timeline (not just parts)
    - Minimum parallax requirements met
    """
    
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
        for region in REGIONS:
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
    
    def analyze_tracking(self, tracking, min_lifespan: int = 5):
        """
        Analyze current tracking data for coverage.
        
        Updates the coverage grid with actual track distribution.
        """
        # Reset counts
        self._init_coverage_grid()
        
        region_lifespans: Dict[str, Dict[Tuple, List[int]]] = {
            r: {s: [] for s in self.coverage[r].keys()} for r in REGIONS
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
            region = get_region(avg_x, avg_y)
            
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
        for region in REGIONS:
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
        
        for region in REGIONS:
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
        total_segments = len(list(self.coverage[REGIONS[0]].keys()))
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
        for region in REGIONS:
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
