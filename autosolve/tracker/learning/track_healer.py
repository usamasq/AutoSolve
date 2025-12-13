# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
TrackHealer - Identifies and heals track gaps using anchor-based motion.

Uses complete "anchor" tracks to estimate motion for filling gaps in broken tracks.
The key insight: with 3+ perfect tracks spanning a gap, we can estimate the 
motion of any point in the frame during that gap.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import math

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None
    Vector = None

from ..utils import get_region


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrackSignature:
    """Signature of a track endpoint for matching."""
    track_name: str
    frame: int
    position: List[float]           # [x, y] normalized
    velocity: List[float]           # [dx, dy] per frame
    acceleration: List[float]       # [ddx, ddy] change in velocity
    region: str
    confidence: float               # Track quality at this point
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AnchorTrack:
    """A complete reference track used for motion estimation."""
    name: str
    start_frame: int
    end_frame: int
    positions: Dict[int, List[float]]   # {frame: [x, y]}
    velocities: Dict[int, List[float]]  # {frame: [dx, dy]}
    region: str
    lifespan: int
    quality_score: float                # 0-1 quality as reference
    
    def covers_gap(self, gap_start: int, gap_end: int) -> bool:
        """Check if this anchor spans the entire gap."""
        return self.start_frame <= gap_start and self.end_frame >= gap_end
    
    def get_velocity_at(self, frame: int) -> Optional[List[float]]:
        """Get velocity at a specific frame, with interpolation."""
        if frame in self.velocities:
            return self.velocities[frame]
        
        # Interpolate between nearest frames
        frames = sorted(self.velocities.keys())
        if not frames:
            return None
        
        if frame < frames[0]:
            return self.velocities[frames[0]]
        if frame > frames[-1]:
            return self.velocities[frames[-1]]
        
        # Find bracketing frames
        for i, f in enumerate(frames[:-1]):
            if f <= frame < frames[i + 1]:
                t = (frame - f) / (frames[i + 1] - f)
                v1 = self.velocities[f]
                v2 = self.velocities[frames[i + 1]]
                return [
                    v1[0] + t * (v2[0] - v1[0]),
                    v1[1] + t * (v2[1] - v1[1])
                ]
        
        return None


@dataclass
class HealingCandidate:
    """A pair of tracks that might be joinable."""
    track_a_name: str
    track_a_end_frame: int
    track_a_end_pos: List[float]
    track_b_name: str
    track_b_start_frame: int
    track_b_start_pos: List[float]
    gap_frames: int
    match_score: float              # 0-1 overall confidence
    spatial_score: float            # Position prediction accuracy
    motion_score: float             # Velocity consistency
    neighbor_score: float           # Correlation with anchors
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class InterpolationTrainingData:
    """Training data for ML-based interpolation."""
    # Gap metadata
    gap_start_frame: int
    gap_end_frame: int
    gap_start_pos: List[float]
    gap_end_pos: List[float]
    gap_start_vel: List[float]
    gap_end_vel: List[float]
    
    # Anchor tracks during gap (THE KEY DATA)
    anchor_count: int
    anchor_positions: List[Dict[int, List[float]]]  # Per-anchor {frame: [x,y]}
    anchor_velocities: List[Dict[int, List[float]]]
    anchor_regions: List[str]
    anchor_qualities: List[float]
    
    # Interpolation result
    interpolated_positions: List[List[float]]
    method_used: str                      # 'spline', 'anchor_weighted', etc.
    
    # Ground truth (after solve)
    post_heal_error: float = 0.0
    heal_success: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# TRACK HEALER
# =============================================================================

class TrackHealer:
    """
    Identifies and heals track gaps using anchor-based motion estimation.
    
    Algorithm:
    1. Find "anchor" tracks - complete, high-quality tracks spanning gaps
    2. Find healing candidates - pairs of tracks with matching endpoints
    3. Score candidates using spatial, motion, and neighbor consistency
    4. Interpolate gaps using weighted anchor motion
    5. Insert markers to heal the gap
    """
    
    # Configuration (relaxed for more aggressive healing)
    MAX_GAP_FRAMES = 50             # Max gap size to attempt healing (was 30)
    MIN_MATCH_SCORE = 0.5           # Minimum score to heal (was 0.7)
    MIN_ANCHOR_TRACKS = 2           # Need at least 2 anchors (was 3)
    MIN_ANCHOR_LIFESPAN = 30        # Anchor must span 30+ frames (was 50)
    SPATIAL_TOLERANCE = 0.08        # 8% of frame for position matching (was 5%)
    VELOCITY_TOLERANCE = 0.02       # Velocity matching tolerance (was 0.01)
    
    def __init__(self):
        self.anchors: List[AnchorTrack] = []
        self.candidates: List[HealingCandidate] = []
        self.training_data: List[InterpolationTrainingData] = []
    
    # =========================================================================
    # ANCHOR TRACK IDENTIFICATION
    # =========================================================================
    
    def find_anchor_tracks(self, tracking, min_lifespan: int = None) -> List[AnchorTrack]:
        """
        Find complete, high-quality tracks to use as motion references.
        
        Anchor criteria:
        - Long lifespan (50+ frames by default)
        - Low jitter (stable tracking)
        - Good region (prefer center)
        """
        if min_lifespan is None:
            min_lifespan = self.MIN_ANCHOR_LIFESPAN
        
        self.anchors.clear()
        
        for track in tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < min_lifespan:
                continue
            
            markers.sort(key=lambda m: m.frame)
            lifespan = markers[-1].frame - markers[0].frame
            
            if lifespan < min_lifespan:
                continue
            
            # Build position and velocity maps
            positions = {}
            velocities = {}
            
            for i, marker in enumerate(markers):
                positions[marker.frame] = [marker.co.x, marker.co.y]
                
                if i > 0:
                    prev = markers[i - 1]
                    dx = marker.co.x - prev.co.x
                    dy = marker.co.y - prev.co.y
                    frame_diff = marker.frame - prev.frame
                    if frame_diff > 0:
                        velocities[marker.frame] = [dx / frame_diff, dy / frame_diff]
            
            # Calculate quality score
            quality = self._compute_anchor_quality(markers, lifespan)
            
            # Get average region
            avg_x = sum(m.co.x for m in markers) / len(markers)
            avg_y = sum(m.co.y for m in markers) / len(markers)
            region = get_region(avg_x, avg_y)
            
            anchor = AnchorTrack(
                name=track.name,
                start_frame=markers[0].frame,
                end_frame=markers[-1].frame,
                positions=positions,
                velocities=velocities,
                region=region,
                lifespan=lifespan,
                quality_score=quality
            )
            self.anchors.append(anchor)
        
        # Sort by quality
        self.anchors.sort(key=lambda a: a.quality_score, reverse=True)
        
        print(f"AutoSolve: Found {len(self.anchors)} anchor tracks "
              f"(min lifespan: {min_lifespan}, need {self.MIN_ANCHOR_TRACKS}+ for healing)")
        
        return self.anchors
    
    def _compute_anchor_quality(self, markers, lifespan: int) -> float:
        """Compute quality score for potential anchor track."""
        quality = 1.0
        
        # Lifespan bonus (longer = better)
        if lifespan >= 100:
            quality *= 1.2
        elif lifespan >= 75:
            quality *= 1.1
        
        # Jitter penalty
        if len(markers) >= 3:
            velocities = []
            for i in range(1, len(markers)):
                dx = markers[i].co.x - markers[i-1].co.x
                dy = markers[i].co.y - markers[i-1].co.y
                velocities.append((dx**2 + dy**2)**0.5)
            
            if velocities:
                mean_v = sum(velocities) / len(velocities)
                if mean_v > 0:
                    variance = sum((v - mean_v)**2 for v in velocities) / len(velocities)
                    jitter = (variance**0.5) / mean_v
                    quality *= max(0.5, 1.0 - jitter)
        
        # Region preference (center is best)
        avg_x = sum(m.co.x for m in markers) / len(markers)
        avg_y = sum(m.co.y for m in markers) / len(markers)
        center_dist = ((avg_x - 0.5)**2 + (avg_y - 0.5)**2)**0.5
        
        if center_dist < 0.2:
            quality *= 1.1
        elif center_dist > 0.4:
            quality *= 0.9
        
        return min(1.0, quality)
    
    # =========================================================================
    # HEALING CANDIDATE DETECTION
    # =========================================================================
    
    def find_healing_candidates(self, tracking) -> List[HealingCandidate]:
        """
        Find pairs of tracks that might be the same point before/after a gap.
        """
        self.candidates.clear()
        
        # Extract signatures for all track endpoints
        end_signatures = []    # Tracks ending (potential gap starts)
        start_signatures = []  # Tracks starting (potential gap ends)
        
        for track in tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 3:
                continue
            
            markers.sort(key=lambda m: m.frame)
            
            # End signature
            end_sig = self._extract_signature(track.name, markers, is_end=True)
            if end_sig:
                end_signatures.append(end_sig)
            
            # Start signature
            start_sig = self._extract_signature(track.name, markers, is_end=False)
            if start_sig:
                start_signatures.append(start_sig)
        
        # Find matching pairs
        for end_sig in end_signatures:
            for start_sig in start_signatures:
                # Skip same track
                if end_sig.track_name == start_sig.track_name:
                    continue
                
                # Check gap size
                gap = start_sig.frame - end_sig.frame
                if gap <= 0 or gap > self.MAX_GAP_FRAMES:
                    continue
                
                # Score the match
                candidate = self._score_candidate(end_sig, start_sig, gap)
                if candidate.match_score >= self.MIN_MATCH_SCORE * 0.5:  # Keep marginal candidates for training
                    self.candidates.append(candidate)
        
        # Sort by score (best first)
        self.candidates.sort(key=lambda c: c.match_score, reverse=True)
        
        if not self.candidates:
            print("AutoSolve: No healing candidates found (no broken tracks or gaps too large)")
        else:
            print(f"AutoSolve: Found {len(self.candidates)} healing candidates "
                  f"(need score >= {self.MIN_MATCH_SCORE} to heal)")
        
        return self.candidates
    
    def _extract_signature(self, track_name: str, markers, is_end: bool) -> Optional[TrackSignature]:
        """Extract signature at track endpoint."""
        try:
            if is_end:
                # Use last few markers
                pts = markers[-min(5, len(markers)):]
            else:
                # Use first few markers
                pts = markers[:min(5, len(markers))]
            
            if len(pts) < 2:
                return None
            
            # Position at endpoint
            endpoint = pts[-1] if is_end else pts[0]
            pos = [endpoint.co.x, endpoint.co.y]
            
            # Velocity (average over last/first few points)
            velocities = []
            for i in range(1, len(pts)):
                dx = pts[i].co.x - pts[i-1].co.x
                dy = pts[i].co.y - pts[i-1].co.y
                frame_diff = pts[i].frame - pts[i-1].frame
                if frame_diff > 0:
                    velocities.append([dx / frame_diff, dy / frame_diff])
            
            if not velocities:
                return None
            
            vel = [
                sum(v[0] for v in velocities) / len(velocities),
                sum(v[1] for v in velocities) / len(velocities)
            ]
            
            # Acceleration
            accel = [0.0, 0.0]
            if len(velocities) >= 2:
                accel = [
                    velocities[-1][0] - velocities[0][0],
                    velocities[-1][1] - velocities[0][1]
                ]
            
            # Confidence from velocity consistency
            if len(velocities) >= 2:
                vel_variance = sum(
                    (v[0] - vel[0])**2 + (v[1] - vel[1])**2 
                    for v in velocities
                ) / len(velocities)
                confidence = max(0.1, 1.0 - vel_variance / 0.001)
            else:
                confidence = 0.5
            
            return TrackSignature(
                track_name=track_name,
                frame=endpoint.frame,
                position=pos,
                velocity=vel,
                acceleration=accel,
                region=get_region(pos[0], pos[1]),
                confidence=confidence
            )
        except (AttributeError, IndexError, ZeroDivisionError):
            return None
    
    def _score_candidate(self, end_sig: TrackSignature, start_sig: TrackSignature, 
                         gap: int) -> HealingCandidate:
        """Score how likely these two track segments are the same point."""
        
        # 1. Spatial score: Does end position + velocity predict start position?
        predicted_x = end_sig.position[0] + end_sig.velocity[0] * gap
        predicted_y = end_sig.position[1] + end_sig.velocity[1] * gap
        
        actual_x = start_sig.position[0]
        actual_y = start_sig.position[1]
        
        spatial_error = ((predicted_x - actual_x)**2 + (predicted_y - actual_y)**2)**0.5
        spatial_score = max(0, 1.0 - spatial_error / self.SPATIAL_TOLERANCE)
        
        # 2. Motion score: Are velocities consistent?
        vel_diff = (
            (end_sig.velocity[0] - start_sig.velocity[0])**2 +
            (end_sig.velocity[1] - start_sig.velocity[1])**2
        )**0.5
        motion_score = max(0, 1.0 - vel_diff / self.VELOCITY_TOLERANCE)
        
        # 3. Neighbor score: Do anchors show consistent motion during gap?
        neighbor_score = self._compute_neighbor_score(end_sig, start_sig, gap)
        
        # Combined score (weighted average)
        match_score = (
            spatial_score * 0.4 +
            motion_score * 0.3 +
            neighbor_score * 0.3
        ) * min(end_sig.confidence, start_sig.confidence)
        
        return HealingCandidate(
            track_a_name=end_sig.track_name,
            track_a_end_frame=end_sig.frame,
            track_a_end_pos=end_sig.position,
            track_b_name=start_sig.track_name,
            track_b_start_frame=start_sig.frame,
            track_b_start_pos=start_sig.position,
            gap_frames=gap,
            match_score=match_score,
            spatial_score=spatial_score,
            motion_score=motion_score,
            neighbor_score=neighbor_score
        )
    
    def _compute_neighbor_score(self, end_sig: TrackSignature, 
                                start_sig: TrackSignature, gap: int) -> float:
        """Score based on how well anchor tracks support this match."""
        if not self.anchors:
            return 0.5  # Neutral if no anchors
        
        # Find anchors that span this gap
        spanning_anchors = [
            a for a in self.anchors 
            if a.covers_gap(end_sig.frame, start_sig.frame)
        ]
        
        if len(spanning_anchors) < 2:
            return 0.3  # Low score without enough anchors
        
        # Compute average anchor motion during gap
        anchor_motions = []
        for anchor in spanning_anchors[:5]:  # Use top 5 anchors
            start_pos = anchor.positions.get(end_sig.frame)
            end_pos = anchor.positions.get(start_sig.frame)
            
            if start_pos and end_pos:
                motion = [
                    end_pos[0] - start_pos[0],
                    end_pos[1] - start_pos[1]
                ]
                anchor_motions.append(motion)
        
        if not anchor_motions:
            return 0.4
        
        # Expected motion for target
        avg_motion = [
            sum(m[0] for m in anchor_motions) / len(anchor_motions),
            sum(m[1] for m in anchor_motions) / len(anchor_motions)
        ]
        
        # Actual motion
        actual_motion = [
            start_sig.position[0] - end_sig.position[0],
            start_sig.position[1] - end_sig.position[1]
        ]
        
        # Compare
        motion_diff = (
            (actual_motion[0] - avg_motion[0])**2 +
            (actual_motion[1] - avg_motion[1])**2
        )**0.5
        
        # Normalize by gap size
        normalized_diff = motion_diff / max(gap * 0.01, 0.01)
        
        return max(0, 1.0 - normalized_diff)
    
    # =========================================================================
    # GAP INTERPOLATION
    # =========================================================================
    
    def interpolate_with_anchors(self, candidate: HealingCandidate, 
                                  anchors: List[AnchorTrack]) -> List[List[float]]:
        """
        Interpolate positions for gap frames using anchor motion.
        
        Algorithm:
        1. For each frame in gap, get weighted anchor velocities
        2. Weight by inverse distance to target position
        3. Integrate path from start to end
        4. Apply spline smoothing with endpoint constraints
        """
        gap_start = candidate.track_a_end_frame
        gap_end = candidate.track_b_start_frame
        start_pos = candidate.track_a_end_pos
        end_pos = candidate.track_b_start_pos
        
        # Filter to anchors spanning this gap
        spanning = [a for a in anchors if a.covers_gap(gap_start, gap_end)]
        
        if not spanning:
            # Fallback to simple linear interpolation
            return self._linear_interpolate(start_pos, end_pos, gap_end - gap_start)
        
        # Build path using weighted anchor velocities
        path = [start_pos]
        current_pos = list(start_pos)
        
        for frame in range(gap_start + 1, gap_end):
            # Compute weighted velocity from anchors
            weighted_vel = [0.0, 0.0]
            total_weight = 0.0
            
            for anchor in spanning:
                anchor_pos = anchor.positions.get(frame)
                anchor_vel = anchor.get_velocity_at(frame)
                
                if anchor_pos and anchor_vel:
                    # Weight by inverse distance (closer anchors have more influence)
                    dist = (
                        (anchor_pos[0] - current_pos[0])**2 +
                        (anchor_pos[1] - current_pos[1])**2
                    )**0.5
                    
                    # Also weight by anchor quality
                    weight = anchor.quality_score / max(dist, 0.01)
                    
                    weighted_vel[0] += anchor_vel[0] * weight
                    weighted_vel[1] += anchor_vel[1] * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_vel[0] /= total_weight
                weighted_vel[1] /= total_weight
            
            # Apply velocity
            current_pos = [
                current_pos[0] + weighted_vel[0],
                current_pos[1] + weighted_vel[1]
            ]
            path.append(list(current_pos))
        
        # Adjust path to ensure it ends at target
        path = self._adjust_path_to_endpoint(path, start_pos, end_pos)
        
        return path
    
    def _linear_interpolate(self, start: List[float], end: List[float], 
                            num_frames: int) -> List[List[float]]:
        """Simple linear interpolation fallback."""
        path = []
        for i in range(1, num_frames):
            t = i / num_frames
            pos = [
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1])
            ]
            path.append(pos)
        return path
    
    def _adjust_path_to_endpoint(self, path: List[List[float]], 
                                  start: List[float], end: List[float]) -> List[List[float]]:
        """Adjust interpolated path to exactly hit endpoints."""
        if not path:
            return path
        
        # Calculate drift
        actual_end = path[-1]
        drift = [
            end[0] - actual_end[0],
            end[1] - actual_end[1]
        ]
        
        # Distribute drift correction linearly
        adjusted = []
        n = len(path)
        for i, pos in enumerate(path):
            t = (i + 1) / n
            adjusted_pos = [
                pos[0] + drift[0] * t,
                pos[1] + drift[1] * t
            ]
            adjusted.append(adjusted_pos)
        
        return adjusted
    
    # =========================================================================
    # TRACK HEALING
    # =========================================================================
    
    def heal_track(self, candidate: HealingCandidate, tracking, 
                   anchors: List[AnchorTrack]) -> bool:
        """
        Heal a track gap by inserting interpolated markers.
        
        Returns True if successful.
        """
        if not bpy:
            return False
        
        try:
            # Find track A
            track_a = None
            for t in tracking.tracks:
                if t.name == candidate.track_a_name:
                    track_a = t
                    break
            
            if not track_a:
                return False
            
            # Interpolate positions
            positions = self.interpolate_with_anchors(candidate, anchors)
            
            # Insert markers
            gap_start = candidate.track_a_end_frame
            
            for i, pos in enumerate(positions):
                frame = gap_start + i + 1
                
                # Insert or update marker
                marker = track_a.markers.find_frame(frame)
                if not marker:
                    marker = track_a.markers.insert_frame(frame)
                
                marker.co = Vector((pos[0], pos[1]))
                marker.mute = False
            
            # TODO: Optionally merge track_b into track_a and delete track_b
            
            print(f"AutoSolve: Healed gap {candidate.track_a_name} → {candidate.track_b_name} "
                  f"({candidate.gap_frames} frames)")
            
            return True
            
        except Exception as e:
            print(f"AutoSolve: Healing failed: {e}")
            return False
    
    # =========================================================================
    # TRAINING DATA COLLECTION
    # =========================================================================
    
    def collect_training_data(self, candidate: HealingCandidate, 
                               anchors: List[AnchorTrack],
                               positions: List[List[float]],
                               success: bool,
                               post_error: float = 0.0) -> InterpolationTrainingData:
        """Collect training data from a healing attempt."""
        
        gap_start = candidate.track_a_end_frame
        gap_end = candidate.track_b_start_frame
        
        # Get spanning anchors
        spanning = [a for a in anchors if a.covers_gap(gap_start, gap_end)]
        
        # Build anchor data for gap frames
        anchor_positions = []
        anchor_velocities = []
        anchor_regions = []
        anchor_qualities = []
        
        for anchor in spanning[:5]:  # Limit to 5 anchors
            ap = {}
            av = {}
            for frame in range(gap_start, gap_end + 1):
                if frame in anchor.positions:
                    ap[frame] = anchor.positions[frame]
                vel = anchor.get_velocity_at(frame)
                if vel:
                    av[frame] = vel
            
            anchor_positions.append(ap)
            anchor_velocities.append(av)
            anchor_regions.append(anchor.region)
            anchor_qualities.append(anchor.quality_score)
        
        # Estimate start/end velocities
        start_vel = [0.0, 0.0]
        end_vel = [0.0, 0.0]
        if positions and len(positions) >= 2:
            start_vel = [
                positions[0][0] - candidate.track_a_end_pos[0],
                positions[0][1] - candidate.track_a_end_pos[1]
            ]
            end_vel = [
                candidate.track_b_start_pos[0] - positions[-1][0],
                candidate.track_b_start_pos[1] - positions[-1][1]
            ]
        
        data = InterpolationTrainingData(
            gap_start_frame=gap_start,
            gap_end_frame=gap_end,
            gap_start_pos=candidate.track_a_end_pos,
            gap_end_pos=candidate.track_b_start_pos,
            gap_start_vel=start_vel,
            gap_end_vel=end_vel,
            anchor_count=len(spanning),
            anchor_positions=anchor_positions,
            anchor_velocities=anchor_velocities,
            anchor_regions=anchor_regions,
            anchor_qualities=anchor_qualities,
            interpolated_positions=positions,
            method_used='anchor_weighted',
            post_heal_error=post_error,
            heal_success=success
        )
        
        self.training_data.append(data)
        return data
    
    def get_training_data(self) -> List[Dict]:
        """Get all collected training data as dicts."""
        return [d.to_dict() for d in self.training_data]
    
    def clear_training_data(self):
        """Clear collected training data."""
        self.training_data.clear()
    
    def merge_overlapping_segments(self, tracking, min_overlap: int = 5) -> int:
        """
        Find and average track segments with significant frame overlap.
        
        When two tracks have overlapping frame ranges (>= min_overlap frames)
        AND are spatially close, they likely represent the same feature
        tracked twice. Averaging them produces a more accurate combined track.
        
        This is called during the healing phase to merge redundant tracks.
        
        Args:
            tracking: bpy.types.MovieTracking object
            min_overlap: Minimum overlapping frames required (default: 5)
            
        Returns:
            Number of track pairs merged
        """
        try:
            from ..averaging import merge_overlapping_segments as merge_fn
            return merge_fn(tracking, min_overlap)
        except Exception as e:
            print(f"AutoSolve: merge_overlapping_segments failed: {e}")
            return 0

