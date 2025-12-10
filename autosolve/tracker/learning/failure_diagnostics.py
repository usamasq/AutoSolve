# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
FailureDiagnostics - Analyzes tracking failures and recommends fixes.

Examines track telemetry to determine WHY tracking failed and
suggests targeted adjustments for retry attempts.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ..constants import EDGE_REGIONS, CENTER_REGIONS


class FailurePattern(Enum):
    """Types of tracking failures."""
    MOTION_BLUR = "motion_blur"
    LOW_CONTRAST = "low_contrast"
    EDGE_DISTORTION = "edge_distortion"
    SCENE_CUT = "scene_cut"
    RAPID_MOTION = "rapid_motion"
    NON_RIGID_MOTION = "non_rigid_motion"  # Waves, water, foliage
    FEATURE_DRIFT = "feature_drift"
    INSUFFICIENT_FEATURES = "insufficient_features"
    UNKNOWN = "unknown"


@dataclass
class DiagnosisResult:
    """Result of failure diagnosis."""
    pattern: FailurePattern
    confidence: float  # 0.0 to 1.0
    description: str
    fix_adjustments: Dict  # Settings adjustments to apply


class FailureDiagnostics:
    """
    Analyzes tracking failures and recommends fixes.
    
    Uses track telemetry to identify failure patterns and
    suggest targeted settings adjustments.
    """
    
    # Thresholds for pattern detection
    HIGH_JITTER_THRESHOLD = 0.5
    SHORT_LIFESPAN_THRESHOLD = 10
    EDGE_FAILURE_THRESHOLD = 0.4
    MASS_DEATH_THRESHOLD = 0.6
    
    def diagnose(self, analysis: Dict, settings: Dict) -> DiagnosisResult:
        """
        Diagnose the failure and recommend adjustments.
        
        Args:
            analysis: Track analysis data from SmartTracker
            settings: Current settings used
            
        Returns:
            DiagnosisResult with pattern and fix recommendations
        """
        # Check patterns in order of likelihood
        patterns = [
            self._check_motion_blur(analysis),
            self._check_rapid_motion(analysis),
            self._check_non_rigid_motion(analysis),
            self._check_low_contrast(analysis, settings),
            self._check_edge_distortion(analysis),
            self._check_scene_cut(analysis),
            self._check_insufficient_features(analysis),
        ]
        
        # Return highest confidence diagnosis
        patterns = [p for p in patterns if p is not None]
        
        if patterns:
            return max(patterns, key=lambda x: x.confidence)
        
        return DiagnosisResult(
            pattern=FailurePattern.UNKNOWN,
            confidence=0.3,
            description="Could not determine specific failure cause",
            fix_adjustments={
                'pattern_size_mult': 1.2,
                'search_size_mult': 1.3,
                'correlation_offset': -0.1,
                'threshold_mult': 0.8,
            }
        )
    
    def _check_motion_blur(self, analysis: Dict) -> Optional[DiagnosisResult]:
        """Check for motion blur pattern: high jitter + short lifespan."""
        tracks = analysis.get('tracks', [])
        if not tracks:
            return None
        
        jittery_short = [
            t for t in tracks 
            if t.get('jitter_score', 0) > self.HIGH_JITTER_THRESHOLD
            and t.get('lifespan', 100) < self.SHORT_LIFESPAN_THRESHOLD
        ]
        
        ratio = len(jittery_short) / len(tracks) if tracks else 0
        
        if ratio > 0.3:
            return DiagnosisResult(
                pattern=FailurePattern.MOTION_BLUR,
                confidence=min(0.9, ratio + 0.3),
                description=f"Motion blur detected ({ratio:.0%} of tracks affected). "
                           f"Tracks are jittery and die quickly.",
                fix_adjustments={
                    'pattern_size_mult': 1.5,  # Larger template
                    'search_size_mult': 1.8,   # Much larger search
                    'correlation_offset': -0.2,  # More lenient matching
                    'motion_model': 'Affine',
                }
            )
        return None
    
    def _check_rapid_motion(self, analysis: Dict) -> Optional[DiagnosisResult]:
        """Check for rapid motion: high velocity tracks."""
        tracks = analysis.get('tracks', [])
        if not tracks:
            return None
        
        # High velocity tracks
        fast_tracks = [
            t for t in tracks 
            if t.get('avg_velocity', 0) > 0.05  # Normalized velocity
        ]
        
        ratio = len(fast_tracks) / len(tracks) if tracks else 0
        
        if ratio > 0.4:
            return DiagnosisResult(
                pattern=FailurePattern.RAPID_MOTION,
                confidence=min(0.85, ratio + 0.2),
                description=f"Rapid camera/object motion ({ratio:.0%} fast tracks). "
                           f"Search area may be too small.",
                fix_adjustments={
                    'search_size_mult': 2.0,   # Double search area
                    'correlation_offset': -0.15,
                    'motion_model': 'Affine',
                }
            )
        return None
    
    def _check_non_rigid_motion(self, analysis: Dict) -> Optional[DiagnosisResult]:
        """
        Check for non-rigid motion: tracks on waves, water, foliage.
        
        Non-rigid objects create high-jitter tracks that don't match
        the camera motion pattern. This is common in ocean/beach footage.
        """
        tracks = analysis.get('tracks', [])
        if not tracks:
            return None
        
        # High jitter tracks (erratic motion typical of waves/foliage)
        jittery_tracks = [
            t for t in tracks 
            if t.get('jitter_score', 0) > self.HIGH_JITTER_THRESHOLD
        ]
        
        ratio = len(jittery_tracks) / len(tracks) if tracks else 0
        
        # Also check for short-lived tracks (features on waves disappear quickly)
        short_lived = [
            t for t in tracks 
            if t.get('lifespan', 100) < self.SHORT_LIFESPAN_THRESHOLD
        ]
        short_ratio = len(short_lived) / len(tracks) if tracks else 0
        
        # Non-rigid pattern: many jittery OR many short-lived tracks
        if ratio > 0.35 or (ratio > 0.2 and short_ratio > 0.5):
            return DiagnosisResult(
                pattern=FailurePattern.NON_RIGID_MOTION,
                confidence=min(0.88, ratio + 0.3),
                description=f"Non-rigid motion detected ({ratio:.0%} jittery tracks). "
                           f"Footage may contain water, waves, or foliage.",
                fix_adjustments={
                    'jitter_filter': True,           # Enable non-rigid filter
                    'jitter_threshold': 0.5,         # Stricter jitter threshold
                    'coherence_threshold': 0.5,      # Stricter coherence check
                    'pattern_size_mult': 1.3,        # Larger patterns more stable
                    'correlation': 0.70,             # Higher correlation to reject bad matches
                }
            )
        return None
    
    def _check_low_contrast(self, analysis: Dict, settings: Dict) -> Optional[DiagnosisResult]:
        """Check for low contrast: few features detected."""
        total_tracks = analysis.get('total_tracks', 0)
        expected_min = 20  # Minimum expected features
        
        if total_tracks < expected_min:
            return DiagnosisResult(
                pattern=FailurePattern.LOW_CONTRAST,
                confidence=0.7,
                description=f"Low contrast/texture ({total_tracks} features found). "
                           f"Scene may lack trackable features.",
                fix_adjustments={
                    'threshold_mult': 0.6,  # Much more sensitive detection
                    'correlation_offset': -0.1,
                    'pattern_size_mult': 0.8,  # Smaller patterns
                }
            )
        return None
    
    def _check_edge_distortion(self, analysis: Dict) -> Optional[DiagnosisResult]:
        """Check for edge distortion: edge regions fail more."""
        region_stats = analysis.get('region_stats', {})
        
        center_regions = ['center', 'top-center', 'mid-left', 'mid-right', 'bottom-center']
        
        def get_success_rate(regions):
            total = sum(region_stats.get(r, {}).get('total_tracks', 0) for r in regions)
            success = sum(region_stats.get(r, {}).get('successful_tracks', 0) for r in regions)
            return success / max(total, 1)
        
        edge_rate = get_success_rate(EDGE_REGIONS)
        center_rate = get_success_rate(center_regions)
        
        if center_rate - edge_rate > 0.3:  # Significant difference
            return DiagnosisResult(
                pattern=FailurePattern.EDGE_DISTORTION,
                confidence=0.75,
                description=f"Edge distortion detected (edges: {edge_rate:.0%}, center: {center_rate:.0%}). "
                           f"Lens distortion may be affecting tracking.",
                fix_adjustments={
                    'avoid_edges': True,  # Special flag
                    'center_weight': 2.0,  # Weight detection toward center
                }
            )
        return None
    
    def _check_scene_cut(self, analysis: Dict) -> Optional[DiagnosisResult]:
        """Check for scene cut: many tracks die at same frame."""
        tracks = analysis.get('tracks', [])
        if not tracks:
            return None
        
        # Count end frames
        end_frames = {}
        for t in tracks:
            end = t.get('end_frame', 0)
            end_frames[end] = end_frames.get(end, 0) + 1
        
        # Check if many tracks end at same frame
        max_deaths = max(end_frames.values()) if end_frames else 0
        death_ratio = max_deaths / len(tracks) if tracks else 0
        
        if death_ratio > self.MASS_DEATH_THRESHOLD:
            death_frame = max(end_frames, key=end_frames.get)
            return DiagnosisResult(
                pattern=FailurePattern.SCENE_CUT,
                confidence=0.9,
                description=f"Possible scene cut at frame {death_frame} "
                           f"({max_deaths} tracks died at once).",
                fix_adjustments={
                    'frame_skip': death_frame,  # Skip this frame
                    'redetect_after': death_frame,
                }
            )
        return None
    
    def _check_insufficient_features(self, analysis: Dict) -> Optional[DiagnosisResult]:
        """Check if we simply don't have enough features."""
        bundle_count = analysis.get('bundle_count', 0)
        
        if bundle_count < 8:  # Minimum for solve
            return DiagnosisResult(
                pattern=FailurePattern.INSUFFICIENT_FEATURES,
                confidence=0.8,
                description=f"Insufficient features for solve ({bundle_count} bundles). "
                           f"Need at least 8 tracked points.",
                fix_adjustments={
                    'threshold_mult': 0.5,  # Detect more features
                    'min_lifespan': 3,  # Accept shorter tracks
                    'correlation_offset': -0.15,
                }
            )
        return None
    
    def apply_fix(self, settings: Dict, diagnosis: DiagnosisResult) -> Dict:
        """
        Apply diagnostic fix adjustments to settings.
        
        Args:
            settings: Current tracking settings
            diagnosis: Diagnosis result with fix recommendations
            
        Returns:
            Updated settings dict
        """
        fixed = settings.copy()
        adj = diagnosis.fix_adjustments
        
        if 'pattern_size_mult' in adj:
            fixed['pattern_size'] = int(settings['pattern_size'] * adj['pattern_size_mult'])
            # Ensure odd
            if fixed['pattern_size'] % 2 == 0:
                fixed['pattern_size'] += 1
        
        if 'search_size_mult' in adj:
            fixed['search_size'] = int(settings['search_size'] * adj['search_size_mult'])
            if fixed['search_size'] % 2 == 0:
                fixed['search_size'] += 1
        
        if 'threshold_mult' in adj:
            fixed['threshold'] = max(0.05, settings['threshold'] * adj['threshold_mult'])
        
        if 'correlation_offset' in adj:
            fixed['correlation'] = max(0.4, min(0.9, 
                settings['correlation'] + adj['correlation_offset']))
        
        # Direct correlation override (used by non-rigid motion fix)
        if 'correlation' in adj and 'correlation_offset' not in adj:
            fixed['correlation'] = max(0.4, min(0.9, adj['correlation']))
        
        if 'motion_model' in adj:
            fixed['motion_model'] = adj['motion_model']
        
        return fixed
