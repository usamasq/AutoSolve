# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Data schema definitions for AutoSolve ML data collection pipeline.
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any


@dataclass
class ClipMetadata:
    """Metadata of the video clip."""
    clip_name: str
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass
class TrackingSettings:
    """Settings used for tracking."""
    quality_preset: str
    footage_type: str
    robust_mode: bool
    tripod_mode: bool
    pattern_size: int
    search_size: int
    correlation: float
    threshold: float
    motion_model: str


@dataclass
class TrackSample:
    """Per-track data for ML training."""
    track_name: str
    region: str
    positions: List[Tuple[float, float]]  # Normalized (x, y) per frame
    velocities: List[Tuple[float, float]]  # (dx, dy) per frame
    jitter_scores: List[float]
    lifespan: int
    survived: bool  # Survived to contribute to solve
    has_bundle: bool  # Got a 3D position
    average_error: float  # Average reprojection error of this track


@dataclass
class SolveSample:
    """Per-solve-attempt data."""
    clip_metadata: ClipMetadata
    settings: TrackingSettings
    tracks: List[TrackSample]
    solve_success: bool
    solve_error: float
    bundle_count: int
    bundle_ratio: float
    runtime_seconds: float


def solve_sample_to_dict(sample: SolveSample) -> Dict[str, Any]:
    """Convert SolveSample instance to a dictionary for JSON serialization."""
    return asdict(sample)
