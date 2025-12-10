# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Shared utility functions for AutoSolve tracking system.
"""

from pathlib import Path
from typing import Tuple, List, Optional
from mathutils import Vector
import bpy

from .constants import REGIONS, EDGE_REGIONS


# ═══════════════════════════════════════════════════════════════════════════
# DATA DIRECTORY PATHS
# ═══════════════════════════════════════════════════════════════════════════

def get_autosolve_data_dir() -> Path:
    """Get the base AutoSolve data directory."""
    return Path(bpy.utils.user_resource('DATAFILES')) / 'autosolve'


def get_sessions_dir() -> Path:
    """Get the sessions data directory."""
    return get_autosolve_data_dir() / 'sessions'


def get_behavior_dir() -> Path:
    """Get the behavior data directory."""
    return get_autosolve_data_dir() / 'behavior'


def get_cache_dir() -> Path:
    """Get the cache directory (under SCRIPTS for performance)."""
    return Path(bpy.utils.user_resource('SCRIPTS')) / 'autosolve' / 'cache'


def get_model_path() -> Path:
    """Get path to the learned model file."""
    return get_autosolve_data_dir() / 'model.json'


# ═══════════════════════════════════════════════════════════════════════════
# REGION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def get_region(x: float, y: float) -> str:
    """
    Get region name from normalized coordinates (0-1).
    
    Divides the frame into a 3x3 grid and returns the region name.
    
    Args:
        x: Normalized X coordinate (0-1)
        y: Normalized Y coordinate (0-1)
        
    Returns:
        Region name like 'center', 'top-left', etc.
    """
    col = 0 if x < 0.33 else (1 if x < 0.66 else 2)
    row = 2 if y < 0.33 else (1 if y < 0.66 else 0)
    
    region_map = [
        ['top-left', 'top-center', 'top-right'],
        ['mid-left', 'center', 'mid-right'],
        ['bottom-left', 'bottom-center', 'bottom-right']
    ]
    return region_map[row][col]


def get_region_bounds(region: str) -> Tuple[float, float, float, float]:
    """
    Get bounding box for a region.
    
    Args:
        region: Region name
        
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in normalized coordinates
    """
    bounds = {
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
    return bounds.get(region, (0.0, 0.0, 1.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# FOOTAGE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def classify_footage(clip) -> str:
    """
    Classify footage by resolution and fps.
    
    Returns a string key like "HD_30fps" or "4K_24fps".
    
    Args:
        clip: Blender MovieClip object
        
    Returns:
        Classification string
    """
    width = clip.size[0]
    fps = clip.fps if clip.fps > 0 else 24
    
    # Resolution class
    if width >= 3840:
        res = '4K'
    elif width >= 1920:
        res = 'HD'
    else:
        res = 'SD'
    
    # FPS class
    if fps >= 50:
        fps_class = '60fps'
    elif fps >= 28:
        fps_class = '30fps'
    else:
        fps_class = '24fps'
    
    return f"{res}_{fps_class}"


# ═══════════════════════════════════════════════════════════════════════════
# TRACK ANALYSIS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def calculate_jitter(markers) -> float:
    """
    Calculate jitter score (variance in velocity).
    
    High jitter indicates an unstable/noisy track.
    
    Args:
        markers: List of marker objects with .co attribute
        
    Returns:
        Jitter score (0 = stable, higher = more jittery)
    """
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


def get_average_position(markers) -> Tuple[float, float]:
    """
    Calculate average position of markers.
    
    Args:
        markers: List of marker objects with .co attribute
        
    Returns:
        Tuple of (avg_x, avg_y) in normalized coordinates
    """
    if not markers:
        return 0.0, 0.0
    avg_x = sum(m.co.x for m in markers) / len(markers)
    avg_y = sum(m.co.y for m in markers) / len(markers)
    return avg_x, avg_y


def get_sorted_markers(markers) -> List:
    """
    Sort markers by frame number.
    
    Args:
        markers: List of marker objects with .frame attribute
        
    Returns:
        Sorted list of markers
    """
    return sorted(markers, key=lambda m: m.frame)


def calculate_lifespan(markers) -> int:
    """
    Calculate track lifespan in frames.
    
    Args:
        markers: List of marker objects
        
    Returns:
        Number of frames the track spans
    """
    if len(markers) < 2:
        return 0
    sorted_markers = get_sorted_markers(markers)
    return sorted_markers[-1].frame - sorted_markers[0].frame


def infer_deletion_reason(track_data: dict) -> str:
    """
    Infer why a track was deleted based on its characteristics.
    
    Args:
        track_data: Dict with track metrics (lifespan, region, error, jitter_score)
        
    Returns:
        Reason string for deletion
    """
    if track_data.get('lifespan', 0) < 10:
        return "short_lifespan"
    
    if track_data.get('region') in EDGE_REGIONS:
        return "edge_region"
    
    if track_data.get('error', 0) > 2.0:
        return "high_error"
    
    if track_data.get('jitter_score', 0) > 0.5:
        return "jittery"
    
    return "user_manual"


def calculate_track_velocity(markers) -> float:
    """
    Calculate average velocity of a track.
    
    Args:
        markers: List of marker objects
        
    Returns:
        Average velocity in normalized units per frame
    """
    if len(markers) < 2:
        return 0.0
    
    sorted_markers = get_sorted_markers(markers)
    total_displacement = 0.0
    
    for i in range(1, len(sorted_markers)):
        dx = sorted_markers[i].co.x - sorted_markers[i-1].co.x
        dy = sorted_markers[i].co.y - sorted_markers[i-1].co.y
        total_displacement += (dx**2 + dy**2) ** 0.5
    
    return total_displacement / len(sorted_markers)
