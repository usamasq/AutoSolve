# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Track marker smoothing utilities.

Reduces jitter in marker positions before running the camera solve.
Uses Gaussian weighting to smooth track marker coordinates across frames.
"""
import math
import bpy
from typing import List, Tuple, Optional
from mathutils import Vector


# ═══════════════════════════════════════════════════════════════════════════
# TRACK MARKER SMOOTHING (Pre-solve)
# ═══════════════════════════════════════════════════════════════════════════

def smooth_track_markers(tracking, strength: float = 0.5) -> int:
    """
    Apply Gaussian-style smoothing to all track marker positions.
    
    Uses a weighted moving average that preserves track start/end positions.
    Higher strength = larger window = smoother but less accurate for fast motion.
    
    Args:
        tracking: bpy.types.MovieTracking object
        strength: Smoothing strength 0-1 (0=none, 1=heavy smoothing)
        
    Returns:
        Number of markers that were smoothed
    """
    if strength <= 0:
        return 0
    
    # Window size scales with strength: 0.5 -> 3, 1.0 -> 7
    window_size = max(3, int(3 + strength * 4))
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window for symmetric smoothing
    
    half_window = window_size // 2
    markers_smoothed = 0

    # Precompute Gaussian weights for the window to avoid repeated calculations
    # in the inner loop.
    gaussian_weights = [
        _gaussian_weight(j, half_window)
        for j in range(-half_window, half_window + 1)
    ]
    
    for track in tracking.tracks:
        markers = [m for m in track.markers if not m.mute]
        if len(markers) < window_size:
            continue
        
        # Sort markers by frame
        markers_sorted = sorted(markers, key=lambda m: m.frame)
        
        # Compute smoothed positions, but don't modify yet
        smoothed_positions = []
        
        for i, marker in enumerate(markers_sorted):
            # Don't smooth first/last few markers (preserve endpoints)
            if i < half_window or i >= len(markers_sorted) - half_window:
                smoothed_positions.append((marker.frame, marker.co.x, marker.co.y))
                continue
            
            # Gaussian-weighted average of surrounding markers
            total_weight = 0.0
            weighted_x = 0.0
            weighted_y = 0.0
            
            for offset_idx, j in enumerate(range(-half_window, half_window + 1)):
                neighbor = markers_sorted[i + j]
                # Gaussian weight: closer = higher weight
                weight = gaussian_weights[offset_idx]
                weighted_x += neighbor.co.x * weight
                weighted_y += neighbor.co.y * weight
                total_weight += weight
            
            if total_weight > 0:
                new_x = weighted_x / total_weight
                new_y = weighted_y / total_weight
                smoothed_positions.append((marker.frame, new_x, new_y))
                markers_smoothed += 1
            else:
                smoothed_positions.append((marker.frame, marker.co.x, marker.co.y))
        
        # Apply smoothed positions
        for frame, new_x, new_y in smoothed_positions:
            marker = track.markers.find_frame(frame)
            if marker and not marker.mute:
                # Blend between original and smoothed based on strength
                orig_x, orig_y = marker.co.x, marker.co.y
                marker.co.x = orig_x + (new_x - orig_x) * strength
                marker.co.y = orig_y + (new_y - orig_y) * strength
    
    if markers_smoothed > 0:
        print(f"AutoSolve: Smoothed {markers_smoothed} markers (strength={strength:.2f}, window={window_size})")
    
    return markers_smoothed


def _gaussian_weight(distance: int, sigma: int) -> float:
    """Calculate Gaussian weight for a given distance from center."""
    return math.exp(-(distance ** 2) / (2 * sigma ** 2))



