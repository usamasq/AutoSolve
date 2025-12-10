# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Track and camera smoothing utilities for reducing jitter.

Provides:
- Pre-solve track marker smoothing (Gaussian/moving average)
- Post-solve camera F-curve smoothing (Butterworth filter via Blender API)
"""

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
            
            for j in range(-half_window, half_window + 1):
                neighbor = markers_sorted[i + j]
                # Gaussian weight: closer = higher weight
                weight = _gaussian_weight(j, half_window)
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
    import math
    return math.exp(-(distance ** 2) / (2 * sigma ** 2))


# ═══════════════════════════════════════════════════════════════════════════
# CAMERA F-CURVE SMOOTHING (Post-solve)
# ═══════════════════════════════════════════════════════════════════════════

def smooth_camera_fcurves(camera_object, cutoff_freq: float = 0.5) -> bool:
    """
    Apply Butterworth smoothing to camera animation curves.
    
    Uses Blender's built-in bpy.ops.graph.butterworth_smooth operator
    to reduce high-frequency jitter while preserving overall motion.
    
    Args:
        camera_object: Camera object with animation data
        cutoff_freq: Cutoff frequency (0.1-2.0, lower = smoother)
        
    Returns:
        True if smoothing was applied successfully
    """
    if not camera_object or not camera_object.animation_data:
        print("AutoSolve: No camera animation data to smooth")
        return False
    
    action = camera_object.animation_data.action
    if not action:
        print("AutoSolve: Camera has no action to smooth")
        return False
    
    # Get all location and rotation fcurves
    fcurves_to_smooth = []
    for fcurve in action.fcurves:
        if fcurve.data_path in ('location', 'rotation_euler', 'rotation_quaternion'):
            fcurves_to_smooth.append(fcurve)
    
    if not fcurves_to_smooth:
        print("AutoSolve: No location/rotation curves found on camera")
        return False
    
    # Try using Butterworth smooth (Blender 3.4+)
    success = _apply_butterworth_smooth(fcurves_to_smooth, cutoff_freq)
    
    if not success:
        # Fallback to manual smoothing for older Blender versions
        success = _apply_manual_smooth(fcurves_to_smooth, cutoff_freq)
    
    if success:
        print(f"AutoSolve: Smoothed camera curves (cutoff={cutoff_freq:.2f})")
    
    return success


def _apply_butterworth_smooth(fcurves: List, cutoff_freq: float) -> bool:
    """
    Apply Butterworth smooth using Blender's operator.
    
    Available in Blender 3.4+
    """
    try:
        # Need to select the fcurves in a graph editor context
        # First, deselect all keyframes, then select only our curves
        for fc in fcurves:
            fc.select = True
            for kp in fc.keyframe_points:
                kp.select_control_point = True
        
        # Get context override for graph editor
        override = _get_graph_editor_context()
        if not override:
            return False
        
        # Apply Butterworth smooth
        with bpy.context.temp_override(**override):
            bpy.ops.graph.butterworth_smooth(
                cutoff_frequency=cutoff_freq,
                filter_order=2,
                samples_per_frame=1,
                blend=1.0
            )
        
        return True
        
    except AttributeError:
        # butterworth_smooth not available (older Blender)
        return False
    except Exception as e:
        print(f"AutoSolve: Butterworth smooth failed: {e}")
        return False


def _apply_manual_smooth(fcurves: List, strength: float) -> bool:
    """
    Fallback manual smoothing for older Blender versions.
    
    Uses simple moving average on keyframe values.
    """
    window = max(3, int(3 + strength * 4))
    half = window // 2
    
    for fcurve in fcurves:
        keyframes = fcurve.keyframe_points
        if len(keyframes) < window:
            continue
        
        # Store original values
        original = [(kp.co.x, kp.co.y) for kp in keyframes]
        
        # Apply smoothing
        for i in range(half, len(keyframes) - half):
            avg_value = sum(original[j][1] for j in range(i - half, i + half + 1)) / window
            # Blend based on strength
            keyframes[i].co.y = original[i][1] + (avg_value - original[i][1]) * min(strength, 1.0)
        
        fcurve.update()
    
    return True


def _get_graph_editor_context() -> Optional[dict]:
    """Get context override for graph editor operations."""
    # Try to find a graph editor area
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'GRAPH_EDITOR':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        return {
                            'window': window,
                            'screen': window.screen,
                            'area': area,
                            'region': region,
                        }
    
    # No graph editor open - try to use dopesheet/any editor
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in ('DOPESHEET_EDITOR', 'NLA_EDITOR'):
                for region in area.regions:
                    if region.type == 'WINDOW':
                        # Temporarily switch to graph editor
                        old_type = area.type
                        area.type = 'GRAPH_EDITOR'
                        ctx = {
                            'window': window,
                            'screen': window.screen,
                            'area': area,
                            'region': region,
                        }
                        # Note: caller should restore if needed
                        return ctx
    
    return None


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_solved_camera() -> Optional[bpy.types.Object]:
    """Find the camera created by motion tracking solve."""
    scene = bpy.context.scene
    
    # Check scene camera first
    if scene.camera and scene.camera.animation_data:
        return scene.camera
    
    # Look for cameras with animation
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA' and obj.animation_data and obj.animation_data.action:
            # Check if it has tracking-style animation
            action = obj.animation_data.action
            for fc in action.fcurves:
                if fc.data_path == 'location':
                    return obj
    
    return None


def smooth_solved_camera(strength: float = 0.5) -> bool:
    """
    Convenience function to smooth the solved camera.
    
    Finds the solved camera and applies smoothing.
    
    Args:
        strength: Smoothing strength (0.1-2.0, maps to cutoff frequency)
        
    Returns:
        True if smoothing was applied
    """
    camera = get_solved_camera()
    if not camera:
        print("AutoSolve: Could not find solved camera to smooth")
        return False
    
    # Convert strength to cutoff frequency
    # Higher strength = lower cutoff = smoother
    cutoff = 2.0 - (strength * 1.5)  # strength 0->2.0, 1->0.5
    cutoff = max(0.1, min(2.0, cutoff))
    
    return smooth_camera_fcurves(camera, cutoff)
