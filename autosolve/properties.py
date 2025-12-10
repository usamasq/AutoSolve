# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve properties module.

Defines all PropertyGroups for storing settings and state.
"""

import bpy
from bpy.props import (
    BoolProperty,
    IntProperty,
    FloatProperty,
    StringProperty,
    PointerProperty,
    EnumProperty,
)
from bpy.types import PropertyGroup


class AutoSolveSettings(PropertyGroup):
    """Main settings for AutoSolve."""
    
    # ═══════════════════════════════════════════════════════════
    # SOLVER OPTIONS
    # ═══════════════════════════════════════════════════════════
    
    quality_preset: EnumProperty(
        name="Quality",
        description="Balance between speed and accuracy",
        items=[
            ('FAST', "Fast", 
             "Quick solve - fewer features, faster tracking", 
             'PLAY', 0),
            ('BALANCED', "Balanced", 
             "Good quality for most footage", 
             'DECORATE_DRIVER', 1),
            ('QUALITY', "Quality", 
             "Best accuracy - more features, stricter filtering", 
             'RENDER_STILL', 2),
        ],
        default='BALANCED',
    )
    
    tripod_mode: BoolProperty(
        name="Tripod Mode",
        description="Use rotation-only solve for nodal pan/tilt shots. "
                    "Enable this if the camera stayed on a tripod",
        default=False,
    )
    
    robust_mode: BoolProperty(
        name="Robust Mode",
        description="For difficult footage (blur, fast motion, low contrast). "
                    "Uses larger search areas and more forgiving thresholds",
        default=False,
    )
    
    footage_type: EnumProperty(
        name="Footage Type",
        description="Type of footage - helps the tracker learn better defaults",
        items=[
            ('AUTO', "Auto-detect", 
             "Automatically determine footage characteristics", 
             'AUTO', 0),
            ('INDOOR', "Indoor", 
             "Indoor scenes with controlled lighting", 
             'HOME', 1),
            ('OUTDOOR', "Outdoor", 
             "Outdoor scenes with natural lighting", 
             'WORLD', 2),
            ('DRONE', "Drone/Aerial", 
             "Aerial footage with parallax and sky", 
             'TRACKING', 3),
            ('HANDHELD', "Handheld", 
             "Handheld camera with shake", 
             'VIEW_PAN', 4),
            ('GIMBAL', "Gimbal/Stabilized", 
             "Smooth stabilized footage", 
             'ORIENTATION_GIMBAL', 5),
            ('ACTION', "Action/Fast", 
             "Fast action with motion blur", 
             'FORCE_TURBULENCE', 6),
            ('VFX', "VFX Plate", 
             "Footage shot specifically for VFX integration", 
             'CAMERA_DATA', 7),
        ],
        default='AUTO',
    )
    
    batch_tracking: BoolProperty(
        name="Batch Tracking (Faster)",
        description="Track all frames at once instead of frame-by-frame. "
                    "Faster but no progress feedback during tracking",
        default=False,
    )
    
    # ═══════════════════════════════════════════════════════════
    # SMOOTHING OPTIONS
    # ═══════════════════════════════════════════════════════════
    
    smooth_tracks: BoolProperty(
        name="Smooth Tracks",
        description="Apply smoothing to track markers before solving. "
                    "Reduces jitter but may reduce accuracy on sharp movements",
        default=False,
    )
    
    track_smooth_factor: FloatProperty(
        name="Track Smoothing",
        description="Strength of track smoothing (0=none, 1=heavy)",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )
    
    smooth_camera: BoolProperty(
        name="Smooth Camera",
        description="Apply Butterworth filter to camera motion after solving. "
                    "Removes high-frequency jitter while preserving overall motion",
        default=False,
    )
    
    camera_smooth_factor: FloatProperty(
        name="Camera Smoothing",
        description="Smoothing strength (0.1=subtle, 1.0=heavy smoothing)",
        default=0.5,
        min=0.1,
        max=1.0,
        subtype='FACTOR',
    )
    
    # ═══════════════════════════════════════════════════════════
    # TRAINING DATA OPTIONS
    # ═══════════════════════════════════════════════════════════
    
    record_edits: BoolProperty(
        name="Learn from My Edits",
        description="AutoSolve learns from your corrections! When you delete bad tracks "
                    "or adjust settings before re-solving, AutoSolve learns what works "
                    "for your footage types. All data stays local and anonymous. "
                    "Disable if you prefer not to contribute to the learning model",
        default=True,
    )
    
    # ═══════════════════════════════════════════════════════════
    # SOLVER STATE (runtime, not saved)
    # ═══════════════════════════════════════════════════════════
    
    is_solving: BoolProperty(
        name="Is Solving",
        description="True while solve is in progress",
        default=False,
        options={'SKIP_SAVE'},
    )
    
    solve_progress: FloatProperty(
        name="Progress",
        description="Current solve progress (0.0 to 1.0)",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        options={'SKIP_SAVE'},
    )
    
    solve_status: StringProperty(
        name="Status",
        description="Current solve status message",
        default="",
        options={'SKIP_SAVE'},
    )
    
    # ═══════════════════════════════════════════════════════════
    # RESULT STATE
    # ═══════════════════════════════════════════════════════════
    
    has_solve: BoolProperty(
        name="Has Solve",
        description="True if a successful solve exists",
        default=False,
        options={'SKIP_SAVE'},
    )
    
    solve_error: FloatProperty(
        name="Solve Error",
        description="Average reprojection error in pixels",
        default=0.0,
        min=0.0,
        precision=2,
        options={'SKIP_SAVE'},
    )
    
    point_count: IntProperty(
        name="Point Count",
        description="Number of tracked points with 3D positions",
        default=0,
        min=0,
        options={'SKIP_SAVE'},
    )


# ═══════════════════════════════════════════════════════════════
# REGISTRATION
# ═══════════════════════════════════════════════════════════════

classes = (
    AutoSolveSettings,
)


def register():
    """Register property classes and attach to Scene."""
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.autosolve = PointerProperty(type=AutoSolveSettings)


def unregister():
    """Unregister property classes and remove from Scene."""
    del bpy.types.Scene.autosolve
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
