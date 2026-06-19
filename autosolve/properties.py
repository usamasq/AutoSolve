# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
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


def _on_python_path_update(self, context):
    self.installer_state = 'IDLE'
    self.installer_progress = ""
    try:
        from .ui import clear_status_cache
        clear_status_cache()
    except Exception:
        pass


def _on_use_external_worker_update(self, context):
    if self.use_external_worker and not self.external_python_path:
        try:
            import bpy
            bpy.ops.autosolve.detect_python('INVOKE_DEFAULT')
        except Exception:
            pass


def _on_tracking_mode_update(self, context):
    if self.tracking_mode == 'AI':
        self.use_external_worker = True
        self.tracking_backend = 'COTRACKER'
        self.masking_backend = 'SAM2'
        self.solving_backend = 'PRECISION'
        if not self.external_python_path:
            try:
                import bpy
                bpy.ops.autosolve.detect_python('INVOKE_DEFAULT')
            except Exception:
                pass
    else:
        self.use_external_worker = False
        self.tracking_backend = 'NATIVE'
        self.masking_backend = 'NONE'
        self.solving_backend = 'NATIVE'


class AutoSolveSettings(PropertyGroup):
    """Main settings for AutoSolve."""

    tracking_mode: EnumProperty(
        name="Tracking Mode",
        description="Choose between Standard native tracking and Local AI-assisted tracking",
        items=[
            ('STANDARD', "Standard", "Fast native KLT tracking and solving. Zero setup required", 'TRACKING', 0),
            ('AI', "AI-Assisted", "Local AI-assisted dense tracking, dynamic masking, and precision solving (runs entirely on your PC)", 'LIGHT', 1),
        ],
        default='STANDARD',
        update=_on_tracking_mode_update,
    )
    
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

    use_external_worker: BoolProperty(
        name="Use Advanced AI Features",
        description="Use deep-learning models for dense tracking (CoTracker v3) and dynamic object masking",
        default=False,
        update=_on_use_external_worker_update,
    )

    external_python_path: StringProperty(
        name="External Python Path",
        description="Path to system Python environment with PyTorch and SciPy (e.g. C:\\Python310\\python.exe)",
        default="",
        subtype='FILE_PATH',
        update=_on_python_path_update,
    )

    installer_state: EnumProperty(
        name="Installer State",
        description="Status of the AI dependencies installer",
        items=[
            ('IDLE', "Idle", "Installer is not active"),
            ('INSTALLING', "Installing", "Dependencies are currently installing"),
            ('SUCCESS', "Success", "Dependencies installed successfully"),
            ('FAILED', "Failed", "Dependency installation failed"),
        ],
        default='IDLE',
        options={'SKIP_SAVE'},
    )

    installer_progress: StringProperty(
        name="Installer Progress",
        description="Status messages from the active package installer",
        default="",
        options={'SKIP_SAVE'},
    )

    tracking_backend: EnumProperty(
        name="Tracking Backend",
        description="Core tracking algorithm",
        items=[
            ('NATIVE', "Blender Native (KLT)", "Use Blender's built-in KLT feature tracker", 'TRACKING', 0),
            ('COTRACKER', "CoTracker v3 (AI)", "Use Meta's CoTracker v3 deep learning model (requires CUDA/MPS)", 'SYSTEM', 1),
        ],
        default='NATIVE',
    )

    masking_backend: EnumProperty(
        name="Dynamic Masking",
        description="Detect and ignore dynamic elements like moving people or vehicles during tracking",
        items=[
            ('NONE', "None", "Do not run dynamic masking", 'X', 0),
            ('SAM2', "YOLO (AI)", "Automatically mask out moving objects using YOLOv8 segmenter (runs locally)", 'MOD_MASK', 1),
        ],
        default='NONE',
    )

    solving_backend: EnumProperty(
        name="Solving Backend",
        description="Solver used for camera reconstruction",
        items=[
            ('NATIVE', "Blender Native", "Use Blender's built-in bundle adjuster", 'RENDER_STILL', 0),
            ('PRECISION', "Precision Solver (AI)", "Use custom multi-pass Levenberg-Marquardt adjuster (requires SciPy)", 'DECORATE_DRIVER', 1),
        ],
        default='NATIVE',
    )
    
    footage_type: EnumProperty(
        name="Footage Type",
        description="Type of footage - optimizes tracking settings for footage characteristics",
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
            ('SCREEN', "Screen Recording", 
             "Screen capture - flat textures, no lens distortion", 
             'WINDOW', 8),
            ('CINEMATIC', "Cinematic", 
             "Cinematic footage - anamorphic, shallow depth of field", 
             'MOVIE', 9),
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
    
    # ═══════════════════════════════════════════════════════════
    # REGION EXCLUSION OPTIONS
    # ═══════════════════════════════════════════════════════════
    
    annotation_mode: EnumProperty(
        name="Annotation Mode",
        description="How to use drawn annotations during feature detection",
        items=[
            ('NONE', "Ignore", 
             "Detect features everywhere (ignore annotations)", 
             'X', 0),
            ('EXCLUDE', "Exclude Region", 
             "Detect features OUTSIDE drawn annotations (skip water, sky, etc.)", 
             'SELECT_SUBTRACT', 1),
            ('INCLUDE', "Include Only", 
             "Detect features ONLY INSIDE drawn annotations", 
             'SELECT_SET', 2),
        ],
        default='NONE',
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
    
    # ═══════════════════════════════════════════════════════════
    # SOLVE REPORT PROPERTIES
    # ═══════════════════════════════════════════════════════════
    
    report_markers_detected: IntProperty(
        name="Markers Detected",
        default=0,
        options={'SKIP_SAVE'},
    )
    
    report_survived_forward: IntProperty(
        name="Survived Forward",
        default=0,
        options={'SKIP_SAVE'},
    )
    
    report_survived_backward: IntProperty(
        name="Survived Backward",
        default=0,
        options={'SKIP_SAVE'},
    )
    
    report_after_cleanup: IntProperty(
        name="Passed Cleanup",
        default=0,
        options={'SKIP_SAVE'},
    )
    
    report_gaps_healed: IntProperty(
        name="Gaps Healed",
        default=0,
        options={'SKIP_SAVE'},
    )
    
    report_bundles: IntProperty(
        name="Report Bundles",
        default=0,
        options={'SKIP_SAVE'},
    )
    
    report_error: FloatProperty(
        name="Report Error",
        default=0.0,
        precision=2,
        options={'SKIP_SAVE'},
    )
    
    report_total_time: FloatProperty(
        name="Report Total Time",
        default=0.0,
        precision=1,
        options={'SKIP_SAVE'},
    )
    
    select_error_threshold: FloatProperty(
        name="Error Threshold",
        description="Select tracks with reprojection error above this threshold (in pixels)",
        default=1.5,
        min=0.1,
        max=10.0,
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
