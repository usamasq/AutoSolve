# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""AutoSolve UI panels - Beginner-friendly phase-based workflow."""

import bpy
from bpy.types import Panel

# Track which clip was last displayed (for detecting clip switches)
_last_displayed_clip_fingerprint = ""


def _get_clip_fingerprint(clip):
    """Get fingerprint for clip identification."""
    if not clip:
        return ""
    try:
        import hashlib
        data = f"{clip.size[0]}x{clip.size[1]}_{clip.fps}_{clip.frame_duration}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]
    except Exception:
        return ""


def _sync_clip_state_if_changed(context):
    """
    Detect clip switch and sync state accordingly.
    
    Called on each panel draw to ensure UI reflects current clip's state.
    """
    global _last_displayed_clip_fingerprint
    
    clip = context.edit_movieclip
    if not clip:
        _last_displayed_clip_fingerprint = ""
        return
    
    current_fingerprint = _get_clip_fingerprint(clip)
    
    if current_fingerprint != _last_displayed_clip_fingerprint:
        # Clip changed! Sync state
        if _last_displayed_clip_fingerprint:
            # There was a previous clip - save its behavior data
            try:
                from .clip_state import get_clip_manager
                manager = get_clip_manager()
                manager.update_from_blender(clip, context.scene)
            except Exception as e:
                print(f"AutoSolve: Error syncing clip state: {e}")
        
        _last_displayed_clip_fingerprint = current_fingerprint


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: Determine current workflow phase
# ═══════════════════════════════════════════════════════════════════════════

def get_workflow_phase(context):
    """
    Determine current phase:
    - 'TRACK': No valid solve yet, or currently solving/retrying
    - 'SCENE_SETUP': Solve valid, no camera with tracking animation yet
    - 'REFINE': Camera with tracking animation exists
    """
    # If currently solving, always show TRACK phase
    settings = context.scene.autosolve
    if settings.is_solving:
        return 'TRACK'
    
    clip = context.edit_movieclip
    if clip is None:
        return 'TRACK'
    
    # Check if solve is valid
    has_solve = clip.tracking.reconstruction.is_valid
    
    # Check if ANY camera in scene has animation (from tracking)
    has_tracking_camera = _find_tracking_camera(context) is not None
    
    if has_tracking_camera:
        return 'REFINE'
    elif has_solve:
        return 'SCENE_SETUP'
    else:
        return 'TRACK'


def _find_tracking_camera(context):
    """
    Find a camera with tracking-derived animation for the CURRENT clip.
    """
    clip = context.edit_movieclip
    if not clip:
        return None
    
    # helper to check object
    def is_tracking_camera(obj):
        if not obj or obj.type != 'CAMERA':
            return False
        
        has_valid_solve = clip.tracking.reconstruction.is_valid
            
        # Check constraints
        for constraint in obj.constraints:
            if constraint.type == 'CAMERA_SOLVER':
                # Explicit clip match
                if hasattr(constraint, 'clip') and constraint.clip == clip:
                    return True
                # Implicit match: constraint.clip is None means "use active clip"
                # Safe only if current clip has a valid solve (prevents false positive on switch)
                if hasattr(constraint, 'clip') and constraint.clip is None and has_valid_solve:
                    return True
        
        # Check baked animation (fallback for constraints that were baked out)
        if has_valid_solve:
            if obj.animation_data and obj.animation_data.action:
                return True
                
        return False

    # 1. Check Active Object (most likely candidate after setup)
    if is_tracking_camera(context.view_layer.objects.active):
        return context.view_layer.objects.active

    # 2. Check Scene Camera
    if is_tracking_camera(context.scene.camera):
        return context.scene.camera
    
    # 3. Search all cameras
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA' and is_tracking_camera(obj):
            return obj
    
    return None


def get_quality_rating(error):
    if error <= 0.0:
        return 0, "No Solve"
    if error < 0.5:
        return 5, "Excellent"
    elif error < 1.0:
        return 4, "Good"
    elif error < 1.5:
        return 3, "Fair"
    elif error < 2.0:
        return 2, "Poor"
    else:
        return 1, "Bad"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════

class AUTOSOLVE_PT_main_panel(Panel):
    """Main AutoSolve panel in the Movie Clip Editor."""
    
    bl_label = "AutoSolve"
    bl_idname = "AUTOSOLVE_PT_main_panel"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'TOOLS'
    bl_category = "AutoSolve"
    bl_order = 0
    
    @classmethod
    def poll(cls, context):
        return context.edit_movieclip is not None
    
    def draw(self, context):
        layout = self.layout
        clip = context.edit_movieclip
        
        # Sync per-clip state when clip changes
        _sync_clip_state_if_changed(context)
        
        # Clip info - compact header with fingerprint indicator
        row = layout.row()
        row.label(text=clip.name, icon='SEQUENCE')
        row.label(text=f"{clip.size[0]}x{clip.size[1]} | {clip.frame_duration}f")



import threading
import time

# Worker status cache
_last_worker_check = 0.0
_cached_port_in_use = False
_cached_ping_status = {"ok": False, "error": "Offline"}
_is_checking_worker = False
_worker_check_thread = None

# Dependency verification cache/thread
_dep_check_thread = None
_dep_check_result = None  # None, "CHECKING", "READY", "MISSING", "SANDBOXED_BLENDER", "INVALID_PATH", "NOT_FUNCTIONAL"
_dep_check_path = None

def check_worker_status_async(force=False):
    global _worker_check_thread, _last_worker_check, _is_checking_worker
    
    now = time.time()
    if not force and (now - _last_worker_check < 3.0):
        return
        
    if _is_checking_worker:
        return
        
    _is_checking_worker = True
    _last_worker_check = now
    
    def target():
        global _cached_port_in_use, _cached_ping_status, _is_checking_worker
        try:
            from .worker.client import is_port_in_use, ping_worker
            port_in_use = is_port_in_use()
            if port_in_use:
                ping_status = ping_worker()
            else:
                ping_status = {"ok": False, "error": "Offline"}
            
            _cached_port_in_use = port_in_use
            _cached_ping_status = ping_status
        except Exception as e:
            _cached_port_in_use = False
            _cached_ping_status = {"ok": False, "error": str(e)}
        finally:
            _is_checking_worker = False
            
    _worker_check_thread = threading.Thread(target=target, daemon=True)
    _worker_check_thread.start()

def check_dependencies_async(python_path):
    global _dep_check_thread, _dep_check_result, _dep_check_path
    if _dep_check_thread and _dep_check_thread.is_alive():
        return
    
    _dep_check_path = python_path
    _dep_check_result = "CHECKING"
    
    def target():
        global _dep_check_result
        try:
            from .worker.client import verify_python_environment
            _dep_check_result = verify_python_environment(python_path)
        except Exception:
            _dep_check_result = "MISSING"
            
    _dep_check_thread = threading.Thread(target=target, daemon=True)
    _dep_check_thread.start()

def clear_status_cache():
    global _last_worker_check, _dep_check_result, _dep_check_path, _cached_port_in_use, _cached_ping_status
    _last_worker_check = 0.0
    _cached_port_in_use = False
    _cached_ping_status = {"ok": False, "error": "Offline"}
    _dep_check_result = None
    _dep_check_path = None


def _draw_tracking_settings(layout, settings, context):
    # Tracking Mode Toggle (Large and Prominent Box)
    toggle_box = layout.box()
    row = toggle_box.row(align=True)
    row.scale_y = 1.5
    row.prop(settings, "tracking_mode", expand=True)
    
    # Core settings
    col = layout.column(align=True)
    row = col.row(align=True)
    row.prop(settings, "quality_preset", text="")
    row.prop(settings, "footage_type", text="")
    
    row = col.row(align=True)
    row.prop(settings, "tripod_mode", toggle=True, text="Tripod")
    row.prop(settings, "robust_mode", toggle=True, text="Robust")
    row.prop(settings, "batch_tracking", toggle=True, text="Batch")
    
    # Inline Local AI Setup card (only shows in AI mode)
    if settings.tracking_mode == 'AI':
        # Trigger non-blocking check
        check_worker_status_async()
        port_in_use = _cached_port_in_use
        
        worker_connected = False
        device_str = "CPU"
        if port_in_use:
            status = _cached_ping_status
            if status.get("ok"):
                worker_connected = True
                device_str = "GPU" if status.get("cuda") else "CPU"
                
        try:
            from .tracker.onnx_predictor import is_onnx_installed
            onnx_installed = is_onnx_installed()
        except Exception:
            onnx_installed = False
            
        ai_box = layout.box()
        ai_box.label(text="Local AI Environment", icon='SYSTEM')
        
        # 1. ONNX Optimizer Engine
        onnx_row = ai_box.row(align=True)
        if onnx_installed:
            onnx_row.label(text="Local Optimizer: Ready", icon='CHECKMARK')
        else:
            onnx_row.label(text="Local Optimizer: Not Installed", icon='ERROR')
            onnx_row.operator("autosolve.install_onnx", text="Install Engine", icon='IMPORT')
            
        # 2. PyTorch Precision/Tracking Service
        py_row = ai_box.row(align=True)
        py_row.prop(settings, "external_python_path", text="Python Path")
        py_row.operator("autosolve.detect_python", text="Auto-Detect", icon='ZOOM_ALL')
        
        state = settings.installer_state
        progress_msg = settings.installer_progress
        
        status_box = ai_box.box()
        if state == 'INSTALLING':
            status_box.label(text="Status: Installing AI Packages...", icon='PREVIEW_LOADING')
            col_p = status_box.column(align=True)
            col_p.label(text=progress_msg or "Starting installation...", icon='INFO')
            col_p.label(text="Please wait - this will not freeze Blender.", icon='INFO')
        elif worker_connected:
            status_box.label(text=f"Status: Connected ({device_str})", icon='CHECKMARK')
            status_box.operator("autosolve.stop_worker", text="Stop Local AI Service", icon='CANCEL')
        else:
            if settings.external_python_path:
                if _dep_check_path != settings.external_python_path or _dep_check_result is None:
                    check_dependencies_async(settings.external_python_path)
                    
                if _dep_check_result == "CHECKING":
                    status_box.label(text="Status: Verifying environment...", icon='PREVIEW_LOADING')
                elif _dep_check_result == "READY":
                    status_box.label(text="Status: Local AI Service Ready", icon='CHECKMARK')
                    status_box.label(text="Will start automatically when tracking is run.", icon='INFO')
                    status_box.operator("autosolve.start_worker", text="Start AI Service Manually", icon='PLAY')
                elif _dep_check_result == "SANDBOXED_BLENDER":
                    warn_box = status_box.box()
                    warn_box.alert = True
                    col_w = warn_box.column(align=True)
                    
                    import sys
                    if sys.platform.startswith('linux'):
                        col_w.label(text="Unsupported Environment (Sandboxed Blender)", icon='ERROR')
                        col_w.separator()
                        col_w.label(text="Why: Blender is running in a Snap or Flatpak sandbox,", icon='INFO')
                        col_w.label(text="     which blocks it from executing external Python processes.")
                        col_w.separator()
                        col_w.label(text="How to fix:")
                        col_w.label(text="1. Uninstall the Snap/Flatpak version of Blender.")
                        col_w.label(text="2. Download the standard portable (.tar.xz) from blender.org.")
                        col_w.label(text="3. Extract and run that standard version of Blender.")
                        col_w.separator()
                        op = col_w.operator("wm.url_open", text="Download Blender (blender.org)", icon='URL')
                        op.url = "https://www.blender.org/download/"
                    else:
                        col_w.label(text="Unsupported Environment (Sandboxed Blender)", icon='ERROR')
                        col_w.separator()
                        col_w.label(text="Why: Blender is sandboxed and cannot launch", icon='INFO')
                        col_w.label(text="     external Python processes.")
                        col_w.separator()
                        col_w.label(text="How to fix:")
                        col_w.label(text="1. Download the standard version of Blender from blender.org.")
                        col_w.label(text="2. Install and run that standard version of Blender.")
                        col_w.separator()
                        op = col_w.operator("wm.url_open", text="Download Blender (blender.org)", icon='URL')
                        op.url = "https://www.blender.org/download/"
                elif _dep_check_result == "INVALID_PATH":
                    status_box.label(text="Status: Python path not found.", icon='ERROR')
                    status_box.separator()
                    help_box = status_box.box()
                    col_h = help_box.column(align=True)
                    col_h.label(text="Please ensure standard Python is installed:", icon='INFO')
                    col_h.label(text="- Standard Python (3.10-3.12) is recommended.")
                    col_h.label(text="- Run the installer and check 'Add python.exe to PATH'.")
                    col_h.separator()
                    op = col_h.operator("wm.url_open", text="Download Python (python.org)", icon='URL')
                    op.url = "https://www.python.org/downloads/"
                elif _dep_check_result == "NOT_FUNCTIONAL":
                    status_box.label(text="Status: Python is not functional.", icon='ERROR')
                    status_box.separator()
                    help_box = status_box.box()
                    col_h = help_box.column(align=True)
                    col_h.label(text="Python installation is broken or restricted.", icon='ERROR')
                    col_h.label(text="Please reinstall standard Python from python.org:")
                    col_h.separator()
                    op = col_h.operator("wm.url_open", text="Download Python (python.org)", icon='URL')
                    op.url = "https://www.python.org/downloads/"
                else: # "MISSING" or None
                    status_box.label(text="Status: AI Packages Missing (approx. 150MB)", icon='QUESTION')
                    status_box.operator("autosolve.install_deps", text="Install AI Packages", icon='IMPORT')
            else:
                status_box.label(text="Status: Python path not specified", icon='QUESTION')
                status_box.label(text="Click 'Auto-Detect' above to locate Python.", icon='INFO')
                status_box.separator()
                
                help_box = status_box.box()
                col_h = help_box.column(align=True)
                col_h.label(text="How to set up Local AI:", icon='HELP')
                col_h.label(text="- Standard Python (3.10-3.12) must be installed on your PC.")
                col_h.label(text="- Click 'Auto-Detect' to scan common paths automatically.")
                col_h.label(text="- If not found, download and install standard Python.")
                col_h.label(text="  (Make sure to check 'Add python.exe to PATH' in installer)")
                col_h.separator()
                op = col_h.operator("wm.url_open", text="Download Python (python.org)", icon='URL')
                op.url = "https://www.python.org/downloads/"
                
        if state == 'FAILED':
            status_box.separator()
            status_box.label(text="Last Installation Failed", icon='ERROR')
            if progress_msg:
                status_box.label(text=f"Info: {progress_msg}")
            status_box.operator("autosolve.install_deps", text="Retry Installation", icon='IMPORT')
            
        # 3. Active Backends
        backends_active = worker_connected or (settings.external_python_path and _dep_check_result == "READY")
        backend_box = ai_box.box()
        backend_box.label(text="Active Backends:", icon='PREFERENCES')
        col_b = backend_box.column(align=True)
        col_b.active = bool(backends_active)
        col_b.prop(settings, "tracking_backend", text="Tracking")
        col_b.prop(settings, "masking_backend", text="Masking")
        col_b.prop(settings, "solving_backend", text="Solving")
        
        if not backends_active:
            info_row = backend_box.row()
            info_row.label(text="Verify Python/packages above to activate backends.", icon='INFO')
            
        # Image sequence warning
        clip = context.edit_movieclip
        if clip and clip.source == 'SEQUENCE':
            if settings.tracking_backend == 'COTRACKER' or settings.masking_backend == 'SAM2':
                warn_box = ai_box.box()
                warn_box.alert = True
                col_warn = warn_box.column(align=True)
                col_warn.label(text="Image Sequence Warning", icon='ERROR')
                col_warn.label(text="AI tracking/masking backends only support movie files.")
                col_warn.label(text="Please use Native tracking or transcode clip.")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: TRACKING
# ═══════════════════════════════════════════════════════════════════════════

class AUTOSOLVE_PT_phase1_tracking(Panel):
    """Phase 1: Tracking configuration and execution."""
    
    bl_label = "Step 1: Track"
    bl_idname = "AUTOSOLVE_PT_phase1_tracking"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'TOOLS'
    bl_category = "AutoSolve"
    bl_parent_id = "AUTOSOLVE_PT_main_panel"
    
    def draw_header(self, context):
        phase = get_workflow_phase(context)
        if phase in ('SCENE_SETUP', 'REFINE'):
            self.layout.label(text="", icon='CHECKMARK')
        elif phase == 'TRACK':
            self.layout.label(text="", icon='FORWARD')
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.autosolve
        phase = get_workflow_phase(context)
        
        # Single outer box for the entire phase
        outer_box = layout.box()
        
        if phase == 'TRACK':
            # ══════════════════════════════════════════════
            # ACTIVE PHASE
            # ══════════════════════════════════════════════
            clip = context.edit_movieclip
            has_tracks = len(clip.tracking.tracks) > 0
            has_solve = clip.tracking.reconstruction.is_valid
            
            # Main action
            if settings.is_solving:
                col = outer_box.column(align=True)
                col.label(text=settings.solve_status, icon='TIME')
                col.prop(settings, "solve_progress", text="")
                col.label(text="Press ESC to cancel", icon='CANCEL')
            elif has_tracks and not has_solve:
                # User has manually added tracks - offer quick solve
                outer_box.label(text="Tracks detected! Ready to solve.", icon='CHECKMARK')
                outer_box.separator()
                
                row = outer_box.row()
                row.scale_y = 2.0
                op = row.operator("clip.solve_camera", text="Solve Camera", icon='PLAY')
                
                outer_box.separator()
                outer_box.label(text="Or start fresh:", icon='INFO')
                row = outer_box.row()
                row.operator("autosolve.run_solve", text="Auto-Track", icon='TRACKING')
            else:
                # Fresh clip - full auto workflow
                outer_box.label(text="One-click camera tracking", icon='LIGHT')
                outer_box.separator()
                
                row = outer_box.row()
                row.scale_y = 2.0
                row.operator("autosolve.run_solve", text="Auto-Track & Solve", icon='PLAY')
            
            outer_box.separator()
            
            # Settings
            outer_box.label(text="Settings:", icon='PREFERENCES')
            _draw_tracking_settings(outer_box, settings, context)
        
        else:
            # ══════════════════════════════════════════════
            # COMPLETED
            # ══════════════════════════════════════════════
            
            clip = context.edit_movieclip
            recon = clip.tracking.reconstruction
            num_tracks = len([t for t in clip.tracking.tracks if t.has_bundle])
            
            # Summary
            row = outer_box.row()
            row.label(text=f"{num_tracks} tracks solved", icon='CHECKMARK')
            row.label(text=f"{recon.average_error:.2f}px")
            
            outer_box.separator()
            
            # Redo section
            outer_box.label(text="Not satisfied?", icon='LOOP_BACK')
            _draw_tracking_settings(outer_box, settings, context)
            
            row = outer_box.row(align=True)
            row.scale_y = 1.4
            row.operator("clip.solve_camera", text="Solve Camera", icon='PLAY')
            row.operator("autosolve.run_solve", text="Re-Track", icon='FILE_REFRESH')


class AUTOSOLVE_PT_region_tools(Panel):
    """Region exclusion tools for problematic areas."""
    
    bl_label = "Region Tools"
    bl_idname = "AUTOSOLVE_PT_region_tools"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'TOOLS'
    bl_category = "AutoSolve"
    bl_parent_id = "AUTOSOLVE_PT_phase1_tracking"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        return context.edit_movieclip is not None
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.autosolve
        
        outer_box = layout.box()
        
        # Help text
        col = outer_box.column(align=True)
        col.scale_y = 0.8
        col.label(text="Highlight focus areas", icon='INFO')
        
        outer_box.separator()
        
        # Annotation mode toggle (the key setting)
        outer_box.label(text="Annotation Mode:")
        outer_box.prop(settings, "annotation_mode", text="")
        
        outer_box.separator()
        
        # Draw annotation button (pencil mode)
        row = outer_box.row(align=True)
        row.operator("gpencil.annotate", text="Draw", icon='GREASEPENCIL').mode = 'DRAW'
        row.operator("gpencil.annotate", text="Eraser", icon='PANEL_CLOSE').mode = 'ERASER'
        row.operator("autosolve.clear_annotations", text="Clear", icon='X')


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: SCENE SETUP
# ═══════════════════════════════════════════════════════════════════════════

class AUTOSOLVE_PT_phase2_scene(Panel):
    """Phase 2: Scene setup after successful solve."""
    
    bl_label = "Step 2: Setup Scene"
    bl_idname = "AUTOSOLVE_PT_phase2_scene"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'TOOLS'
    bl_category = "AutoSolve"
    bl_parent_id = "AUTOSOLVE_PT_main_panel"
    
    @classmethod
    def poll(cls, context):
        if context.edit_movieclip is None:
            return False
        return context.edit_movieclip.tracking.reconstruction.is_valid
    
    def draw_header(self, context):
        phase = get_workflow_phase(context)
        if phase == 'REFINE':
            self.layout.label(text="", icon='CHECKMARK')
        elif phase == 'SCENE_SETUP':
            self.layout.label(text="", icon='FORWARD')
    
    def draw(self, context):
        layout = self.layout
        clip = context.edit_movieclip
        recon = clip.tracking.reconstruction
        settings = context.scene.autosolve
        phase = get_workflow_phase(context)
        
        # Single outer box for the entire phase
        outer_box = layout.box()
        
        # Solve Report Box
        if settings.has_solve:
            report_box = outer_box.box()
            report_box.label(text="Solve Summary Report", icon='TEXT')
            
            row = report_box.row()
            col1 = row.column()
            col1.label(text="Detection:")
            col1.label(text="Tracking (Fwd):")
            col1.label(text="Tracking (Bwd):")
            col1.label(text="Cleanup:")
            col1.label(text="Healing:")
            col1.label(text="Solve:")
            col1.label(text="Quality:")
            col1.label(text="Time:")
            
            col2 = row.column()
            col2.label(text=f"{settings.report_markers_detected} markers")
            col2.label(text=f"{settings.report_survived_forward} tracks")
            col2.label(text=f"{settings.report_survived_backward} tracks")
            col2.label(text=f"{settings.report_after_cleanup} tracks")
            col2.label(text=f"{settings.report_gaps_healed} gaps healed")
            col2.label(text=f"{settings.report_bundles} bundles @ {settings.report_error:.2f}px")
            
            stars, rating_text = get_quality_rating(settings.report_error)
            stars_str = "★" * stars + "☆" * (5 - stars)
            col2.label(text=f"{stars_str} ({rating_text})")
            col2.label(text=f"{settings.report_total_time:.1f}s")
            
            outer_box.separator()
        
        if phase == 'SCENE_SETUP':
            # ══════════════════════════════════════════════
            # ACTIVE PHASE
            # ══════════════════════════════════════════════
            
            # Quality indicator
            error = recon.average_error
            if error < 0.5:
                outer_box.label(text="Excellent quality!", icon='CHECKMARK')
            elif error < 1.0:
                outer_box.label(text="Good quality", icon='INFO')
            else:
                row = outer_box.row()
                row.alert = True
                row.label(text="High error – consider re-tracking", icon='ERROR')
            
            outer_box.separator()
            
            # Guidance
            outer_box.label(text="Set up your 3D scene", icon='LIGHT')
            
            outer_box.separator()
            
            # Action button
            row = outer_box.row()
            row.scale_y = 2.0
            row.operator("autosolve.setup_scene", text="Create 3D Scene", icon='SCENE_DATA')
        
        else:
            # ══════════════════════════════════════════════
            # COMPLETED
            # ══════════════════════════════════════════════
            
            outer_box.label(text="Scene ready!", icon='CHECKMARK')
            
            outer_box.separator()
            
            # Redo
            outer_box.label(text="Need changes?", icon='LOOP_BACK')
            row = outer_box.row()
            row.operator("autosolve.setup_scene", text="Redo Scene Setup", icon='FILE_REFRESH')


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: REFINE
# ═══════════════════════════════════════════════════════════════════════════

class AUTOSOLVE_PT_phase3_refine(Panel):
    """Phase 3: Refinement tools after scene setup."""
    
    bl_label = "Step 3: Refine"
    bl_idname = "AUTOSOLVE_PT_phase3_refine"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'TOOLS'
    bl_category = "AutoSolve"
    bl_parent_id = "AUTOSOLVE_PT_main_panel"
    
    @classmethod
    def poll(cls, context):
        # Show if ANY camera with tracking animation exists
        return _find_tracking_camera(context) is not None
    
    def draw_header(self, context):
        self.layout.label(text="", icon='FORWARD')
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.autosolve
        
        # Single outer box for the entire phase
        outer_box = layout.box()
        
        # Guidance
        outer_box.label(text="Polish your results", icon='LIGHT')
        
        outer_box.separator()
        
        # Smoothing tools
        outer_box.label(text="Reduce Jitter:", icon='MOD_SMOOTH')
        
        # Strength slider
        col = outer_box.column(align=True)
        col.prop(settings, "track_smooth_factor", text="Smoothing", slider=True)
        
        # Button
        row = outer_box.row(align=True)
        row.scale_y = 1.4
        row.operator("autosolve.smooth_tracks", text="Apply Smoothing", icon='CURVE_PATH')
        
        outer_box.separator()
        
        # Clean Bad Tracks Section
        outer_box.label(text="Clean Bad Tracks:", icon='TRACKING_BACKWARDS')
        col = outer_box.column(align=True)
        col.prop(settings, "select_error_threshold", text="Threshold", slider=True)
        
        row = outer_box.row(align=True)
        row.scale_y = 1.4
        row.operator("autosolve.select_high_error", text="Select High Error", icon='RESTRICT_SELECT_OFF')
        row.operator("autosolve.resolve", text="Clean & Re-Solve", icon='FILE_REFRESH')
        
        outer_box.separator()
        
        # Restart options
        outer_box.label(text="Need to restart?", icon='LOOP_BACK')
        row = outer_box.row(align=True)
        row.operator("autosolve.run_solve", text="Re-Track", icon='TRACKING')
        row.operator("autosolve.setup_scene", text="Redo Scene", icon='SCENE_DATA')
        
        


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════

classes = (
    AUTOSOLVE_PT_main_panel,
    AUTOSOLVE_PT_phase1_tracking,
    AUTOSOLVE_PT_region_tools,
    AUTOSOLVE_PT_phase2_scene,
    AUTOSOLVE_PT_phase3_refine,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)



