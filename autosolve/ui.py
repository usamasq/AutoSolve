# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
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
    except:
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
    
    Important: Must verify the camera is actually tracking this specific clip,
    not just any camera with animation (could be from a different clip).
    """
    clip = context.edit_movieclip
    if not clip:
        return None
    
    # Check scene camera first - most common case after setup_tracking_scene
    cam = context.scene.camera
    if cam:
        # Check for Camera Solver constraint pointing to THIS clip
        for constraint in cam.constraints:
            if constraint.type == 'CAMERA_SOLVER':
                if hasattr(constraint, 'clip') and constraint.clip == clip:
                    return cam
        
        # Check for animation data that might be baked from THIS clip's tracking
        # Note: Baked animation loses the clip reference, so we need to check
        # if the clip has a valid solve AND camera has animation
        if clip.tracking.reconstruction.is_valid:
            has_animation = cam.animation_data and cam.animation_data.action
            if has_animation:
                # Camera has animation and clip has valid solve - likely from this clip
                # This is a heuristic; ideally we'd store metadata about which clip
                return cam
    
    # Fallback: search all cameras for Camera Solver constraint pointing to THIS clip
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            for constraint in obj.constraints:
                if constraint.type == 'CAMERA_SOLVER':
                    if hasattr(constraint, 'clip') and constraint.clip == clip:
                        return obj
    
    return None


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
        row.label(text=f"{clip.frame_duration}f")


# ═══════════════════════════════════════════════════════════════════════════
# RESEARCH DATA (Always on top, always collapsed)
# ═══════════════════════════════════════════════════════════════════════════

class AUTOSOLVE_PT_research_panel(Panel):
    """Research data management - beta participation."""
    
    bl_label = "Research Beta"
    bl_idname = "AUTOSOLVE_PT_research_panel"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'TOOLS'
    bl_category = "AutoSolve"
    bl_parent_id = "AUTOSOLVE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw_header(self, context):
        self.layout.label(text="", icon='EXPERIMENTAL')
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.autosolve
        
        col = layout.column(align=True)
        col.prop(settings, "record_edits", text="Contribute to Research", icon='REC')
        
        try:
            from .tracker.learning.settings_predictor import SettingsPredictor
            predictor = SettingsPredictor()
            stats = predictor.get_stats()
            col.label(text=f"Sessions: {stats.get('total_sessions', 0)} | Success: {stats.get('success_rate', 0):.0%}")
        except Exception:
            col.label(text="No data collected yet")
        
        row = layout.row(align=True)
        row.operator("autosolve.export_training_data", text="Export", icon='EXPORT')
        row.operator("autosolve.contribute_data", text="Share", icon='URL')
        row.operator("autosolve.reset_training_data", text="Reset", icon='LOOP_BACK')


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
            
            # Guidance
            outer_box.label(text="Click to auto-track your footage", icon='LIGHT')
            
            outer_box.separator()
            
            # Main action
            if settings.is_solving:
                col = outer_box.column(align=True)
                col.label(text=settings.solve_status, icon='TIME')
                col.prop(settings, "solve_progress", text="")
                col.label(text="Press ESC to cancel", icon='CANCEL')
            else:
                row = outer_box.row()
                row.scale_y = 2.0
                row.operator("autosolve.run_solve", text="Analyze & Solve", icon='PLAY')
            
            outer_box.separator()
            
            # Settings
            outer_box.label(text="Options:", icon='PREFERENCES')
            col = outer_box.column(align=True)
            row = col.row(align=True)
            row.prop(settings, "quality_preset", text="")
            row.prop(settings, "footage_type", text="")
            
            row = col.row(align=True)
            row.prop(settings, "tripod_mode", toggle=True)
            row.prop(settings, "robust_mode", toggle=True)
        
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
            
            # Redo
            outer_box.label(text="Want to try again?", icon='LOOP_BACK')
            col = outer_box.column(align=True)
            row = col.row(align=True)
            row.prop(settings, "quality_preset", text="")
            row.prop(settings, "footage_type", text="")
            
            row = outer_box.row()
            row.operator("autosolve.run_solve", text="Re-track", icon='FILE_REFRESH')


class AUTOSOLVE_PT_phase1_advanced(Panel):
    """Advanced tracking options."""
    
    bl_label = "Advanced Options"
    bl_idname = "AUTOSOLVE_PT_phase1_advanced"
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
        
        # Single outer box
        outer_box = layout.box()
        col = outer_box.column(align=True)
        col.prop(settings, "batch_tracking")
        col.prop(settings, "smooth_tracks")
        if settings.smooth_tracks:
            col.prop(settings, "track_smooth_factor", slider=True)


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
        
        if phase == 'SCENE_SETUP':
            # ══════════════════════════════════════════════
            # ACTIVE PHASE
            # ══════════════════════════════════════════════
            
            # Quality indicator
            error = recon.average_error
            if error < 0.5:
                outer_box.label(text="Excellent solve quality!", icon='CHECKMARK')
            elif error < 1.0:
                outer_box.label(text="Good solve quality", icon='INFO')
            else:
                row = outer_box.row()
                row.alert = True
                row.label(text="High error - consider re-tracking", icon='ERROR')
            
            outer_box.separator()
            
            # Guidance
            outer_box.label(text="Now create your 3D camera", icon='LIGHT')
            
            outer_box.separator()
            
            # Action button
            row = outer_box.row()
            row.scale_y = 2.0
            row.operator("autosolve.setup_scene", text="Setup Tracking Scene", icon='SCENE_DATA')
        
        else:
            # ══════════════════════════════════════════════
            # COMPLETED
            # ══════════════════════════════════════════════
            
            outer_box.label(text="Camera created successfully", icon='CHECKMARK')
            
            outer_box.separator()
            
            # Redo
            outer_box.label(text="Need to redo?", icon='LOOP_BACK')
            row = outer_box.row()
            row.operator("autosolve.setup_scene", text="Redo Setup", icon='FILE_REFRESH')


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
        
        # Single outer box for the entire phase
        outer_box = layout.box()
        
        # Guidance
        outer_box.label(text="Fine-tune your camera motion", icon='LIGHT')
        
        outer_box.separator()
        
        # Smoothing tools
        outer_box.label(text="Smoothing Tools:", icon='MOD_SMOOTH')
        settings = context.scene.autosolve
        
        # Strength sliders
        col = outer_box.column(align=True)
        col.prop(settings, "track_smooth_factor", text="Track Strength", slider=True)

        
        # Buttons
        row = outer_box.row(align=True)
        row.scale_y = 1.4
        row.operator("autosolve.smooth_tracks", text="Tracks", icon='CURVE_PATH')

        
        outer_box.separator()
        
        # Start over
        outer_box.label(text="Start Over:", icon='LOOP_BACK')
        row = outer_box.row(align=True)
        row.operator("autosolve.run_solve", text="Re-track", icon='TRACKING')
        row.operator("autosolve.setup_scene", text="Redo Scene", icon='SCENE_DATA')
        
        




# ═══════════════════════════════════════════════════════════════════════════
# REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════

classes = (
    AUTOSOLVE_PT_main_panel,
    AUTOSOLVE_PT_research_panel,
    AUTOSOLVE_PT_phase1_tracking,
    AUTOSOLVE_PT_phase1_advanced,
    AUTOSOLVE_PT_phase2_scene,
    AUTOSOLVE_PT_phase3_refine,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)



