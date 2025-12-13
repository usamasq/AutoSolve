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
            col = outer_box.column(align=True)
            row = col.row(align=True)
            row.prop(settings, "quality_preset", text="")
            row.prop(settings, "footage_type", text="")
            
            row = col.row(align=True)
            row.prop(settings, "tripod_mode", toggle=True, text="Tripod")
            row.prop(settings, "robust_mode", toggle=True, text="Robust")
            row.prop(settings, "batch_tracking", toggle=True, text="Batch")
        
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
            col = outer_box.column(align=True)
            row = col.row(align=True)
            row.prop(settings, "quality_preset", text="")
            row.prop(settings, "footage_type", text="")
            
            row = col.row(align=True)
            row.prop(settings, "tripod_mode", toggle=True, text="Tripod")
            row.prop(settings, "robust_mode", toggle=True, text="Robust")
            row.prop(settings, "batch_tracking", toggle=True, text="Batch")
            
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
        row.operator("autosolve.clear_annotations", text="", icon='X')


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
        
        # Single outer box for the entire phase
        outer_box = layout.box()
        
        # Guidance
        outer_box.label(text="Polish your results", icon='LIGHT')
        
        outer_box.separator()
        
        # Smoothing tools
        outer_box.label(text="Reduce Jitter:", icon='MOD_SMOOTH')
        settings = context.scene.autosolve
        
        # Strength slider
        col = outer_box.column(align=True)
        col.prop(settings, "track_smooth_factor", text="Smoothing", slider=True)
        
        # Button
        row = outer_box.row(align=True)
        row.scale_y = 1.4
        row.operator("autosolve.smooth_tracks", text="Apply Smoothing", icon='CURVE_PATH')
        
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
    AUTOSOLVE_PT_research_panel,
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



