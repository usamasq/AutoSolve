# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""AutoSolve UI panels - Professional layout with phase-based workflow."""

import bpy
from bpy.types import Panel


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════

class AUTOSOLVE_PT_main_panel(Panel):
    """Main AutoSolve panel in the Movie Clip Editor."""
    
    bl_label = "AutoSolve"
    bl_idname = "AUTOSOLVE_PT_main_panel"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = "AutoSolve"
    
    @classmethod
    def poll(cls, context):
        return context.edit_movieclip is not None
    
    def draw(self, context):
        layout = self.layout
        clip = context.edit_movieclip
        
        # Clip info header
        box = layout.box()
        row = box.row()
        row.label(text=clip.name, icon='SEQUENCE')
        row.label(text=f"{clip.frame_duration} frames")


# ═══════════════════════════════════════════════════════════════════════════
# RESEARCH DATA (Top, collapsed) - Beta research participation
# ═══════════════════════════════════════════════════════════════════════════

class AUTOSOLVE_PT_research_panel(Panel):
    """Research data management panel."""
    
    bl_label = "Research Data"
    bl_idname = "AUTOSOLVE_PT_research_panel"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = "AutoSolve"
    bl_parent_id = "AUTOSOLVE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw_header(self, context):
        self.layout.label(text="", icon='EXPERIMENTAL')
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.autosolve
        
        # Participation toggle
        box = layout.box()
        row = box.row()
        row.prop(settings, "record_edits", text="Contribute to Research", icon='REC')
        
        # Stats summary
        try:
            from .tracker.learning.settings_predictor import SettingsPredictor
            predictor = SettingsPredictor()
            stats = predictor.get_stats()
            
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text=f"Sessions: {stats.get('total_sessions', 0)} | Success: {stats.get('success_rate', 0):.0%}")
        except:
            box.label(text="No data collected yet", icon='INFO')
        
        # Actions
        layout.separator()
        row = layout.row(align=True)
        row.operator("autosolve.export_training_data", text="Export", icon='EXPORT')
        row.operator("autosolve.contribute_data", text="Share", icon='URL')
        row.operator("autosolve.reset_training_data", text="Reset", icon='LOOP_BACK')


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: TRACKING
# ═══════════════════════════════════════════════════════════════════════════

class AUTOSOLVE_PT_phase1_tracking(Panel):
    """Phase 1: Tracking configuration and execution."""
    
    bl_label = "1. Track"
    bl_idname = "AUTOSOLVE_PT_phase1_tracking"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = "AutoSolve"
    bl_parent_id = "AUTOSOLVE_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.autosolve
        
        # Everything in a box for visual distinction
        box = layout.box()
        
        # Main solve button (prominent)
        if settings.is_solving:
            # Progress display
            col = box.column(align=True)
            col.label(text=settings.solve_status, icon='TIME')
            col.prop(settings, "solve_progress", text="")
            col.label(text="Press ESC to cancel", icon='INFO')
        else:
            row = box.row()
            row.scale_y = 1.6
            row.operator("autosolve.run_solve", text="Analyze & Solve", icon='PLAY')
        
        # Quick settings section
        box.separator()
        
        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(settings, "quality_preset", text="")
        row.prop(settings, "footage_type", text="")
        
        row = col.row(align=True)
        row.prop(settings, "tripod_mode", toggle=True)
        row.prop(settings, "robust_mode", toggle=True)


class AUTOSOLVE_PT_phase1_advanced(Panel):
    """Advanced tracking options."""
    
    bl_label = "Advanced"
    bl_idname = "AUTOSOLVE_PT_phase1_advanced"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = "AutoSolve"
    bl_parent_id = "AUTOSOLVE_PT_phase1_tracking"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        settings = context.scene.autosolve
        
        col = layout.column(align=True)
        col.prop(settings, "batch_tracking")
        col.prop(settings, "smooth_tracks")
        if settings.smooth_tracks:
            col.prop(settings, "track_smooth_factor", slider=True)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: SCENE SETUP (only if solved)
# ═══════════════════════════════════════════════════════════════════════════

class AUTOSOLVE_PT_phase2_scene(Panel):
    """Phase 2: Scene setup after successful solve."""
    
    bl_label = "2. Setup Scene"
    bl_idname = "AUTOSOLVE_PT_phase2_scene"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = "AutoSolve"
    bl_parent_id = "AUTOSOLVE_PT_main_panel"
    
    @classmethod
    def poll(cls, context):
        if context.edit_movieclip is None:
            return False
        return context.edit_movieclip.tracking.reconstruction.is_valid
    
    def draw(self, context):
        layout = self.layout
        clip = context.edit_movieclip
        recon = clip.tracking.reconstruction
        settings = context.scene.autosolve
        
        # Everything in a box
        box = layout.box()
        
        # Solve quality indicator
        error = recon.average_error
        num_tracks = len([t for t in clip.tracking.tracks if t.has_bundle])
        
        if error < 0.5:
            icon = 'CHECKMARK'
            quality = "Excellent"
        elif error < 1.0:
            icon = 'INFO'
            quality = "Good"
        else:
            icon = 'ERROR'
            quality = "Review tracks"
        
        row = box.row()
        row.label(text=f"{num_tracks} tracks • {error:.2f}px", icon='TRACKER')
        row.label(text=quality, icon=icon)
        
        box.separator()
        
        # Camera smoothing option
        col = box.column(align=True)
        col.prop(settings, "smooth_camera", text="Smooth Camera Motion")
        if settings.smooth_camera:
            col.prop(settings, "camera_smooth_factor", slider=True)
        
        # Main setup button
        row = box.row()
        row.scale_y = 1.4
        row.operator("autosolve.setup_scene", text="Setup Tracking Scene", icon='SCENE_DATA')


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: REFINE (only if camera exists)
# ═══════════════════════════════════════════════════════════════════════════

class AUTOSOLVE_PT_phase3_refine(Panel):
    """Phase 3: Refinement tools after scene setup."""
    
    bl_label = "3. Refine"
    bl_idname = "AUTOSOLVE_PT_phase3_refine"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = "AutoSolve"
    bl_parent_id = "AUTOSOLVE_PT_main_panel"
    
    @classmethod
    def poll(cls, context):
        # Only show if camera with animation exists
        return (context.scene.camera and 
                context.scene.camera.animation_data and 
                context.scene.camera.animation_data.action)
    
    def draw(self, context):
        layout = self.layout
        
        # Everything in a box
        box = layout.box()
        
        col = box.column(align=True)
        col.label(text="Re-apply Smoothing", icon='MOD_SMOOTH')
        
        row = col.row(align=True)
        row.operator("autosolve.smooth_tracks", text="Tracks", icon='CURVE_PATH')
        row.operator("autosolve.smooth_camera", text="Camera", icon='FCURVE_SNAPSHOT')


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
