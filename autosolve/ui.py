# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""AutoSolve UI panels."""

import bpy
from bpy.types import Panel


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
        settings = context.scene.autosolve
        clip = context.edit_movieclip
        
        # Clip info
        box = layout.box()
        row = box.row()
        row.label(text=clip.name, icon='SEQUENCE')
        row.label(text=f"{clip.frame_duration} frames")
        
        layout.separator()
        
        # Main action - button or progress
        if settings.is_solving:
            # Progress display
            box = layout.box()
            col = box.column(align=True)
            
            # Status text
            col.label(text=settings.solve_status, icon='TIME')
            
            # Progress bar
            col.prop(settings, "solve_progress", text="")
            
            # Cancel hint
            col.label(text="Press ESC to cancel", icon='INFO')
        else:
            # Solve button
            row = layout.row(align=True)
            row.scale_y = 1.5
            row.operator("autosolve.run_solve", text="Analyze & Solve", icon='PLAY')


class AUTOSOLVE_PT_options_panel(Panel):
    """Solver options subpanel."""
    
    bl_label = "Options"
    bl_idname = "AUTOSOLVE_PT_options_panel"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = "AutoSolve"
    bl_parent_id = "AUTOSOLVE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        settings = context.scene.autosolve
        
        col = layout.column(align=True)
        col.prop(settings, "quality_preset")
        col.prop(settings, "footage_type")
        col.separator()
        col.prop(settings, "tripod_mode")
        col.prop(settings, "robust_mode")


class AUTOSOLVE_PT_result_panel(Panel):
    """Solve result display."""
    
    bl_label = "Result"
    bl_idname = "AUTOSOLVE_PT_result_panel"
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
        
        col = layout.column(align=True)
        
        # Track count
        num_tracks = len([t for t in clip.tracking.tracks if t.has_bundle])
        col.label(text=f"Tracks: {num_tracks}", icon='TRACKER')
        
        # Solve error with quality indicator
        error = recon.average_error
        if error < 0.5:
            icon = 'CHECKMARK'
        elif error < 1.0:
            icon = 'INFO'
        else:
            icon = 'ERROR'
        col.label(text=f"Error: {error:.2f} px", icon=icon)
        
        layout.separator()
        
        # Action buttons
        col = layout.column(align=True)
        col.operator("autosolve.setup_scene", text="Setup Tracking Scene", icon='SCENE_DATA')


class AUTOSOLVE_PT_training_panel(Panel):
    """Training data management panel."""
    
    bl_label = "Training Data"
    bl_idname = "AUTOSOLVE_PT_training_panel"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = "AutoSolve"
    bl_parent_id = "AUTOSOLVE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Statistics
        box = layout.box()
        box.label(text="Local Learning Model", icon='EXPERIMENTAL')
        
        try:
            from .solver.smart_tracker import LocalLearningModel
            model = LocalLearningModel()
            
            session_count = model.model.get('session_count', 0)
            classes = len(model.model.get('footage_classes', {}))
            
            col = box.column(align=True)
            col.label(text=f"Sessions tracked: {session_count}")
            col.label(text=f"Footage classes: {classes}")
            
            # Show best settings if available
            if classes > 0:
                # Find best performing class
                best_class = None
                best_error = 999.0
                for cls_name, cls_data in model.model.get('footage_classes', {}).items():
                    err = cls_data.get('best_error', 999.0)
                    if err < best_error:
                        best_error = err
                        best_class = cls_name
                
                if best_class and best_error < 999:
                    col.label(text=f"Best: {best_class} ({best_error:.2f}px)")
        except:
            box.label(text="No training data yet")
        
        layout.separator()
        
        # Actions
        col = layout.column(align=True)
        col.operator("autosolve.export_training_data", text="Export Data", icon='EXPORT')
        col.operator("autosolve.import_training_data", text="Import Data", icon='IMPORT')
        
        layout.separator()
        
        row = layout.row()
        row.operator("autosolve.reset_training_data", text="Reset", icon='LOOP_BACK')


# Registration
classes = (
    AUTOSOLVE_PT_main_panel,
    AUTOSOLVE_PT_options_panel,
    AUTOSOLVE_PT_result_panel,
    AUTOSOLVE_PT_training_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

