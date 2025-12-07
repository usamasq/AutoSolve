# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve Operators - Adaptive Learning Pipeline.

Learns from tracking failures and improves over iterations.
"""

import bpy
from bpy.types import Operator


class TrackingState:
    """State for the modal pipeline."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.tracker = None
        self.phase = 'INIT'
        self.frame_current = 0
        self.frame_start = 0
        self.frame_end = 0
        self.segment_start = 0
        self.tripod_mode = False
        self.iteration = 0
        self.last_analysis = None


_state = TrackingState()


class AUTOSOLVE_OT_run_solve(Operator):
    """Adaptive learning auto-tracking."""
    
    bl_idname = "autosolve.run_solve"
    bl_label = "Analyze & Solve"
    bl_description = "Smart tracking with adaptive learning from failures"
    bl_options = {'REGISTER'}
    
    _timer = None
    
    # Tracking parameters
    SEGMENT_SIZE = 30  # Longer segments before checking
    MIN_TRACKS = 25
    MIN_LIFESPAN = 5   # Minimum frames for a track to count
    
    @classmethod
    def poll(cls, context):
        if context.edit_movieclip is None:
            return False
        return not context.scene.autosolve.is_solving
    
    def execute(self, context):
        clip = context.edit_movieclip
        settings = context.scene.autosolve
        
        if clip.frame_duration < 10:
            self.report({'ERROR'}, "Clip must have at least 10 frames")
            return {'CANCELLED'}
        
        from .solver.smart_tracker import SmartTracker, sync_scene_to_clip
        
        robust = getattr(settings, 'robust_mode', False)
        footage_type = getattr(settings, 'footage_type', 'AUTO')
        _state.reset()
        _state.tracker = SmartTracker(clip, robust_mode=robust, footage_type=footage_type)
        _state.frame_start = clip.frame_start
        _state.frame_end = clip.frame_start + clip.frame_duration - 1
        _state.frame_current = clip.frame_start
        _state.segment_start = clip.frame_start
        _state.tripod_mode = settings.tripod_mode
        _state.phase = 'LOAD_LEARNING'
        _state.iteration = 0
        
        sync_scene_to_clip(clip)
        
        settings.is_solving = True
        settings.solve_progress = 0.0
        settings.solve_status = "Initializing..."
        
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        if event.type == 'ESC':
            return self._cancel(context)
        
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        
        settings = context.scene.autosolve
        tracker = _state.tracker
        clip = tracker.clip
        
        try:
            # ═══════════════════════════════════════════════════════════════
            # PHASE: LOAD LEARNING DATA
            # ═══════════════════════════════════════════════════════════════
            if _state.phase == 'LOAD_LEARNING':
                settings.solve_status = "Loading learning data..."
                settings.solve_progress = 0.01
                
                # Try to load previous learning
                tracker.load_learning(clip.name)
                
                _state.phase = 'CONFIGURE'
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: CONFIGURE
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'CONFIGURE':
                settings.solve_status = f"Configuring (iteration {_state.iteration + 1})..."
                settings.solve_progress = 0.03
                
                tracker.analyze_footage()
                tracker.configure_settings()
                
                # Learn from any user-placed templates BEFORE clearing
                if len(tracker.tracking.tracks) > 0:
                    user_learned = tracker.learn_from_user_templates()
                    if user_learned.get('total_templates', 0) > 0:
                        _state.user_learned = user_learned  # Store for later use
                        print(f"AutoSolve: Learned from {user_learned['total_templates']} user templates")
                
                tracker.clear_tracks()
                
                # Pre-tracking validation
                is_valid, issues = tracker.validate_pre_tracking()
                if not is_valid and _state.iteration >= tracker.MAX_ITERATIONS:
                    self.report({'ERROR'}, f"Validation failed: {'; '.join(issues)}")
                    return self._cancel(context)
                
                _state.phase = 'DETECT'
                _state.frame_current = _state.frame_start
                context.scene.frame_set(_state.frame_start)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: STRATEGIC DETECT (Distributed, not carpet-bombing)
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'DETECT':
                settings.solve_status = f"Detecting features (strategic)..."
                settings.solve_progress = 0.05
                
                # Use strategic detection: ~3 markers per region
                num = tracker.detect_strategic_features(markers_per_region=3)
                
                if num < 8:
                    self.report({'WARNING'}, f"Only {num} features detected")
                    if _state.iteration < tracker.MAX_ITERATIONS:
                        _state.phase = 'RETRY_DECISION'
                    else:
                        return self._cancel(context)
                    return {'RUNNING_MODAL'}
                
                print(f"AutoSolve: Strategic detection - {num} balanced markers")
                tracker.select_all_tracks()
                _state.phase = 'TRACK_FORWARD'
                _state.segment_start = _state.frame_current
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: TRACK FORWARD
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'TRACK_FORWARD':
                progress = (_state.frame_current - _state.frame_start) / clip.frame_duration
                settings.solve_status = f"Tracking... {int(progress*100)}%"
                settings.solve_progress = 0.05 + progress * 0.40
                
                if _state.frame_current < _state.frame_end:
                    tracker.track_frame(backwards=False)
                    _state.frame_current += 1
                    context.scene.frame_set(_state.frame_current)
                    
                    # Live validation every 10 frames
                    if _state.frame_current % 10 == 0:
                        tracker.validate_track_quality(_state.frame_current)
                    
                    # Check segment for replenishment
                    frames_in_seg = _state.frame_current - _state.segment_start
                    if frames_in_seg >= self.SEGMENT_SIZE:
                        active = tracker.count_active_tracks(_state.frame_current)
                        if active < self.MIN_TRACKS:
                            # Detect more features
                            threshold = tracker.current_settings.get('threshold', 0.3)
                            tracker.detect_features(threshold * 1.2)  # Slightly more lenient
                            tracker.select_all_tracks()
                        _state.segment_start = _state.frame_current
                    
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                else:
                    _state.phase = 'TRACK_BACKWARD'
                    _state.frame_current = _state.frame_end
                    context.scene.frame_set(_state.frame_end)
                    tracker.select_all_tracks()
                    return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: TRACK BACKWARD
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'TRACK_BACKWARD':
                progress = (_state.frame_end - _state.frame_current) / clip.frame_duration
                settings.solve_status = f"Backfilling... {int(progress*100)}%"
                settings.solve_progress = 0.45 + progress * 0.15
                
                if _state.frame_current > _state.frame_start:
                    tracker.track_frame(backwards=True)
                    _state.frame_current -= 1
                    context.scene.frame_set(_state.frame_current)
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                else:
                    _state.phase = 'ANALYZE'
                    return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: ANALYZE (Learning)
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'ANALYZE':
                settings.solve_status = "Analyzing track quality..."
                settings.solve_progress = 0.62
                
                _state.last_analysis = tracker.analyze_and_learn()
                
                # Check if we should retry (very low success rate)
                if tracker.should_retry(_state.last_analysis):
                    _state.phase = 'RETRY_DECISION'
                else:
                    # Move to coverage analysis for balanced distribution
                    _state.phase = 'ANALYZE_COVERAGE'
                
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: ANALYZE COVERAGE (Balance Check)
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'ANALYZE_COVERAGE':
                settings.solve_status = "Checking coverage balance..."
                settings.solve_progress = 0.65
                
                summary = tracker.get_coverage_analysis()
                print(f"AutoSolve: Coverage - {summary['regions_with_tracks']}/9 regions, "
                      f"balance: {summary['balance_score']:.2f}, temporal: {summary['temporal_coverage']:.0%}")
                
                if not summary['is_balanced'] and tracker.should_continue_strategic():
                    _state.phase = 'FILL_GAPS'
                else:
                    _state.phase = 'FILTER_SHORT'
                
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: FILL COVERAGE GAPS (Strategic Iteration)
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'FILL_GAPS':
                settings.solve_status = f"Filling coverage gaps (iter {tracker.strategic_iteration + 1})..."
                settings.solve_progress = 0.67
                
                # Fill gaps and track new markers
                result = tracker.strategic_track_iteration()
                
                if result['markers_added'] > 0:
                    # Track the new markers
                    tracker.select_all_tracks()
                    _state.phase = 'TRACK_NEW'
                    _state.frame_current = _state.frame_start
                    context.scene.frame_set(_state.frame_start)
                else:
                    # No more gaps to fill, proceed
                    _state.phase = 'FILTER_SHORT'
                
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: TRACK NEW (Track newly added markers)
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'TRACK_NEW':
                progress = (_state.frame_current - _state.frame_start) / clip.frame_duration
                settings.solve_status = f"Tracking new markers... {int(progress*100)}%"
                settings.solve_progress = 0.68 + progress * 0.02
                
                if _state.frame_current < _state.frame_end:
                    tracker.track_frame(backwards=False)
                    _state.frame_current += 1
                    context.scene.frame_set(_state.frame_current)
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                else:
                    # Check coverage again
                    _state.phase = 'ANALYZE_COVERAGE'
                    return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: RETRY DECISION
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'RETRY_DECISION':
                if _state.iteration < tracker.MAX_ITERATIONS:
                    settings.solve_status = f"Retrying with improved settings..."
                    settings.solve_progress = 0.10
                    
                    _state.iteration += 1
                    tracker.prepare_retry()
                    
                    _state.phase = 'DETECT'
                    _state.frame_current = _state.frame_start
                    context.scene.frame_set(_state.frame_start)
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                else:
                    # Max retries reached, proceed with what we have
                    _state.phase = 'FILTER_SHORT'
                    return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: FILTER SHORT TRACKS
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'FILTER_SHORT':
                settings.solve_status = "Filtering short tracks..."
                settings.solve_progress = 0.68
                
                tracker.filter_short_tracks(min_frames=self.MIN_LIFESPAN)
                
                num = len(tracker.tracking.tracks)
                if num < 8:
                    self.report({'ERROR'}, f"Only {num} tracks - footage may be too difficult")
                    return self._cancel(context)
                
                _state.phase = 'FILTER_SPIKES'
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: FILTER SPIKES
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'FILTER_SPIKES':
                settings.solve_status = "Filtering outliers..."
                settings.solve_progress = 0.72
                
                tracker.filter_spikes(limit_multiplier=8.0)
                
                # Pre-solve validation
                is_valid, issues = tracker.validate_pre_solve()
                if not is_valid:
                    self.report({'WARNING'}, f"Pre-solve issues: {'; '.join(issues[:2])}")
                    # Continue anyway but log warning
                
                _state.phase = 'SOLVE_DRAFT'
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: DRAFT SOLVE
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'SOLVE_DRAFT':
                settings.solve_status = "Draft solve..."
                settings.solve_progress = 0.78
                
                success = tracker.solve_camera(tripod_mode=_state.tripod_mode)
                
                if success:
                    _state.phase = 'FILTER_ERROR'
                else:
                    _state.phase = 'SOLVE_FINAL'
                
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: FILTER HIGH ERROR
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'FILTER_ERROR':
                settings.solve_status = "Refining..."
                settings.solve_progress = 0.85
                
                tracker.filter_high_error(max_error=3.0)
                
                _state.phase = 'SOLVE_FINAL'
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: FINAL SOLVE
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'SOLVE_FINAL':
                settings.solve_status = "Final solve..."
                settings.solve_progress = 0.92
                
                # Sanitize tracks to prevent Ceres solver errors
                removed = tracker.sanitize_tracks_before_solve()
                
                num = len(tracker.tracking.tracks)
                if num < 8:
                    self.report({'ERROR'}, f"Only {num} tracks (removed {removed} bad)")
                    return self._cancel(context)
                
                success = tracker.solve_camera(tripod_mode=_state.tripod_mode)
                
                if not success:
                    self.report({'ERROR'}, "Solve failed - try Tripod Mode")
                    return self._cancel(context)
                
                # Check if we need refinement (error > 2px)
                error = tracker.get_solve_error()
                tracker.best_solve_error = error
                tracker.best_bundle_count = tracker.get_bundle_count()
                
                if error > 2.0:
                    # Start refinement loop
                    _state.phase = 'REFINE'
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                else:
                    # Good enough, finish
                    _state.phase = 'COMPLETE'
                    return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: ITERATIVE REFINEMENT
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'REFINE':
                settings.solve_status = f"Refining (iteration {tracker.refinement_iteration + 1})..."
                settings.solve_progress = 0.94 + (tracker.refinement_iteration * 0.01)
                
                # Check if we should continue refining
                if tracker.should_continue_refinement():
                    success = tracker.refine_solve()
                    if success:
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                
                # Done refining (or can't improve further)
                _state.phase = 'COMPLETE'
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: COMPLETE
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'COMPLETE':
                error = tracker.get_solve_error()
                bundles = tracker.get_bundle_count()
                
                # Extract training data and save session
                training_data = tracker.extract_training_data()
                tracker.save_session_results(success=True, solve_error=error)
                
                # Save user template learning (with success metrics)
                if hasattr(_state, 'user_learned') and _state.user_learned:
                    final_learned = tracker.learn_from_user_templates()
                    tracker.save_user_learning(final_learned)
                
                settings.solve_status = "Complete!"
                settings.solve_progress = 1.0
                settings.has_solve = True
                settings.solve_error = error
                settings.point_count = bundles
                
                msg = f"Solved: {bundles} tracks, {error:.2f}px error"
                if _state.iteration > 0:
                    msg += f" (after {_state.iteration + 1} iterations)"
                if tracker.refinement_iteration > 0:
                    msg += f" + {tracker.refinement_iteration} refinements"
                self.report({'INFO'}, msg)
                
                return self._finish(context)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Error: {str(e)}")
            return self._cancel(context)
        
        return {'RUNNING_MODAL'}
    
    def _finish(self, context):
        self._cleanup(context)
        return {'FINISHED'}
    
    def _cancel(self, context):
        self._cleanup(context)
        return {'CANCELLED'}
    
    def _cleanup(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        
        context.scene.autosolve.is_solving = False
        context.area.tag_redraw()


class AUTOSOLVE_OT_setup_scene(Operator):
    """Set up scene from tracking."""
    
    bl_idname = "autosolve.setup_scene"
    bl_label = "Setup Tracking Scene"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        if context.edit_movieclip is None:
            return False
        return context.edit_movieclip.tracking.reconstruction.is_valid
    
    def execute(self, context):
        bpy.ops.clip.setup_tracking_scene(action='BACKGROUND_AND_CAMERA')
        self.report({'INFO'}, "Scene set up successfully")
        return {'FINISHED'}


class AUTOSOLVE_OT_export_training_data(Operator):
    """Export training data to share with others or backup."""
    
    bl_idname = "autosolve.export_training_data"
    bl_label = "Export Training Data"
    bl_options = {'REGISTER'}
    
    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to export training data",
        default="eztrack_training.json",
        subtype='FILE_PATH',
    )
    
    filter_glob: bpy.props.StringProperty(
        default="*.json",
        options={'HIDDEN'},
    )
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        import json
        from pathlib import Path
        
        try:
            from .solver.smart_tracker import LocalLearningModel
            
            model = LocalLearningModel()
            
            # Prepare export data with metadata
            export_data = {
                'format_version': 1,
                'export_type': 'eztrack_training_data',
                'session_count': model.model.get('session_count', 0),
                'footage_classes': model.model.get('footage_classes', {}),
                'region_models': model.model.get('region_models', {}),
            }
            
            filepath = Path(self.filepath)
            if not filepath.suffix:
                filepath = filepath.with_suffix('.json')
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.report({'INFO'}, f"Exported {export_data['session_count']} sessions to {filepath.name}")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {str(e)}")
            return {'CANCELLED'}


class AUTOSOLVE_OT_import_training_data(Operator):
    """Import training data from file."""
    
    bl_idname = "autosolve.import_training_data"
    bl_label = "Import Training Data"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to training data file",
        subtype='FILE_PATH',
    )
    
    filter_glob: bpy.props.StringProperty(
        default="*.json",
        options={'HIDDEN'},
    )
    
    merge: bpy.props.BoolProperty(
        name="Merge with Existing",
        description="Merge with existing data instead of replacing",
        default=True,
    )
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        import json
        from pathlib import Path
        
        try:
            from .solver.smart_tracker import LocalLearningModel
            
            filepath = Path(self.filepath)
            
            if not filepath.exists():
                self.report({'ERROR'}, "File not found")
                return {'CANCELLED'}
            
            with open(filepath) as f:
                import_data = json.load(f)
            
            # Validate format
            if import_data.get('export_type') != 'eztrack_training_data':
                self.report({'ERROR'}, "Invalid training data format")
                return {'CANCELLED'}
            
            model = LocalLearningModel()
            
            if self.merge:
                # Merge data
                for cls, data in import_data.get('footage_classes', {}).items():
                    if cls not in model.model['footage_classes']:
                        model.model['footage_classes'][cls] = data
                    else:
                        # Keep the one with more samples
                        existing = model.model['footage_classes'][cls]
                        if data.get('sample_count', 0) > existing.get('sample_count', 0):
                            model.model['footage_classes'][cls] = data
                
                for region, data in import_data.get('region_models', {}).items():
                    if region not in model.model['region_models']:
                        model.model['region_models'][region] = data
                    else:
                        model.model['region_models'][region]['total'] += data.get('total', 0)
                        model.model['region_models'][region]['successful'] += data.get('successful', 0)
                
                model.model['session_count'] += import_data.get('session_count', 0)
            else:
                # Replace
                model.model['footage_classes'] = import_data.get('footage_classes', {})
                model.model['region_models'] = import_data.get('region_models', {})
                model.model['session_count'] = import_data.get('session_count', 0)
            
            model.save()
            
            imported_count = import_data.get('session_count', 0)
            action = "Merged" if self.merge else "Imported"
            self.report({'INFO'}, f"{action} {imported_count} sessions from {filepath.name}")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {str(e)}")
            return {'CANCELLED'}


class AUTOSOLVE_OT_reset_training_data(Operator):
    """Reset all training data to defaults."""
    
    bl_idname = "autosolve.reset_training_data"
    bl_label = "Reset Training Data"
    bl_options = {'REGISTER', 'UNDO'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)
    
    def execute(self, context):
        try:
            from .solver.smart_tracker import LocalLearningModel
            
            model = LocalLearningModel()
            
            # Reset to empty
            model.model = {
                'version': 2,
                'session_count': 0,
                'footage_classes': {},
                'region_models': {},
                'pretrained_used': True,
            }
            model.save()
            
            self.report({'INFO'}, "Training data reset to defaults")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Reset failed: {str(e)}")
            return {'CANCELLED'}


class AUTOSOLVE_OT_view_training_stats(Operator):
    """View training data statistics."""
    
    bl_idname = "autosolve.view_training_stats"
    bl_label = "View Training Statistics"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        try:
            from .solver.smart_tracker import LocalLearningModel
            
            model = LocalLearningModel()
            
            session_count = model.model.get('session_count', 0)
            classes = len(model.model.get('footage_classes', {}))
            
            # Calculate success stats
            total_success = 0
            total_samples = 0
            for cls_data in model.model.get('footage_classes', {}).values():
                total_success += cls_data.get('success_count', 0)
                total_samples += cls_data.get('sample_count', 0)
            
            success_rate = total_success / max(total_samples, 1)
            
            self.report({'INFO'}, 
                f"Sessions: {session_count} | Classes: {classes} | Success: {success_rate:.0%}")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            return {'CANCELLED'}


# Registration
classes = (
    AUTOSOLVE_OT_run_solve,
    AUTOSOLVE_OT_setup_scene,
    AUTOSOLVE_OT_export_training_data,
    AUTOSOLVE_OT_import_training_data,
    AUTOSOLVE_OT_reset_training_data,
    AUTOSOLVE_OT_view_training_stats,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

