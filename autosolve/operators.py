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

# Global recorders (persist between solves to capture user behavior)
_edit_recorder = None
_behavior_recorder = None
_last_solve_settings = None
_last_solve_error = None
_last_solve_footage_class = None

# Pending behavior to learn from AFTER the new solve completes
_pending_behavior = None
_pending_behavior_footage_class = None


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
        
        # ═══════════════════════════════════════════════════════════════
        # BEHAVIOR LEARNING: Capture and learn from user edits since last solve
        # ═══════════════════════════════════════════════════════════════
        global _edit_recorder, _behavior_recorder
        global _last_solve_settings, _last_solve_error, _last_solve_footage_class
        
        if settings.record_edits:
            # 1. Stop edit monitoring and save
            if _edit_recorder:
                edit_session = _edit_recorder.stop_monitoring()
                if edit_session:
                    from .solver.learning.session_recorder import SessionRecorder
                    try:
                        recorder = SessionRecorder()
                        recorder._save_edit_session(edit_session)
                    except Exception as e:
                        print(f"AutoSolve: Failed to save edit session: {e}")
            
            # 2. Stop behavior monitoring and STORE for later
            #    (Don't learn yet - we need the new solve's error to compare)
            if _behavior_recorder and _last_solve_footage_class:
                # Get current settings (user may have changed them)
                current_settings = {
                    'pattern_size': clip.tracking.settings.default_pattern_size,
                    'search_size': clip.tracking.settings.default_search_size,
                    'correlation': clip.tracking.settings.default_correlation_min,
                }
                
                # Stop monitoring - pass settings for comparison, 
                # but error comparison will happen AFTER new solve
                behavior = _behavior_recorder.stop_monitoring(current_settings, None)
                if behavior:
                    _behavior_recorder.save_behavior(behavior)
                    
                    # Store pending behavior - will learn after new solve completes
                    from dataclasses import asdict
                    _pending_behavior = asdict(behavior)
                    _pending_behavior_footage_class = _last_solve_footage_class
                    
                    # Store old error for comparison after new solve
                    _pending_behavior['_previous_error'] = _last_solve_error
                    
                    print(f"AutoSolve: Captured user behavior - will learn after new solve")
        
        # Reset recorders (but keep pending behavior)
        _behavior_recorder = None
        _edit_recorder = None
        _last_solve_settings = None
        _last_solve_error = None
        
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
                
                # 1.3 Failure pattern warning: Check if similar footage has failed before
                if tracker.predictor.should_avoid_settings(
                    tracker.footage_class, tracker.current_settings
                ):
                    self.report({'WARNING'}, "Similar clips have failed - using robust mode")
                    if not settings.robust_mode:
                        settings.robust_mode = True
                        tracker.robust_mode = True
                        # Re-apply robust mode adjustments
                        tracker.current_settings['pattern_size'] = int(
                            tracker.current_settings.get('pattern_size', 15) * 1.4)
                        tracker.current_settings['search_size'] = int(
                            tracker.current_settings.get('search_size', 71) * 1.4)
                        tracker.current_settings['correlation'] = max(
                            0.45, tracker.current_settings.get('correlation', 0.7) - 0.15)
                        tracker.current_settings['motion_model'] = 'Affine'
                
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
                # Use optimal start frame (middle of clip for bidirectional tracking)
                _state.optimal_start = tracker.get_optimal_start_frame()
                _state.frame_current = _state.optimal_start
                context.scene.frame_set(_state.optimal_start)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: SMART DETECT (Unified detection with learning)
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'DETECT':
                settings.solve_progress = 0.05
                settings.solve_status = "Detecting features..."
                
                # Unified smart detection (uses cached probe, learned settings)
                num = tracker.detect_features_smart(
                    markers_per_region=3,
                    use_cached_probe=(_state.iteration > 0)  # Cache on retry
                )
                
                if num < 8:
                    self.report({'WARNING'}, f"Only {num} features detected")
                    if _state.iteration < tracker.MAX_ITERATIONS:
                        _state.phase = 'RETRY_DECISION'
                    else:
                        return self._cancel(context)
                    return {'RUNNING_MODAL'}
                
                print(f"AutoSolve: Smart detection - {num} balanced markers")
                tracker.select_all_tracks()
                _state.phase = 'TRACK_FORWARD'
                _state.segment_start = _state.frame_current
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: TRACK FORWARD
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'TRACK_FORWARD':
                # Check if user wants batch tracking (faster, no progress feedback)
                if settings.batch_tracking and _state.frame_current == _state.segment_start:
                    # Batch mode: track all frames at once
                    settings.solve_status = "Batch tracking forward..."
                    settings.solve_progress = 0.25
                    
                    frames = tracker.track_sequence(
                        _state.frame_current, 
                        _state.frame_end, 
                        backwards=False
                    )
                    print(f"AutoSolve: Batch tracked {frames} frames forward")
                    
                    _state.frame_current = _state.frame_end
                    _state.phase = 'TRACK_BACKWARD'
                    # Start 3 FRAMES AFTER optimal_start (into forward-tracked territory) to ensure overlap
                    # This re-tracks existing markers to establish trajectory before hitting new frames
                    optimal_start = getattr(_state, 'optimal_start', _state.frame_start)
                    _state.frame_current = min(optimal_start + 3, _state.frame_end)
                    context.scene.frame_set(_state.frame_current)
                    tracker.select_all_tracks()
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                
                # Frame-by-frame mode with ADAPTIVE monitoring
                progress = (_state.frame_current - _state.frame_start) / clip.frame_duration
                settings.solve_status = f"Tracking... {int(progress*100)}%"
                settings.solve_progress = 0.05 + progress * 0.40
                
                if _state.frame_current < _state.frame_end:
                    tracker.track_frame(backwards=False)
                    _state.frame_current += 1
                    context.scene.frame_set(_state.frame_current)
                    
                    # ADAPTIVE monitoring every MONITOR_INTERVAL frames
                    if _state.frame_current % tracker.MONITOR_INTERVAL == 0:
                        tracker.monitor_and_replenish(_state.frame_current, backwards=False)
                    
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                else:
                    # Forward tracking complete, now track BACKWARD from END to cover all markers
                    _state.phase = 'TRACK_BACKWARD'
                    # Start from frame_end (not optimal_start) to ensure markers added
                    # during forward pass are fully covered backward
                    _state.frame_current = _state.frame_end
                    context.scene.frame_set(_state.frame_current)
                    tracker.select_all_tracks()
                    return {'RUNNING_MODAL'}
            # ═══════════════════════════════════════════════════════════════
            # PHASE: TRACK BACKWARD (Adaptive)
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'TRACK_BACKWARD':
                # Check if user wants batch tracking (faster, no progress feedback)
                if settings.batch_tracking and _state.frame_current == _state.frame_end:
                    settings.solve_status = "Batch tracking backward..."
                    settings.solve_progress = 0.55
                    
                    frames = tracker.track_sequence(
                        _state.frame_current,
                        _state.frame_start,
                        backwards=True
                    )
                    print(f"AutoSolve: Batch tracked {frames} frames backward")
                    
                    # Cleanup and go directly to ANALYZE
                    tracker.cleanup_tracks()
                    _state.phase = 'ANALYZE'
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                
                # Frame-by-frame mode with ADAPTIVE monitoring
                progress = (_state.frame_end - _state.frame_current) / clip.frame_duration
                settings.solve_status = f"Tracking backward... {int(progress*100)}%"
                settings.solve_progress = 0.45 + progress * 0.15
                
                if _state.frame_current > _state.frame_start:
                    tracker.track_frame(backwards=True)
                    _state.frame_current -= 1
                    context.scene.frame_set(_state.frame_current)
                    
                    # ADAPTIVE monitoring every MONITOR_INTERVAL frames
                    if _state.frame_current % tracker.MONITOR_INTERVAL == 0:
                        tracker.monitor_and_replenish(_state.frame_current, backwards=True)
                    
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                else:
                    # Bidirectional tracking complete!
                    # Cleanup and go directly to ANALYZE (skip MID_REPLENISH and other phases)
                    tracker.cleanup_tracks()
                    _state.phase = 'ANALYZE'
                    return {'RUNNING_MODAL'}
            
            # NOTE: MID_REPLENISH, MID_TRACK_FORWARD, MID_TRACK_BACKWARD phases removed.
            # Replaced by adaptive monitor_and_replenish() during TRACK_FORWARD/BACKWARD.
            
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: ANALYZE (Learning)
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'ANALYZE':
                settings.solve_status = "Analyzing track quality..."
                settings.solve_progress = 0.65
                
                _state.last_analysis = tracker.analyze_and_learn()
                
                # Check if we should retry (very low success rate)
                if tracker.should_retry(_state.last_analysis):
                    _state.phase = 'RETRY_DECISION'
                else:
                    # Go directly to cleanup (adaptive monitoring handles gaps)
                    _state.phase = 'FILTER_SHORT'
                
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # NOTE: ANALYZE_COVERAGE, FILL_GAPS, TRACK_GAP_*, VERIFY_TIMELINE, EXTEND_* phases removed.
            # These were replaced by adaptive monitor_and_replenish() during TRACK_FORWARD/BACKWARD,
            # which handles gaps in real-time without requiring additional full-clip tracking passes.
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: RETRY DECISION
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'RETRY_DECISION':
                if _state.iteration < tracker.MAX_ITERATIONS:
                    settings.solve_status = f"Retrying with improved settings..."
                    settings.solve_progress = 0.10
                    
                    _state.iteration += 1
                    
                    # Try diagnostic-driven fix first
                    applied_diagnostic_fix = False
                    diagnosis = None
                    if _state.last_analysis:
                        from .solver.learning.failure_diagnostics import FailureDiagnostics
                        diagnostics = FailureDiagnostics()
                        diagnosis = diagnostics.diagnose(_state.last_analysis, tracker.current_settings)
                        
                        if diagnosis.confidence > 0.5:
                            # Record failure for learning (avoid these settings next time)
                            if hasattr(tracker, 'predictor') and tracker.predictor:
                                footage_class = tracker.predictor.classify_footage(clip)
                                tracker.predictor.record_failure(
                                    footage_class, 
                                    diagnosis.pattern.value, 
                                    tracker.current_settings
                                )
                            
                            tracker.current_settings = diagnostics.apply_fix(
                                tracker.current_settings, diagnosis
                            )
                            print(f"AutoSolve: Applied {diagnosis.pattern.value} fix: {diagnosis.description}")
                            applied_diagnostic_fix = True
                    
                    # If no confident diagnosis, apply aggressive generic fix
                    if not applied_diagnostic_fix:
                        old_search = tracker.current_settings.get('search_size', 71)
                        tracker.current_settings['search_size'] = int(old_search * 1.5)
                        tracker.current_settings['correlation'] = max(
                            0.5, tracker.current_settings.get('correlation', 0.7) - 0.1
                        )
                        tracker.current_settings['motion_model'] = 'Affine'
                        print(f"AutoSolve: Aggressive retry - search_size: {old_search} → {tracker.current_settings['search_size']}")
                    
                    # Clear probe cache so detection re-analyzes
                    tracker.cached_motion_probe = None
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
            # PHASE: CLEANUP (unified filtering in one pass)
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'FILTER_SHORT':
                # Note: keeping 'FILTER_SHORT' as phase name for backwards compat
                # but this now does ALL cleanup in one pass
                settings.solve_status = "Cleaning tracks..."
                settings.solve_progress = 0.75
                
                # Unified cleanup: short tracks + spikes + non-rigid
                tracker.cleanup_tracks(
                    min_frames=self.MIN_LIFESPAN,
                    spike_multiplier=8.0,
                    jitter_threshold=0.6,
                    coherence_threshold=0.4
                )
                
                num = len(tracker.tracking.tracks)
                if num < 8:
                    self.report({'ERROR'}, f"Only {num} tracks - footage may be too difficult")
                    return self._cancel(context)
                
                # Pre-solve validation
                is_valid, issues = tracker.validate_pre_solve()
                if not is_valid:
                    self.report({'WARNING'}, f"Pre-solve issues: {'; '.join(issues[:2])}")
                
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
                
                return self._finish(context, success=True)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Error: {str(e)}")
            return self._cancel(context)
        
        return {'RUNNING_MODAL'}
    
    def _finish(self, context, success=False):
        # Start behavior and edit monitoring after successful solve
        global _edit_recorder, _behavior_recorder
        global _last_solve_settings, _last_solve_error, _last_solve_footage_class
        global _pending_behavior, _pending_behavior_footage_class
        
        settings = context.scene.autosolve
        clip = context.edit_movieclip
        tracker = _state.tracker
        
        # ═══════════════════════════════════════════════════════════════
        # LEARN FROM PENDING BEHAVIOR (now we have the ACTUAL new error)
        # ═══════════════════════════════════════════════════════════════
        if success and _pending_behavior and _pending_behavior_footage_class:
            new_error = clip.tracking.reconstruction.average_error if clip.tracking.reconstruction.is_valid else None
            previous_error = _pending_behavior.get('_previous_error')
            
            if new_error is not None and previous_error is not None:
                # NOW we can correctly compute if user's changes helped
                improvement = previous_error - new_error
                _pending_behavior['re_solve'] = {
                    'attempted': True,
                    'error_before': previous_error,
                    'error_after': new_error,
                    'improvement': improvement,
                    'improved': improvement > 0
                }
                
                # Learn from behavior with correct error comparison
                from .solver.learning.settings_predictor import SettingsPredictor
                try:
                    predictor = SettingsPredictor()
                    predictor.learn_from_behavior(_pending_behavior_footage_class, _pending_behavior)
                    
                    if improvement > 0:
                        print(f"AutoSolve: Learned from user behavior (error improved: {previous_error:.2f}→{new_error:.2f})")
                    else:
                        print(f"AutoSolve: Noted user behavior (error not improved: {previous_error:.2f}→{new_error:.2f})")
                except Exception as e:
                    print(f"AutoSolve: Failed to learn from behavior: {e}")
            
            # Clear pending behavior
            _pending_behavior = None
            _pending_behavior_footage_class = None
        
        if success and settings.record_edits and clip:
            # Store current solve state for comparison when user re-solves
            _last_solve_settings = {
                'pattern_size': clip.tracking.settings.default_pattern_size,
                'search_size': clip.tracking.settings.default_search_size,
                'correlation': clip.tracking.settings.default_correlation_min,
            }
            _last_solve_error = clip.tracking.reconstruction.average_error if clip.tracking.reconstruction.is_valid else 0.0
            _last_solve_footage_class = tracker.footage_class if tracker else None
            
            # Create session ID for linking behavior with session
            from datetime import datetime
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Start edit recorder
            from .solver.learning.user_edit_recorder import UserEditRecorder
            _edit_recorder = UserEditRecorder(clip)
            _edit_recorder.start_monitoring()
            
            # Start behavior recorder
            from .solver.learning.behavior_recorder import BehaviorRecorder
            _behavior_recorder = BehaviorRecorder()
            _behavior_recorder.start_monitoring(
                clip=clip,
                settings=_last_solve_settings,
                solve_error=_last_solve_error,
                session_id=session_id
            )
            
            print(f"AutoSolve: Monitoring for user edits and behavior changes")
        
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
    """Export training data as ZIP for ML training."""
    
    bl_idname = "autosolve.export_training_data"
    bl_label = "Export Training Data"
    bl_options = {'REGISTER'}
    
    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to export training data",
        default="autosolve_training.zip",
        subtype='FILE_PATH',
    )
    
    filter_glob: bpy.props.StringProperty(
        default="*.zip",
        options={'HIDDEN'},
    )
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        import json
        import zipfile
        from pathlib import Path
        from datetime import datetime
        
        try:
            from .solver.learning.settings_predictor import SettingsPredictor
            
            predictor = SettingsPredictor()
            base_dir = Path(bpy.utils.user_resource('DATAFILES')) / 'autosolve'
            
            filepath = Path(self.filepath)
            if not filepath.suffix or filepath.suffix != '.zip':
                filepath = filepath.with_suffix('.zip')
            
            session_count = 0
            behavior_count = 0
            total_tracks = 0
            
            with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 1. Export sessions
                sessions_dir = base_dir / 'sessions'
                if sessions_dir.exists():
                    for session_file in sessions_dir.glob('*.json'):
                        try:
                            with open(session_file) as f:
                                data = json.load(f)
                                total_tracks += len(data.get('tracks', []))
                            # Use just timestamp as filename
                            zf.write(session_file, f"sessions/{session_file.name}")
                            session_count += 1
                        except Exception as e:
                            print(f"AutoSolve: Failed to export {session_file.name}: {e}")
                
                # 2. Export behavior data
                behavior_dir = base_dir / 'behavior'
                if behavior_dir.exists():
                    for behavior_file in behavior_dir.glob('*.json'):
                        try:
                            zf.write(behavior_file, f"behavior/{behavior_file.name}")
                            behavior_count += 1
                        except Exception as e:
                            print(f"AutoSolve: Failed to export {behavior_file.name}: {e}")
                
                # 3. Export model
                model_data = {
                    'version': predictor.model.get('version', 2),
                    'global_stats': predictor.model.get('global_stats', {}),
                    'footage_classes': predictor.model.get('footage_classes', {}),
                    'region_models': predictor.model.get('region_models', {}),
                    'failure_patterns': predictor.model.get('failure_patterns', {}),
                }
                zf.writestr('model.json', json.dumps(model_data, indent=2))
                
                # 4. Create manifest
                manifest = {
                    'export_version': 1,
                    'export_date': datetime.now().isoformat(),
                    'addon_version': '1.0.0',
                    'session_count': session_count,
                    'behavior_count': behavior_count,
                    'total_tracks': total_tracks,
                }
                zf.writestr('manifest.json', json.dumps(manifest, indent=2))
            
            self.report({'INFO'}, 
                f"Exported {session_count} sessions, {behavior_count} behaviors to {filepath.name}")
            return {'FINISHED'}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Export failed: {str(e)}")
            return {'CANCELLED'}


class AUTOSOLVE_OT_import_training_data(Operator):
    """Import training data from ZIP file."""
    
    bl_idname = "autosolve.import_training_data"
    bl_label = "Import Training Data"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to training data file",
        subtype='FILE_PATH',
    )
    
    filter_glob: bpy.props.StringProperty(
        default="*.zip;*.json",
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
        import zipfile
        from pathlib import Path
        
        try:
            from .solver.learning.settings_predictor import SettingsPredictor
            
            filepath = Path(self.filepath)
            
            if not filepath.exists():
                self.report({'ERROR'}, "File not found")
                return {'CANCELLED'}
            
            predictor = SettingsPredictor()
            base_dir = Path(bpy.utils.user_resource('DATAFILES')) / 'autosolve'
            session_count = 0
            behavior_count = 0
            
            # Handle ZIP format
            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zf:
                    # Extract sessions
                    sessions_dir = base_dir / 'sessions'
                    sessions_dir.mkdir(parents=True, exist_ok=True)
                    
                    for name in zf.namelist():
                        if name.startswith('sessions/') and name.endswith('.json'):
                            data = zf.read(name)
                            out_path = sessions_dir / Path(name).name
                            if not out_path.exists() or not self.merge:
                                out_path.write_bytes(data)
                                session_count += 1
                    
                    # Extract behavior
                    behavior_dir = base_dir / 'behavior'
                    behavior_dir.mkdir(parents=True, exist_ok=True)
                    
                    for name in zf.namelist():
                        if name.startswith('behavior/') and name.endswith('.json'):
                            data = zf.read(name)
                            out_path = behavior_dir / Path(name).name
                            if not out_path.exists() or not self.merge:
                                out_path.write_bytes(data)
                                behavior_count += 1
                    
                    # Import model
                    if 'model.json' in zf.namelist():
                        model_data = json.loads(zf.read('model.json'))
                        self._merge_model(predictor, model_data, self.merge)
                
                predictor._save_model()
                self.report({'INFO'}, 
                    f"Imported {session_count} sessions, {behavior_count} behaviors from {filepath.name}")
            
            # Handle legacy JSON format
            else:
                with open(filepath) as f:
                    import_data = json.load(f)
                
                if import_data.get('export_type') != 'autosolve_training_data':
                    self.report({'ERROR'}, "Invalid training data format")
                    return {'CANCELLED'}
                
                self._merge_model(predictor, import_data, self.merge)
                predictor._save_model()
                
                session_count = import_data.get('global_stats', {}).get('total_sessions', 0)
                self.report({'INFO'}, f"Imported {session_count} sessions from {filepath.name}")
            
            return {'FINISHED'}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Import failed: {str(e)}")
            return {'CANCELLED'}
    
    def _merge_model(self, predictor, import_data: dict, merge: bool):
        """Merge or replace model data."""
        if merge:
            # Merge footage classes
            for cls, data in import_data.get('footage_classes', {}).items():
                if cls not in predictor.model['footage_classes']:
                    predictor.model['footage_classes'][cls] = data
                else:
                    existing = predictor.model['footage_classes'][cls]
                    if data.get('sample_count', 0) > existing.get('sample_count', 0):
                        predictor.model['footage_classes'][cls] = data
            
            # Merge region models
            for region, data in import_data.get('region_models', {}).items():
                if region not in predictor.model['region_models']:
                    predictor.model['region_models'][region] = data
                else:
                    predictor.model['region_models'][region]['total_tracks'] = \
                        predictor.model['region_models'][region].get('total_tracks', 0) + data.get('total_tracks', 0)
                    predictor.model['region_models'][region]['successful_tracks'] = \
                        predictor.model['region_models'][region].get('successful_tracks', 0) + data.get('successful_tracks', 0)
            
            # Merge failure patterns
            for key, data in import_data.get('failure_patterns', {}).items():
                if key not in predictor.model.get('failure_patterns', {}):
                    if 'failure_patterns' not in predictor.model:
                        predictor.model['failure_patterns'] = {}
                    predictor.model['failure_patterns'][key] = data
            
            # Update global stats
            stats = import_data.get('global_stats', {})
            predictor.model['global_stats']['total_sessions'] += stats.get('total_sessions', 0)
            predictor.model['global_stats']['successful_sessions'] += stats.get('successful_sessions', 0)
        else:
            # Replace entire model
            predictor.model['footage_classes'] = import_data.get('footage_classes', {})
            predictor.model['region_models'] = import_data.get('region_models', {})
            predictor.model['failure_patterns'] = import_data.get('failure_patterns', {})
            predictor.model['global_stats'] = import_data.get('global_stats', {
                'total_sessions': 0,
                'successful_sessions': 0
            })


class AUTOSOLVE_OT_reset_training_data(Operator):
    """Reset all training data to defaults."""
    
    bl_idname = "autosolve.reset_training_data"
    bl_label = "Reset Training Data"
    bl_options = {'REGISTER', 'UNDO'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)
    
    def execute(self, context):
        try:
            from .solver.learning.settings_predictor import SettingsPredictor
            
            predictor = SettingsPredictor()
            
            # Reset to empty
            predictor.model = {
                'version': 1,
                'footage_classes': {},
                'region_models': {},
                'failure_patterns': {},
                'footage_type_adjustments': {},
                'global_stats': {
                    'total_sessions': 0,
                    'successful_sessions': 0,
                }
            }
            predictor._save_model()
            
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
            from .solver.learning.settings_predictor import SettingsPredictor
            
            predictor = SettingsPredictor()
            stats = predictor.get_stats()
            
            sessions = stats.get('total_sessions', 0)
            classes = stats.get('footage_classes_known', 0)
            success_rate = stats.get('success_rate', 0)
            regions = stats.get('regions_analyzed', 0)
            
            self.report({'INFO'}, 
                f"Sessions: {sessions} | Classes: {classes} | Success: {success_rate:.0%} | Regions: {regions}")
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

