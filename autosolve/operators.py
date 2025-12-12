# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
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

# ═══════════════════════════════════════════════════════════════════════════
# MULTI-CLIP STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════
# Users can work on multiple clips in one session. Instead of global recorders,
# we now use ClipStateManager to isolate state per-clip.
#
# Legacy globals (kept for backward compat, but prefer ClipStateManager):
_behavior_recorder = None

_last_solve_settings = None
_last_solve_error = None
_last_solve_footage_class = None
_last_solve_clip_fingerprint = None  # Track which clip the behavior belongs to
_last_solve_session_id = None  # For linking behaviors to previous sessions

# Iteration tracking per clip (fingerprint -> iteration count)
_clip_iteration_count: dict = {}

# Pending behavior to learn from AFTER the new solve completes
_pending_behavior = None
_pending_behavior_footage_class = None


def _get_clip_manager():
    """Get the ClipStateManager for per-clip state isolation."""
    try:
        from .clip_state import get_clip_manager
        return get_clip_manager()
    except ImportError:
        return None


def _generate_clip_fingerprint(clip):
    """Generate privacy-safe fingerprint for clip identification."""
    if not clip:
        return ""
    try:
        import hashlib
        data = f"{clip.size[0]}x{clip.size[1]}_{clip.fps}_{clip.frame_duration}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]
    except Exception:
        return ""


class AUTOSOLVE_OT_run_solve(Operator):
    """Adaptive learning auto-tracking."""
    
    bl_idname = "autosolve.run_solve"
    bl_label = "Analyze & Solve"
    bl_description = "Full automatic tracking: analyzes footage, detects features, and solves camera (clears existing tracks)"
    bl_options = {'REGISTER'}
    
    _timer = None
    
    # Tracking parameters
    SEGMENT_SIZE = 30  # Longer segments before checking
    MIN_TRACKS = 25
    MIN_LIFESPAN = 5   # Minimum frames for a track to count
    
    @classmethod
    def poll(cls, context):
        # Safe access to edit_movieclip (it may not exist in some contexts)
        clip = getattr(context, "edit_movieclip", None)
        if clip is None:
            return False
        return not context.scene.autosolve.is_solving
    
    def execute(self, context):
        clip = context.edit_movieclip
        settings = context.scene.autosolve
        
        if clip.frame_duration < 10:
            self.report({'ERROR'}, "Clip must have at least 10 frames")
            return {'CANCELLED'}
        
        from .tracker.smart_tracker import SmartTracker, sync_scene_to_clip
        
        robust = getattr(settings, 'robust_mode', False)
        footage_type = getattr(settings, 'footage_type', 'AUTO')
        quality_preset = getattr(settings, 'quality_preset', 'BALANCED')
        tripod_mode = getattr(settings, 'tripod_mode', False)
        _state.reset()
        
        # ═══════════════════════════════════════════════════════════════
        # BEHAVIOR LEARNING: Capture and learn from user edits since last solve
        # ═══════════════════════════════════════════════════════════════
        global _behavior_recorder
        global _last_solve_settings, _last_solve_error, _last_solve_footage_class
        global _pending_behavior, _pending_behavior_footage_class, _last_solve_clip_fingerprint
        
        # Generate fingerprint for current clip
        current_fingerprint = _generate_clip_fingerprint(clip)
        
        if settings.record_edits:
            # Check if user switched clips using fingerprint comparison
            clip_changed = (
                _last_solve_clip_fingerprint and 
                current_fingerprint != _last_solve_clip_fingerprint
            )
            
            if clip_changed:
                print(f"AutoSolve: Clip changed ({_last_solve_clip_fingerprint[:8]} -> {current_fingerprint[:8]})")
                # Save behavior for OLD clip before switching
                if _behavior_recorder and _behavior_recorder.is_monitoring:
                    behavior = _behavior_recorder.stop_monitoring(None, None)
                    if behavior:
                        _behavior_recorder.save_behavior(behavior)
                        print(f"AutoSolve: Saved behavior for previous clip")
                
                # Update ClipStateManager
                manager = _get_clip_manager()
                if manager:
                    manager.set_current_clip(clip)
                
                _behavior_recorder = None
                _last_solve_footage_class = None
            
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
        _last_solve_settings = None
        _last_solve_error = None
        _last_solve_clip_fingerprint = current_fingerprint  # Track current clip
        
        _state.tracker = SmartTracker(
            clip, 
            robust_mode=robust, 
            footage_type=footage_type,
            quality_preset=quality_preset,
            tripod_mode=tripod_mode
        )
        
        # Set session linkage for multi-attempt analysis
        # (iteration is tracked per clip_fingerprint, previous_session_id links to prior attempt)
        # Note: current_fingerprint already generated at line 125
        _state.tracker.iteration = _clip_iteration_count.get(current_fingerprint, 1)
        _state.tracker.previous_session_id = _last_solve_session_id or ""
        
        _state.frame_start = clip.frame_start
        _state.frame_end = clip.frame_start + clip.frame_duration - 1
        _state.frame_current = clip.frame_start
        _state.segment_start = clip.frame_start
        _state.tripod_mode = tripod_mode
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
            # PHASE: CONFIGURE
            # ═══════════════════════════════════════════════════════════════
            if _state.phase == 'LOAD_LEARNING':
                # Learning data is loaded automatically in SmartTracker __init__
                _state.phase = 'CONFIGURE'
                return {'RUNNING_MODAL'}
            
            elif _state.phase == 'CONFIGURE':
                settings.solve_status = f"Configuring (iteration {_state.iteration + 1})..."
                settings.solve_progress = 0.03
                
                # Configure optimal tracker settings
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
                    return self._finish(context, success=False)
                
                _state.phase = 'DETECT'
                # Use optimal start frame (middle of clip for bidirectional tracking)
                _state.optimal_start = tracker.get_optimal_start_frame()
                _state.frame_current = _state.optimal_start
                context.scene.frame_set(_state.optimal_start)
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: SMART DETECT (Unified detection with learning)
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'DETECT':
                settings.solve_progress = 0.05
                settings.solve_status = "Detecting features..."
                
                # Unified smart detection (uses cached probe, learned settings)
                # Calculate markers_per_region from quality-based target_tracks
                # 9 regions, so target_tracks / 9 gives markers per region
                markers_per_region = max(2, tracker.target_tracks // 9)
                num = tracker.detect_features_smart(
                    markers_per_region=markers_per_region,
                    use_cached_probe=(_state.iteration > 0)  # Cache on retry
                )
                
                if num < 8:
                    self.report({'WARNING'}, f"Only {num} features detected")
                    if _state.iteration < tracker.MAX_ITERATIONS:
                        _state.phase = 'RETRY_DECISION'
                    else:
                        # Record failure before cancelling
                        tracker.save_session_results(success=False, solve_error=999.0)
                        return self._finish(context, success=False)
                    return {'RUNNING_MODAL'}
                
                print(f"AutoSolve: Smart detection - {num} balanced markers")
                tracker.select_all_tracks()
                
                # Prefetch frames forward from optimal start for faster tracking
                tracker.prefetch_frames(from_frame=_state.frame_current)
                
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
                    
                    # Reset cache for direction change and prefetch backward
                    tracker.reset_cache_window()
                    tracker.prefetch_frames(from_frame=_state.frame_current, backwards=True)
                    
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
                    
                    # Smart cache management: called every frame, internally decides if refresh needed
                    tracker.prefetch_frames(from_frame=_state.frame_current, backwards=False)
                    
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                else:
                    # Forward tracking complete, now track BACKWARD from END to cover all markers
                    _state.phase = 'TRACK_BACKWARD'
                    # Start from frame_end (not optimal_start) to ensure markers added
                    # during forward pass are fully covered backward
                    _state.frame_current = _state.frame_end
                    context.scene.frame_set(_state.frame_current)
                    
                    # Reset cache for direction change and prefetch backward
                    tracker.reset_cache_window()
                    tracker.prefetch_frames(from_frame=_state.frame_end, backwards=True)
                    
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
                    
                    # Prefetch frames backward from end for faster backward tracking
                    tracker.prefetch_frames(from_frame=_state.frame_end, backwards=True)
                    
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
                    
                    # Smart cache management: called every frame, internally decides if refresh needed
                    tracker.prefetch_frames(from_frame=_state.frame_current, backwards=True)
                    
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                else:
                    # Bidirectional tracking complete!
                    # Cleanup and go directly to ANALYZE (skip MID_REPLENISH and other phases)
                    tracker.cleanup_tracks()
                    _state.phase = 'ANALYZE'
                    return {'RUNNING_MODAL'}
            
            # Adaptive monitoring now handles replenishment during tracking
            
            
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
            
            # Real-time adaptive monitoring handles coverage gaps
            
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
                        from .tracker.learning.failure_diagnostics import FailureDiagnostics
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
                            
                            # Record failure diagnostics for session data (ML training)
                            if hasattr(tracker, 'recorder') and tracker.recorder:
                                tracker.recorder.record_failure_diagnostics(
                                    diagnosis.pattern.value,
                                    _state.last_analysis.get('frame_of_failure')
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
                # Use quality-based min_lifespan from tracker
                tracker.cleanup_tracks(
                    min_frames=tracker.min_lifespan,
                    spike_multiplier=8.0,
                    jitter_threshold=0.6,
                    coherence_threshold=0.4
                )
                
                num = len(tracker.tracking.tracks)
                if num < 8:
                    self.report({'ERROR'}, f"Only {num} tracks - footage may be too difficult")
                    # Record failure before cancelling
                    tracker.save_session_results(success=False, solve_error=999.0)
                    return self._finish(context, success=False)
                
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
                
                # Apply pre-solve track smoothing if enabled
                if settings.smooth_tracks:
                    try:
                        from .tracker.smoothing import smooth_track_markers
                        smoothed = smooth_track_markers(
                            tracker.tracking, 
                            settings.track_smooth_factor
                        )
                        if smoothed > 0:
                            print(f"AutoSolve: Pre-solve track smoothing - {smoothed} markers smoothed")
                    except Exception as e:
                        print(f"AutoSolve: Track smoothing failed: {e}")
                
                # Compute pre-solve confidence for ML training
                pre_confidence = tracker.compute_pre_solve_confidence()
                if pre_confidence.get('confidence', 1.0) < 0.4:
                    self.report({'WARNING'}, f"Low solve confidence: {pre_confidence.get('warnings', ['unknown'])}")
                
                success = tracker.solve_camera(tripod_mode=_state.tripod_mode)
                
                if success:
                    _state.phase = 'FILTER_ERROR'
                else:
                    # Solve failed - retry with adjusted settings if iterations remaining
                    if _state.iteration < tracker.MAX_ITERATIONS:
                        print("AutoSolve: Draft solve failed - retrying with adjusted settings")
                        _state.phase = 'RETRY_DECISION'
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
                    # Record failure before cancelling
                    tracker.save_session_results(success=False, solve_error=999.0)
                    return self._finish(context, success=False)
                
                success = tracker.solve_camera(tripod_mode=_state.tripod_mode)
                
                if not success:
                    # Check if this was a quality failure (low bundle ratio)
                    quality_failure = hasattr(tracker, '_solve_quality_failure') and tracker._solve_quality_failure
                    
                    # For quality failures, retry with robust mode (more markers + learned behavior)
                    if quality_failure and not tracker.robust_mode and _state.iteration < tracker.MAX_ITERATIONS:
                        print("AutoSolve: Quality failure detected - retrying with Robust Mode...")
                        self.report({'WARNING'}, "Quality failure - retrying with more markers and learned behavior")
                        
                        # Enable robust mode
                        settings.robust_mode = True
                        tracker.robust_mode = True
                        
                        # Apply robust mode adjustments
                        tracker.current_settings['pattern_size'] = int(
                            tracker.current_settings.get('pattern_size', 15) * 1.4)
                        tracker.current_settings['search_size'] = int(
                            tracker.current_settings.get('search_size', 71) * 1.4)
                        tracker.current_settings['correlation'] = max(
                            0.45, tracker.current_settings.get('correlation', 0.7) - 0.15)
                        tracker.current_settings['motion_model'] = 'Affine'
                        
                        # Reset and restart from detection with more markers
                        _state.iteration += 1
                        tracker.clear_tracks()
                        _state.phase = 'DETECT'
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                    else:
                        # Already tried robust mode or max iterations reached
                        if quality_failure:
                            self.report({'ERROR'}, "Solve failed - check camera focal length and lens distortion")
                        else:
                            self.report({'ERROR'}, "Solve failed - footage may be too difficult (try Robust Mode or adjust settings)")
                        tracker.save_session_results(success=False, solve_error=999.0)
                        return self._finish(context, success=False)
                
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
                
                # Camera setup handled by AUTOSOLVE_OT_setup_scene operator
                
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
            return self._finish(context, success=False)
        
        return {'RUNNING_MODAL'}
    
    def _finish(self, context, success=False):
        # Start behavior monitoring after successful solve
        global _behavior_recorder
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
                from .tracker.learning.settings_predictor import SettingsPredictor
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
        
        # Start behavior monitoring after solve (success OR failure)
        # This captures user edits including fixes after failures
        if settings.record_edits and clip:
            # Store current solve state for comparison when user re-solves
            _last_solve_settings = {
                'pattern_size': clip.tracking.settings.default_pattern_size,
                'search_size': clip.tracking.settings.default_search_size,
                'correlation': clip.tracking.settings.default_correlation_min,
            }
            # Use actual error if available, otherwise mark as failed with high error
            if clip.tracking.reconstruction.is_valid:
                _last_solve_error = clip.tracking.reconstruction.average_error
            else:
                _last_solve_error = 999.0  # Marker for failed solve
            _last_solve_footage_class = tracker.footage_class if tracker else None
            
            # Update ClipStateManager with solve results
            manager = _get_clip_manager()
            if manager:
                manager.update_from_blender(clip, context.scene)
                manager.set_current_clip(clip)
                
                # Store behavior recorder in clip state
                state = manager.get_state(clip)
                state.solve_success = success
                state.has_solve = clip.tracking.reconstruction.is_valid
                state.solve_error = _last_solve_error if _last_solve_error < 999 else 0.0
                state.last_settings = _last_solve_settings.copy()
                state.last_footage_class = _last_solve_footage_class
            
            # Create session ID for linking behavior with session
            from datetime import datetime
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Track iterations per clip (for multi-attempt analysis)
            global _clip_iteration_count, _last_solve_session_id
            current_fingerprint = _last_solve_clip_fingerprint or ""
            _clip_iteration_count[current_fingerprint] = _clip_iteration_count.get(current_fingerprint, 0) + 1
            iteration = _clip_iteration_count[current_fingerprint]
            
            # Previous session ID for linking (what user was editing)
            previous_session_id = _last_solve_session_id or ""
            _last_solve_session_id = session_id  # Store for next iteration
            
            # Start behavior recorder (captures settings changes, track deletions, additions, etc.)
            from .tracker.learning.behavior_recorder import BehaviorRecorder
            from .tracker.utils import get_contributor_id
            _behavior_recorder = BehaviorRecorder()
            _behavior_recorder.start_monitoring(
                clip=clip,
                settings=_last_solve_settings,
                solve_error=_last_solve_error,
                session_id=session_id,
                clip_fingerprint=current_fingerprint,
                previous_session_id=previous_session_id,
                iteration=iteration,
                contributor_id=get_contributor_id()
            )
            

            
            # Store in ClipState for multi-clip support
            if manager:
                state = manager.get_state(clip)
                state.behavior_recorder = _behavior_recorder

            
            status = "success" if success else "failure"
            print(f"AutoSolve: Monitoring for user behavior changes after {status} (iteration {iteration})")
        
        self._cleanup(context)
        return {'FINISHED'} if success else {'CANCELLED'}
    
    def _cancel(self, context):
        self._cleanup(context)
        return {'CANCELLED'}
    
    def _cleanup(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        
        context.scene.autosolve.is_solving = False
        if context.area:
            context.area.tag_redraw()


class AUTOSOLVE_OT_setup_scene(Operator):
    """Smart scene setup with auto floor detection."""
    
    bl_idname = "autosolve.setup_scene"
    bl_label = "Setup Tracking Scene"
    bl_description = "Initialize 3D scene: sets up background, creates camera, and orients floor/origin from tracks"
    bl_options = {'REGISTER', 'UNDO'}
    
    # Floor setup mode
    floor_mode: bpy.props.EnumProperty(
        name="Floor Detection",
        items=[
            ('AUTO', "Auto-detect", "Let AutoSolve find floor tracks automatically"),
            ('MANUAL', "I'll select tracks", "Close this and select 3+ floor tracks first"),
        ],
        default='AUTO',
    )
    
    @classmethod
    def poll(cls, context):
        # Safe access to edit_movieclip
        clip = getattr(context, "edit_movieclip", None)
        if clip is None:
            return False
        return clip.tracking.reconstruction.is_valid
    
    def invoke(self, context, event):
        clip = context.edit_movieclip
        tracking = clip.tracking
        
        # Count selected tracks with bundles
        selected_with_bundle = [t for t in tracking.tracks 
                                if t.select and t.has_bundle]
        
        # If 3+ tracks selected, use them directly (skip popup)
        if len(selected_with_bundle) >= 3:
            return self._setup_with_floor(context, selected_with_bundle)
        
        # Otherwise show popup dialog
        return context.window_manager.invoke_props_dialog(self, width=320)
    
    def draw(self, context):
        layout = self.layout
        
        # Clear header
        box = layout.box()
        box.label(text="How should the floor be set?", icon='ORIENTATION_NORMAL')
        
        # Radio buttons with descriptions
        col = layout.column(align=True)
        col.prop(self, "floor_mode", expand=True)
        
        # Hint
        layout.separator()
        if self.floor_mode == 'MANUAL':
            layout.label(text="Tip: Select 3 tracks on a flat surface", icon='INFO')
    
    def execute(self, context):
        if self.floor_mode == 'MANUAL':
            # User wants to select tracks manually - just cancel
            self.report({'INFO'}, "Select 3+ floor tracks, then click Setup again")
            return {'CANCELLED'}
        
        # Auto mode - use standard Blender setup (robust and proven)
        try:
            bpy.ops.clip.setup_tracking_scene()
            self.report({'INFO'}, "Scene set up successfully")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Setup failed: {str(e)}")
            return {'CANCELLED'}
    
    def _setup_with_floor(self, context, floor_tracks):
        """Setup scene with floor alignment using given tracks."""
        clip = context.edit_movieclip
        tracking = clip.tracking
        
        # Step 1: Select floor tracks and set plane BEFORE creating camera
        for track in tracking.tracks:
            track.select = False
        for track in floor_tracks[:3]:
            track.select = True
        
        try:
            # Set floor plane first - this affects the reconstruction orientation
            bpy.ops.clip.set_plane(plane='FLOOR')
            print(f"AutoSolve: Floor plane set using 3 tracks")
        except Exception as e:
            print(f"AutoSolve: set_plane failed: {e}")
        
        # Step 2: Select ALL floor tracks and set origin at their center
        for track in tracking.tracks:
            track.select = False
        for track in floor_tracks:
            track.select = True
        
        try:
            # set_origin places origin at median of selected bundles
            bpy.ops.clip.set_origin()
            print(f"AutoSolve: Origin set to center of {len(floor_tracks)} floor tracks")
        except Exception as e:
            print(f"AutoSolve: set_origin failed: {e}")
        
        # Step 3: Now setup scene - camera will be created with correct orientation
        bpy.ops.clip.setup_tracking_scene()
        
        self.report({'INFO'}, "Scene set up with floor alignment")
        return {'FINISHED'}




class AUTOSOLVE_OT_contribute_data(Operator):
    """Open the HuggingFace dataset to upload training data."""
    
    bl_idname = "autosolve.contribute_data"
    bl_label = "Contribute Data"
    bl_description = "Open HuggingFace dataset page to upload your training data"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        import webbrowser
        # Using HuggingFace dataset as primary data upload destination
        webbrowser.open("https://huggingface.co/datasets/UsamaSQ/autosolve-telemetry")
        self.report({'INFO'}, "Opened HuggingFace dataset page")
        return {'FINISHED'}


class AUTOSOLVE_OT_export_training_data(Operator):
    """Export training data as ZIP for ML training."""
    
    bl_idname = "autosolve.export_training_data"
    bl_label = "Export Training Data"
    bl_description = "Export sessions, behavior, and model data to a ZIP file for backup or sharing"
    bl_options = {'REGISTER'}
    
    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to export training data",
        default="autosolve_telemetry.zip",
        subtype='FILE_PATH',
    )
    
    filter_glob: bpy.props.StringProperty(
        default="*.zip",
        options={'HIDDEN'},
    )
    
    def invoke(self, context, event):
        # Generate timestamped filename for unique HuggingFace uploads
        import os
        from datetime import datetime
        from pathlib import Path
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"autosolve_telemetry_{timestamp}.zip"
        
        # Use user's home directory as default location
        home = Path.home()
        self.filepath = str(home / filename)
        
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        import json
        import zipfile
        import base64
        from pathlib import Path
        from datetime import datetime
        
        try:
            from .tracker.learning.settings_predictor import SettingsPredictor
            
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
                            
                            # Write session JSON
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
                    'addon_version': '0.1.0',
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
    bl_description = "Import training data from a ZIP file to improve your local tracking model"
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
            from .tracker.learning.settings_predictor import SettingsPredictor
            
            filepath = Path(self.filepath)
            
            if not filepath.exists():
                self.report({'ERROR'}, "File not found")
                return {'CANCELLED'}
            
            print(f"AutoSolve: Importing from {filepath}")
            
            predictor = SettingsPredictor()
            base_dir = Path(bpy.utils.user_resource('DATAFILES')) / 'autosolve'
            session_count = 0
            behavior_count = 0
            
            # Handle ZIP format
            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zf:
                    files_in_zip = zf.namelist()
                    print(f"AutoSolve: ZIP contains {len(files_in_zip)} files: {files_in_zip[:10]}...")
                    
                    # Extract sessions
                    sessions_dir = base_dir / 'sessions'
                    sessions_dir.mkdir(parents=True, exist_ok=True)
                    
                    for name in files_in_zip:
                        if name.startswith('sessions/') and name.endswith('.json'):
                            data = zf.read(name)
                            out_path = sessions_dir / Path(name).name
                            # Always write (merge just affects model, not raw files)
                            out_path.write_bytes(data)
                            session_count += 1
                            print(f"AutoSolve: Imported session {Path(name).name}")
                    
                    # Extract behavior
                    behavior_dir = base_dir / 'behavior'
                    behavior_dir.mkdir(parents=True, exist_ok=True)
                    
                    for name in files_in_zip:
                        if name.startswith('behavior/') and name.endswith('.json'):
                            data = zf.read(name)
                            out_path = behavior_dir / Path(name).name
                            out_path.write_bytes(data)
                            behavior_count += 1
                            print(f"AutoSolve: Imported behavior {Path(name).name}")
                    
                    # Import model
                    if 'model.json' in files_in_zip:
                        model_data = json.loads(zf.read('model.json'))
                        self._merge_model(predictor, model_data, self.merge)
                        print(f"AutoSolve: Merged model data")
                    else:
                        print(f"AutoSolve: No model.json found in ZIP")
                
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
    bl_description = "WARNING: Permanently delete all learned data, sessions, and behavior history. Cannot be undone"
    bl_options = {'REGISTER', 'UNDO'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)
    
    def execute(self, context):
        import shutil
        from pathlib import Path
        from .tracker.utils import get_sessions_dir, get_behavior_dir, get_cache_dir, get_model_path
        
        try:
            deleted_counts = {'sessions': 0, 'behaviors': 0, 'cache': 0}
            failed_counts = {'sessions': 0, 'behaviors': 0, 'cache': 0}
            
            # 1. Clear sessions folder
            sessions_dir = get_sessions_dir()
            print(f"AutoSolve: Clearing sessions from {sessions_dir}")
            if sessions_dir.exists():
                for f in sessions_dir.glob('*.json'):
                    try:
                        f.unlink()
                        deleted_counts['sessions'] += 1
                    except Exception as e:
                        print(f"AutoSolve: Could not delete {f.name}: {e}")
                        failed_counts['sessions'] += 1
                print(f"AutoSolve: Deleted {deleted_counts['sessions']} session files")
            
            # 2. Clear behavior folder
            behavior_dir = get_behavior_dir()
            print(f"AutoSolve: Clearing behaviors from {behavior_dir}")
            if behavior_dir.exists():
                for f in behavior_dir.glob('*.json'):
                    try:
                        f.unlink()
                        deleted_counts['behaviors'] += 1
                    except Exception as e:
                        print(f"AutoSolve: Could not delete {f.name}: {e}")
                        failed_counts['behaviors'] += 1
                print(f"AutoSolve: Deleted {deleted_counts['behaviors']} behavior files")
            
            # 3. Clear probe cache
            cache_dir = get_cache_dir()
            print(f"AutoSolve: Clearing cache from {cache_dir}")
            if cache_dir.exists():
                for f in cache_dir.glob('*.json'):
                    try:
                        f.unlink()
                        deleted_counts['cache'] += 1
                    except Exception as e:
                        print(f"AutoSolve: Could not delete {f.name}: {e}")
                        failed_counts['cache'] += 1
                print(f"AutoSolve: Deleted {deleted_counts['cache']} cache files")
            
            # 4. Reset the model - delete first to avoid any stale data issues
            from .tracker.learning.settings_predictor import SettingsPredictor
            
            # Delete existing model.json first
            model_path = get_model_path()
            print(f"AutoSolve: Deleting model at {model_path}")
            if model_path.exists():
                model_path.unlink()
                print("AutoSolve: Deleted model.json")
            
            predictor = SettingsPredictor()
            
            # Reset to empty (predictor will have loaded pretrained, so overwrite)
            predictor.model = {
                'version': 1,
                'footage_classes': {},
                'region_models': {},
                'failure_patterns': {},
                'footage_type_adjustments': {},
                'global_stats': {
                    'total_sessions': 0,
                    'successful_sessions': 0,
                },
                'behavior_patterns': {},  # Also clear behavior patterns
            }
            predictor._save_model()
            
            total = sum(deleted_counts.values())
            total_failed = sum(failed_counts.values())
            if total_failed > 0:
                self.report({'WARNING'}, f"Reset: {total} deleted, {total_failed} failed (check console)")
            else:
                self.report({'INFO'}, f"Reset complete: {total} files deleted, model cleared")
            
            # Force UI to redraw with new values
            if context.screen:
                for area in context.screen.areas:
                    area.tag_redraw()
            
            return {'FINISHED'}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Reset failed: {str(e)}")
            return {'CANCELLED'}


class AUTOSOLVE_OT_view_training_stats(Operator):
    """View training data statistics."""
    
    bl_idname = "autosolve.view_training_stats"
    bl_label = "View Training Statistics"
    bl_description = "Show current learning stats: number of sessions, known footage types, and success rates"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        try:
            from .tracker.learning.settings_predictor import SettingsPredictor
            
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


class AUTOSOLVE_OT_smooth_tracks(Operator):
    """Smooth track markers to reduce jitter, then re-solve."""
    
    bl_idname = "autosolve.smooth_tracks"
    bl_label = "Smooth Tracks"
    bl_description = "Smooth track markers to reduce jitter, then automatically re-solve camera (preserves floor orientation)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        # Safe access to edit_movieclip
        clip = getattr(context, "edit_movieclip", None)
        if clip is None:
            return False
        return len(clip.tracking.tracks) > 0
    
    def execute(self, context):
        clip = context.edit_movieclip
        settings = context.scene.autosolve
        
        try:
            # ═══════════════════════════════════════════════════════════════
            # 0. Check if camera has baked F-curves (will need re-setup)
            # ═══════════════════════════════════════════════════════════════
            camera_has_fcurves = False
            scene_camera = context.scene.camera
            if scene_camera and scene_camera.animation_data and scene_camera.animation_data.action:
                camera_has_fcurves = True
            
            # ═══════════════════════════════════════════════════════════════
            # 1. Capture Orientation (Floor/Origin) ONLY if solve was oriented
            # ═══════════════════════════════════════════════════════════════
            floor_tracks = []
            origin_track = None
            floor_was_set = False
            
            recon = clip.tracking.reconstruction
            if recon.is_valid:
                # Collect all bundle Z values to determine if floor was actually set
                z_values = []
                for track in clip.tracking.tracks:
                    if track.has_bundle:
                        z_values.append(track.bundle.z)
                        
                        # Check if on Z plane (tolerance 0.05)
                        if abs(track.bundle.z) < 0.05:
                            floor_tracks.append(track.name)
                        
                        # Check if at origin (tolerance 0.05)
                        if track.bundle.length < 0.05:
                            origin_track = track.name
                
                # Heuristic: Floor was set if there's a cluster of tracks at Z~0
                # (at least 3 tracks with Z < 0.05, AND they represent a meaningful portion)
                if len(floor_tracks) >= 3 and len(z_values) > 0:
                    # Check that floor tracks aren't just random - 
                    # min Z should be close to 0 if floor was set
                    min_z = min(z_values)
                    if min_z > -0.1:  # Floor plane is near Z=0
                        floor_was_set = True
                    else:
                        print(f"AutoSolve: Floor not detected (min_z={min_z:.2f}), skipping orientation restore")
                        floor_tracks = []
                            
            # ═══════════════════════════════════════════════════════════════
            # 2. Smooth Tracks (Backend)
            # ═══════════════════════════════════════════════════════════════
            from .tracker.smoothing import smooth_track_markers
            strength = settings.track_smooth_factor
            count = smooth_track_markers(clip.tracking, strength)
            
            if count == 0:
                self.report({'WARNING'}, "No markers were smoothed (tracks too short)")
                return {'CANCELLED'}
                
            # ═══════════════════════════════════════════════════════════════
            # 3. Prevent Learning (Update snapshot)
            # ═══════════════════════════════════════════════════════════════
            global _behavior_recorder
            if _behavior_recorder and _behavior_recorder.is_monitoring:
                _behavior_recorder.update_snapshot(clip)
            
            # ═══════════════════════════════════════════════════════════════
            # 4. Auto-Solve Camera
            # ═══════════════════════════════════════════════════════════════
            try:
                bpy.ops.clip.solve_camera()
            except Exception as e:
                self.report({'ERROR'}, f"Camera solve failed: {e}")
                return {'CANCELLED'}
            
            # ═══════════════════════════════════════════════════════════════
            # 5. Restore Orientation (only if floor was previously set)
            # ═══════════════════════════════════════════════════════════════
            orientation_restored = False
            if clip.tracking.reconstruction.is_valid and floor_was_set:
                # Restore Floor
                if len(floor_tracks) >= 3:
                    for track in clip.tracking.tracks:
                        track.select = False
                    
                    selected_count = 0
                    for name in floor_tracks[:10]:  # Limit to 10 tracks for set_plane
                        t = clip.tracking.tracks.get(name)
                        if t and t.has_bundle:  # Must still have bundle after re-solve
                            t.select = True
                            selected_count += 1
                            
                    if selected_count >= 3:
                        try:
                            bpy.ops.clip.set_plane(plane='FLOOR')
                            orientation_restored = True
                        except Exception as e:
                            print(f"AutoSolve: Failed to restore floor: {e}")
                            
                # Restore Origin
                if origin_track:
                    for track in clip.tracking.tracks:
                        track.select = False
                    
                    t = clip.tracking.tracks.get(origin_track)
                    if t and t.has_bundle:
                        t.select = True
                        try:
                            bpy.ops.clip.set_origin()
                        except Exception as e:
                            print(f"AutoSolve: Failed to restore origin: {e}")
                            
                # Re-select all tracks for convenience
                for track in clip.tracking.tracks:
                    track.select = True
                
            # ═══════════════════════════════════════════════════════════════
            # 6. Update learning snapshot with new solve state
            # ═══════════════════════════════════════════════════════════════
            if _behavior_recorder and _behavior_recorder.is_monitoring:
                if clip.tracking.reconstruction.is_valid:
                    error = clip.tracking.reconstruction.average_error
                    _behavior_recorder.update_snapshot(clip, solve_error=error)
            
            # ═══════════════════════════════════════════════════════════════
            # 7. Report result with appropriate messages
            # ═══════════════════════════════════════════════════════════════
            if not clip.tracking.reconstruction.is_valid:
                self.report({'WARNING'}, f"Smoothed {count} markers, but solve failed")
                return {'FINISHED'}
            
            # Build result message
            msg_parts = [f"Smoothed {count} markers"]
            
            error = clip.tracking.reconstruction.average_error
            msg_parts.append(f"{error:.2f}px")
            
            if orientation_restored:
                msg_parts.append("floor preserved")
            
            # Warn about baked camera if applicable
            if camera_has_fcurves:
                self.report({'WARNING'}, f"{' | '.join(msg_parts)} — Camera has baked animation, run Setup Scene to update")
            else:
                self.report({'INFO'}, " | ".join(msg_parts))

            return {'FINISHED'}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Smoothing failed: {str(e)}")
            return {'CANCELLED'}





# Registration
classes = (
    AUTOSOLVE_OT_run_solve,
    AUTOSOLVE_OT_setup_scene,
    AUTOSOLVE_OT_contribute_data,
    AUTOSOLVE_OT_export_training_data,
    AUTOSOLVE_OT_import_training_data,
    AUTOSOLVE_OT_reset_training_data,
    AUTOSOLVE_OT_view_training_stats,
    AUTOSOLVE_OT_smooth_tracks,
)


@bpy.app.handlers.persistent
def _save_behavior_on_quit(dummy):
    """Save pending behavior data when Blender quits or file is saved."""
    global _behavior_recorder
    
    # First, save behavior from ClipStateManager (all clips)
    try:
        manager = _get_clip_manager()
        if manager:
            manager.clear_all()  # This saves all pending behavior
    except Exception as e:
        print(f"AutoSolve: Error clearing clip state on quit: {e}")
    
    # Also save from legacy global recorder
    if _behavior_recorder and _behavior_recorder.is_monitoring:
        try:
            behavior = _behavior_recorder.stop_monitoring(None, None)
            if behavior:
                _behavior_recorder.save_behavior(behavior)
                print("AutoSolve: Saved behavior data on quit/save")
        except Exception as e:
            print(f"AutoSolve: Error saving behavior on quit: {e}")
        finally:
            _behavior_recorder = None


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Register save handler to capture behavior on quit
    if _save_behavior_on_quit not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(_save_behavior_on_quit)


def unregister():
    # Remove save handler
    if _save_behavior_on_quit in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(_save_behavior_on_quit)
    
    # Reset clip state manager singleton to prevent stale state on addon reload
    try:
        from .clip_state import reset_clip_manager
        reset_clip_manager()
    except ImportError:
        pass
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
