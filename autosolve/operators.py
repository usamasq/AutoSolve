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
# Iteration tracking per clip (fingerprint -> iteration count)
_clip_iteration_count: dict = {}


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
    bl_label = "Auto-Track & Solve"
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
        
        # Generate fingerprint for current clip
        current_fingerprint = _generate_clip_fingerprint(clip)
        
        # Update ClipStateManager
        manager = _get_clip_manager()
        if manager:
            manager.set_current_clip(clip)
        
        _state.tracker = SmartTracker(
            clip, 
            robust_mode=robust, 
            footage_type=footage_type,
            quality_preset=quality_preset,
            tripod_mode=tripod_mode
        )
        
        # Set session linkage for multi-attempt analysis
        _state.tracker.iteration = _clip_iteration_count.get(current_fingerprint, 1)
        
        _state.frame_start = clip.frame_start
        _state.frame_end = clip.frame_start + clip.frame_duration - 1
        _state.frame_current = clip.frame_start
        _state.segment_start = clip.frame_start
        _state.tripod_mode = tripod_mode
        _state.phase = 'CONFIGURE'
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
            if _state.phase == 'CONFIGURE':
                settings.solve_status = f"Configuring (iteration {_state.iteration + 1})..."
                settings.solve_progress = 0.03
                
                # Configure optimal tracker settings
                tracker.configure_settings()
                
                # On retry or when existing tracks exist, preserve good ones
                # On first fresh run, clear all to start clean
                if _state.iteration > 0 or (len(tracker.tracking.tracks) > 0 and _state.iteration == 0):
                    preserved = tracker.preserve_good_tracks()
                    _state.preserved_tracks = preserved
                else:
                    tracker.clear_tracks()
                    _state.preserved_tracks = 0
                
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
                
                # Account for preserved tracks when detecting new ones
                preserved = getattr(_state, 'preserved_tracks', 0)
                existing_tracks = len(tracker.tracking.tracks)
                
                # Calculate markers_per_region from quality-based target_tracks
                # Reduce target if we already have preserved tracks
                effective_target = max(tracker.target_tracks - existing_tracks, 0)
                markers_per_region = max(1, effective_target // 9) if effective_target > 0 else 0
                
                if markers_per_region > 0:
                    num = tracker.detect_features_smart(
                        markers_per_region=markers_per_region,
                        use_cached_probe=(_state.iteration > 0)  # Cache on retry
                    )
                    total_tracks = existing_tracks + num
                    print(f"AutoSolve: Total tracks: {total_tracks} ({existing_tracks} preserved + {num} new)")
                else:
                    num = 0
                    total_tracks = existing_tracks
                    print(f"AutoSolve: Using {existing_tracks} preserved tracks (target already met)")
                
                if total_tracks < 8:
                    self.report({'WARNING'}, f"Only {total_tracks} total tracks")
                    if _state.iteration < tracker.MAX_ITERATIONS:
                        _state.phase = 'RETRY_DECISION'
                    else:
                        # Record failure before cancelling
                        
                        return self._finish(context, success=False)
                    return {'RUNNING_MODAL'}
                
                print(f"AutoSolve: Ready with {total_tracks} tracks")
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
                    
                    if context.area:
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
                if settings.batch_tracking:
                    settings.solve_status = "Batch tracking backward..."
                    settings.solve_progress = 0.55
                    
                    frames = tracker.track_sequence(
                        _state.frame_current,
                        _state.frame_start,
                        backwards=True
                    )
                    print(f"AutoSolve: Batch tracked {frames} frames backward")
                    
                    # Go to healing phase
                    _state.phase = 'HEAL_TRACKS'
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
                    # Go to healing phase
                    _state.phase = 'HEAL_TRACKS'
                    return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: HEAL TRACKS (Filter spikes + Clean segments + Gap healing)
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'HEAL_TRACKS':
                settings.solve_status = "Filtering & healing tracks..."
                settings.solve_progress = 0.62
                
                # Mark healing pending to preserve short tracks (they may be candidates)
                tracker.mark_healing_pending(True)
                
                # Step 1: Use Blender's filter_tracks to detect drift/dislocation
                # This mutes markers with motion spikes
                spikes_found = tracker.filter_motion_spikes(threshold=5.0)
                if spikes_found > 0:
                    print(f"AutoSolve: Filtered {spikes_found} tracks with motion spikes")
                
                # Step 2: Use Blender's clean_tracks with DELETE_SEGMENTS to remove
                # only the bad portions of tracks (creates gaps for healing)
                # Use loose threshold (5.0px) - strict filtering happens in FILTER_ERROR
                segments_cleaned = tracker.clean_bad_segments(max_error=5.0, min_frames=3)
                if segments_cleaned > 0:
                    print(f"AutoSolve: Cleaned bad segments in {segments_cleaned} tracks")
                
                # Step 3: Extend tracks that stopped early (re-track from where they stopped)
                extended = tracker.extend_lost_tracks()
                
                # Step 4: Attempt to heal gaps using anchor-based interpolation
                # (joins separate tracks that represent the same point)
                healed = tracker.heal_tracks()
                if healed > 0:
                    print(f"AutoSolve: Healed {healed} track gaps")
                
                # Don't call cleanup_tracks here - it happens in FILTER_SHORT phase
                # Just mark healing done so short tracks can be filtered
                tracker.mark_healing_pending(False)
                
                _state.phase = 'ANALYZE'
                return {'RUNNING_MODAL'}
            
            
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
                        from .tracker.failure_diagnostics import FailureDiagnostics
                        diagnostics = FailureDiagnostics()
                        diagnosis = diagnostics.diagnose(_state.last_analysis, tracker.current_settings)
                        
                        if diagnosis.confidence > 0.5:
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
                
                # Strict threshold (2.0px) - now we have actual solve errors to compare
                tracker.filter_high_error(max_error=2.0)
                
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
                    
                    return self._finish(context, success=False)
                
                # Select optimal keyframes based on parallax BEFORE solving
                tracker.select_optimal_keyframes()
                
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
                        # Preserve good tracks instead of clearing all
                        _state.iteration += 1
                        preserved = tracker.preserve_good_tracks()
                        _state.preserved_tracks = preserved
                        _state.phase = 'DETECT'
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                    else:
                        # Already tried robust mode or max iterations reached
                        if quality_failure:
                            self.report({'ERROR'}, "Solve failed - check camera focal length and lens distortion")
                        else:
                            self.report({'ERROR'}, "Solve failed - footage may be too difficult (try Robust Mode or adjust settings)")
                        
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
        clip = getattr(context, "edit_movieclip", None)
        if clip is None:
            # Context lost access to clip - use tracker's cached reference
            clip = _state.tracker.clip if _state.tracker else None
            
        # Update ClipStateManager with solve results
        manager = _get_clip_manager()
        if manager and clip:
            manager.update_from_blender(clip, context.scene)
            manager.set_current_clip(clip)
            
            state = manager.get_state(clip)
            state.solve_success = success
            state.has_solve = clip.tracking.reconstruction.is_valid
            state.solve_error = clip.tracking.reconstruction.average_error if clip.tracking.reconstruction.is_valid else 0.0
            
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
                    for name in floor_tracks:
                        t = clip.tracking.tracks.get(name)
                        if t and t.has_bundle:  # Must still have bundle after re-solve
                            t.select = True
                            selected_count += 1
                            if selected_count == 3:
                                break
                            
                    if selected_count == 3:
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



# ═══════════════════════════════════════════════════════════════════════════
# REGION DETECTION OPERATORS (Annotation-based)
# ═══════════════════════════════════════════════════════════════════════════

class AUTOSOLVE_OT_detect_inside_annotation(Operator):
    """Detect features only INSIDE drawn annotation regions."""
    
    bl_idname = "autosolve.detect_inside_annotation"
    bl_label = "Detect Inside"
    bl_description = "Detect tracking markers only inside annotation regions"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.edit_movieclip is not None
    
    def execute(self, context):
        try:
            # Use Blender's detect with placement=INSIDE_GPENCIL
            bpy.ops.clip.detect_features(
                threshold=0.4,
                min_distance=50,
                margin=16,
                placement='INSIDE_GPENCIL'
            )
            self.report({'INFO'}, "Features detected inside annotation")
            return {'FINISHED'}
        except Exception as e:
            self.report({'WARNING'}, f"Detection failed: {e}. Draw an annotation first.")
            return {'CANCELLED'}


class AUTOSOLVE_OT_detect_outside_annotation(Operator):
    """Detect features only OUTSIDE drawn annotation regions."""
    
    bl_idname = "autosolve.detect_outside_annotation"
    bl_label = "Detect Outside"
    bl_description = "Detect tracking markers only outside annotation regions (exclude annotated area)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.edit_movieclip is not None
    
    def execute(self, context):
        try:
            # Use Blender's detect with placement=OUTSIDE_GPENCIL
            bpy.ops.clip.detect_features(
                threshold=0.4,
                min_distance=50,
                margin=16,
                placement='OUTSIDE_GPENCIL'
            )
            self.report({'INFO'}, "Features detected outside annotation (annotated area excluded)")
            return {'FINISHED'}
        except Exception as e:
            self.report({'WARNING'}, f"Detection failed: {e}. Draw an annotation first.")
            return {'CANCELLED'}


class AUTOSOLVE_OT_clear_annotations(Operator):
    """Clear all annotation strokes."""
    
    bl_idname = "autosolve.clear_annotations"
    bl_label = "Clear Annotations"
    bl_description = "Clear all drawn annotations"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        clip = getattr(context, "edit_movieclip", None)
        if clip:
            if hasattr(clip, "annotation") and clip.annotation is not None:
                return True
            if hasattr(clip, "grease_pencil") and clip.grease_pencil is not None:
                return True
        if hasattr(context, "annotation_data") and context.annotation_data is not None:
            return True
        return False
    
    def execute(self, context):
        try:
            # Try clearing layers directly (Blender 4.2, 4.3, 4.4, 5.0, 5.1 safe)
            clip = context.edit_movieclip
            gpd = None
            if clip:
                if hasattr(clip, "annotation"):
                    gpd = clip.annotation
                elif hasattr(clip, "grease_pencil"):
                    gpd = clip.grease_pencil
            
            if gpd is None and hasattr(context, "annotation_data"):
                gpd = context.annotation_data
                
            if gpd and hasattr(gpd, "layers"):
                gpd.layers.clear()
                self.report({'INFO'}, "Annotations cleared")
                return {'FINISHED'}
        except Exception:
            pass

        # Fallback to legacy operator if direct clearing failed/unsupported
        try:
            bpy.ops.gpencil.data_unlink()
            self.report({'INFO'}, "Annotations cleared")
            return {'FINISHED'}
        except Exception as e:
            self.report({'WARNING'}, f"Could not clear: {e}")
            return {'CANCELLED'}



# Registration

classes = (
    AUTOSOLVE_OT_run_solve,
    AUTOSOLVE_OT_setup_scene,
    AUTOSOLVE_OT_smooth_tracks,
    # Region detection operators
    AUTOSOLVE_OT_detect_inside_annotation,
    AUTOSOLVE_OT_detect_outside_annotation,
    AUTOSOLVE_OT_clear_annotations,
)





def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    # Reset clip state manager singleton to prevent stale state on addon reload
    try:
        from .clip_state import reset_clip_manager
        reset_clip_manager()
    except ImportError:
        pass
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
