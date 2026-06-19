# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve Operators - Adaptive Learning Pipeline.

Learns from tracking failures and improves over iterations.
"""

import bpy
from bpy.types import Operator
from .worker.client import is_port_in_use, start_worker, stop_worker, send_worker_command_async, poll_request_status


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
        self.sam2_masks = None
        
        # Report fields
        self.start_time = 0.0
        self.markers_detected = 0
        self.survived_forward = 0
        self.survived_backward = 0
        self.after_cleanup = 0
        self.gaps_healed = 0


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


def prepare_solve_data(clip):
    recon = clip.tracking.reconstruction
    
    # Map tracks to point indices
    point_idx = 0
    track_idx_to_point_idx = {}
    init_points = []
    
    for idx, track in enumerate(clip.tracking.tracks):
        # We only optimize tracks that have a bundle from the draft solve
        if track.has_bundle:
            init_points.append([track.bundle[0], track.bundle[1], track.bundle[2]])
            track_idx_to_point_idx[idx] = point_idx
            point_idx += 1
            
    # Gather observations
    obs_data = []
    for idx, track in enumerate(clip.tracking.tracks):
        p_idx = track_idx_to_point_idx.get(idx)
        if p_idx is None:
            continue
        for marker in track.markers:
            if not marker.mute:
                frame_idx = marker.frame - clip.frame_start
                # uv is normalized [0, 1]
                obs_data.append({
                    "frame": int(frame_idx),
                    "point_idx": int(p_idx),
                    "uv": [float(marker.co.x), float(marker.co.y)]
                })
                
    # Gather cameras
    init_cameras = []
    for f_idx in range(clip.frame_duration):
        scene_frame = clip.frame_start + f_idx
        camera = recon.cameras.find_frame(frame=scene_frame)
        if camera:
            # camera.matrix is a 4x4 matrix
            matrix_list = [list(camera.matrix[i]) for i in range(4)]
            init_cameras.append(matrix_list)
        else:
            init_cameras.append(None)
            
    return obs_data, init_cameras, init_points, track_idx_to_point_idx


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
        
        if settings.is_solving:
            self.report({'WARNING'}, "AutoSolve tracking is already running")
            return {'CANCELLED'}
            
        if clip.frame_duration < 10:
            self.report({'ERROR'}, "Clip must have at least 10 frames")
            return {'CANCELLED'}
        
        # Check for framerate alignment mismatch (potential Variable Frame Rate or misaligned render settings)
        scene = context.scene
        scene_fps = scene.render.fps / scene.render.fps_base if scene.render.fps_base > 0 else scene.render.fps
        clip_fps = clip.fps
        if abs(scene_fps - clip_fps) > 0.1:
            self.report({'WARNING'}, f"Framerate mismatch! Scene: {scene_fps:.2f} FPS, Clip: {clip_fps:.2f} FPS. Consider matching scene FPS or transcoding clip to Constant Frame Rate to prevent drift.")
        
        # Check for image sequence source block if AI backend is enabled
        if settings.use_external_worker and clip.source == 'SEQUENCE':
            if settings.tracking_backend == 'COTRACKER' or settings.masking_backend == 'SAM2':
                self.report({'ERROR'}, "AI tracking (CoTracker) and masking (YOLO) require a movie file clip. Please use Blender Native tracking or transcode your sequence.")
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
        import time
        _state.start_time = time.time()
        
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
                
                # Route to local AI service if enabled
                if settings.use_external_worker:
                    python_path = settings.external_python_path
                    if not python_path:
                        from .worker.client import detect_system_python
                        python_path = detect_system_python()
                        if python_path:
                            settings.external_python_path = python_path
                            self.report({'INFO'}, f"Auto-detected Python path: {python_path}")
                        else:
                            self.report({'ERROR'}, "Failed to start Local AI Service: Python path not specified.")
                            return self._finish(context, success=False)
                            
                    from .worker.client import check_dependencies
                    if not check_dependencies(python_path):
                        self.report({'ERROR'}, "Required AI packages (torch, scipy, ultralytics, opencv-python) are missing. Please click Install first.")
                        settings.installer_state = 'FAILED'
                        settings.installer_progress = "Required AI packages are missing."
                        return self._finish(context, success=False)

                    if not is_port_in_use():
                        success, msg = start_worker(python_path)
                        if not success:
                            self.report({'ERROR'}, f"Failed to start Local AI Service: {msg}")
                            return self._finish(context, success=False)
                            
                    if settings.masking_backend == 'SAM2':
                        import time
                        send_worker_command_async("sam2_mask", {"video_path": bpy.path.abspath(clip.filepath)})
                        _state.phase = 'WAITING_FOR_WORKER_MASK'
                        _state.request_start_time = time.time()
                        settings.solve_status = "Waiting for YOLO Dynamic Masking..."
                        settings.solve_progress = 0.04
                        if context.area:
                            context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                        
                    elif settings.tracking_backend == 'COTRACKER':
                        import time
                        grid_size = 8
                        if settings.quality_preset == 'FAST':
                            grid_size = 6
                        elif settings.quality_preset == 'QUALITY':
                            grid_size = 10
                        send_worker_command_async("cotrack", {"video_path": bpy.path.abspath(clip.filepath), "grid_size": grid_size})
                        _state.phase = 'WAITING_FOR_WORKER_TRACK'
                        _state.request_start_time = time.time()
                        settings.solve_status = "Waiting for CoTracker AI tracking..."
                        settings.solve_progress = 0.05
                        if context.area:
                            context.area.tag_redraw()
                        return {'RUNNING_MODAL'}

                _state.phase = 'DETECT'
                # Use optimal start frame (middle of clip for bidirectional tracking)
                _state.optimal_start = tracker.get_optimal_start_frame()
                _state.frame_current = _state.optimal_start
                context.scene.frame_set(_state.optimal_start)
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: WAITING FOR DYNAMIC MASK
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'WAITING_FOR_WORKER_MASK':
                import time
                if hasattr(_state, "request_start_time") and (time.time() - _state.request_start_time > 90.0):
                    from .worker.client import kill_worker
                    kill_worker()
                    self.report({'ERROR'}, "YOLO Masking timed out after 90 seconds. Aborting.")
                    return self._finish(context, success=False)
                completed, result, error = poll_request_status()
                if completed:
                    if error:
                        self.report({'ERROR'}, f"YOLO Masking Error: {error}")
                        return self._finish(context, success=False)
                    
                    _state.sam2_masks = result.get("masks", {})
                    print(f"AutoSolve: Received dynamic masks for {len(_state.sam2_masks)} frames.")
                    
                    # Next step: check if CoTracker tracking is enabled
                    if settings.tracking_backend == 'COTRACKER':
                        import time
                        grid_size = 8
                        if settings.quality_preset == 'FAST':
                            grid_size = 6
                        elif settings.quality_preset == 'QUALITY':
                            grid_size = 10
                        send_worker_command_async("cotrack", {"video_path": bpy.path.abspath(clip.filepath), "grid_size": grid_size})
                        _state.phase = 'WAITING_FOR_WORKER_TRACK'
                        _state.request_start_time = time.time()
                        settings.solve_status = "Waiting for CoTracker AI tracking..."
                        settings.solve_progress = 0.08
                    else:
                        # Fallback to standard detect
                        _state.phase = 'DETECT'
                        _state.optimal_start = tracker.get_optimal_start_frame()
                        _state.frame_current = _state.optimal_start
                        context.scene.frame_set(_state.optimal_start)
                else:
                    settings.solve_status = "Running dynamic object masking in background..."
                
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # ═══════════════════════════════════════════════════════════════
            # PHASE: WAITING FOR COTRACKER TRACK
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'WAITING_FOR_WORKER_TRACK':
                import time
                if hasattr(_state, "request_start_time") and (time.time() - _state.request_start_time > 90.0):
                    from .worker.client import kill_worker
                    kill_worker()
                    self.report({'ERROR'}, "CoTracker Tracking timed out after 90 seconds. Aborting.")
                    return self._finish(context, success=False)
                completed, result, error = poll_request_status()
                if completed:
                    if error:
                        self.report({'ERROR'}, f"CoTracker Tracking Error: {error}")
                        return self._finish(context, success=False)
                    
                    trajectories = result.get("trajectories", [])
                    meta = result.get("meta", {})
                    
                    # Apply SAM2 masking filter if available
                    if _state.sam2_masks:
                        filtered_trajectories = []
                        for traj in trajectories:
                            keep = True
                            if traj:
                                x_start, y_start = traj[0]
                                y_start_td = 1.0 - y_start
                                
                                boxes = _state.sam2_masks.get("0", [])
                                for box in boxes:
                                    bx1, by1, bx2, by2 = box
                                    if bx1 <= x_start <= bx2 and by1 <= y_start_td <= by2:
                                        keep = False
                                        break
                            if keep:
                                truncated_traj = []
                                for f_idx, (x, y) in enumerate(traj):
                                    y_td = 1.0 - y
                                    frame_boxes = _state.sam2_masks.get(str(f_idx), [])
                                    hit = False
                                    for box in frame_boxes:
                                        bx1, by1, bx2, by2 = box
                                        if bx1 <= x <= bx2 and by1 <= y_td <= by2:
                                            hit = True
                                            break
                                    if hit:
                                        break
                                    truncated_traj.append((x, y))
                                if len(truncated_traj) >= tracker.min_lifespan:
                                    filtered_trajectories.append(truncated_traj)
                        
                        print(f"AutoSolve: YOLO Masking filtered out {len(trajectories) - len(filtered_trajectories)} of {len(trajectories)} trajectories.")
                        trajectories = filtered_trajectories
                    
                    tracker.import_external_trajectories(trajectories, meta)
                    
                    # Jump directly to filtering
                    _state.phase = 'FILTER_SHORT'
                    settings.solve_progress = 0.50
                else:
                    settings.solve_status = "Running CoTracker AI tracking on GPU/CPU..."
                
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
                
                _state.markers_detected = len(tracker.tracking.tracks)
                _state.phase = 'TRACK_FORWARD'
                _state.segment_start = _state.frame_current
                if context.area:
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
                    
                    # Calculate survived forward
                    clip_frame_end = tracker.scene_to_clip_frame(_state.frame_end)
                    _state.survived_forward = sum(
                        1 for t in tracker.tracking.tracks
                        if (marker := t.markers.find_frame(clip_frame_end)) and not marker.mute
                    )
                    _state.frame_current = _state.frame_end
                    _state.phase = 'TRACK_BACKWARD'
                    # Start 3 FRAMES AFTER optimal_start (into forward-tracked territory) to ensure overlap
                    # This re-tracks existing markers to establish trajectory before hitting new frames
                    optimal_start = getattr(_state, 'optimal_start', _state.frame_start)
                    _state.frame_current = min(optimal_start + 3, _state.frame_end)
                    context.scene.frame_set(_state.frame_current)
                    
                    tracker.select_all_tracks()
                    if context.area:
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
                    # Calculate survived forward
                    clip_frame_end = tracker.scene_to_clip_frame(_state.frame_end)
                    _state.survived_forward = sum(
                        1 for t in tracker.tracking.tracks
                        if (marker := t.markers.find_frame(clip_frame_end)) and not marker.mute
                    )
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
                    
                    # Calculate survived backward
                    clip_frame_start = tracker.scene_to_clip_frame(_state.frame_start)
                    _state.survived_backward = sum(
                        1 for t in tracker.tracking.tracks
                        if (marker := t.markers.find_frame(clip_frame_start)) and not marker.mute
                    )
                    # Go to healing phase
                    _state.phase = 'HEAL_TRACKS'
                    if context.area:
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
                    
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                else:
                    # Calculate survived backward
                    clip_frame_start = tracker.scene_to_clip_frame(_state.frame_start)
                    _state.survived_backward = sum(
                        1 for t in tracker.tracking.tracks
                        if (marker := t.markers.find_frame(clip_frame_start)) and not marker.mute
                    )
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
                _state.gaps_healed = healed if healed else 0
                
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
                
                if context.area:
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
                    if context.area:
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
                
                _state.after_cleanup = len(tracker.tracking.tracks)
                
                # Pre-solve validation
                is_valid, issues = tracker.validate_pre_solve()
                if not is_valid:
                    self.report({'WARNING'}, f"Pre-solve issues: {'; '.join(issues[:2])}")
                
                _state.phase = 'SOLVE_DRAFT'
                if context.area:
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
                    if settings.use_external_worker and settings.solving_backend == 'PRECISION':
                        if not is_port_in_use():
                            success_w, msg_w = start_worker(settings.external_python_path)
                            if not success_w:
                                self.report({'WARNING'}, f"Failed to start worker: {msg_w}. Falling back to native solver.")
                                _state.phase = 'FILTER_ERROR'
                                if context.area:
                                    context.area.tag_redraw()
                                return {'RUNNING_MODAL'}
                        
                        obs, cams, pts, idx_map = prepare_solve_data(clip)
                        _state.track_idx_map = idx_map
                        init_f = clip.tracking.camera.focal_length / clip.tracking.camera.sensor_width
                        
                        send_worker_command_async("precision_solve", {
                            "obs_data": obs,
                            "init_cameras": cams,
                            "init_points": pts,
                            "init_f": init_f,
                            "init_k1": float(clip.tracking.camera.k1),
                            "init_k2": float(clip.tracking.camera.k2)
                        })
                        
                        import time
                        _state.phase = 'WAITING_FOR_WORKER_SOLVE'
                        _state.request_start_time = time.time()
                        settings.solve_status = "Waiting for Precision Bundle Solver..."
                        settings.solve_progress = 0.82
                    else:
                        _state.phase = 'FILTER_ERROR'
                else:
                    # Solve failed - retry with adjusted settings if iterations remaining
                    if _state.iteration < tracker.MAX_ITERATIONS:
                        print("AutoSolve: Draft solve failed - retrying with adjusted settings")
                        _state.phase = 'RETRY_DECISION'
                    else:
                        _state.phase = 'SOLVE_FINAL'
                
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE: WAITING FOR PRECISION SOLVE
            # ═══════════════════════════════════════════════════════════════
            elif _state.phase == 'WAITING_FOR_WORKER_SOLVE':
                import time
                if hasattr(_state, "request_start_time") and (time.time() - _state.request_start_time > 90.0):
                    from .worker.client import kill_worker
                    kill_worker()
                    self.report({'WARNING'}, "Precision Solver timed out after 90 seconds. Falling back to native solve.")
                    _state.phase = 'FILTER_ERROR'
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                
                completed, result, error = poll_request_status()
                if completed:
                    if error:
                        self.report({'WARNING'}, f"Precision Solver error: {error}. Falling back to native solve.")
                        _state.phase = 'FILTER_ERROR'
                    else:
                        cameras_c2w = result.get("cameras", [])
                        points_3d = result.get("points", [])
                        f_opt = result.get("f", 1.0)
                        k1_opt = result.get("k1", 0.0)
                        k2_opt = result.get("k2", 0.0)
                        
                        recon = clip.tracking.reconstruction
                        
                        # 1. Update camera matrices
                        for f_idx in range(clip.frame_duration):
                            scene_frame = clip.frame_start + f_idx
                            camera = recon.cameras.find_frame(frame=scene_frame)
                            if camera and f_idx < len(cameras_c2w) and cameras_c2w[f_idx] is not None:
                                camera.matrix = cameras_c2w[f_idx]
                                
                        # 2. Update track bundles
                        track_idx_map = getattr(_state, "track_idx_map", {})
                        for track_idx, track in enumerate(clip.tracking.tracks):
                            point_idx = track_idx_map.get(track_idx)
                            if point_idx is not None and point_idx < len(points_3d):
                                track.bundle = points_3d[point_idx]
                                track.has_bundle = True
                                
                        # 3. Update camera parameters
                        clip.tracking.camera.focal_length = f_opt * clip.tracking.camera.sensor_width
                        clip.tracking.camera.k1 = k1_opt
                        clip.tracking.camera.k2 = k2_opt
                        
                        # Mark solve success
                        self.report({'INFO'}, f"Precision Solve complete. Focal length: {clip.tracking.camera.focal_length:.2f}mm")
                        _state.phase = 'COMPLETE'
                        settings.solve_progress = 1.0
                else:
                    settings.solve_status = "Running multi-pass least-squares solver in SciPy..."
                
                if context.area:
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
                if context.area:
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
                    # For solve or quality failures, retry with robust mode (more markers + learned behavior)
                    if not tracker.robust_mode and _state.iteration < tracker.MAX_ITERATIONS:
                        print("AutoSolve: Solve failure or quality failure detected - retrying with Robust Mode...")
                        self.report({'WARNING'}, "Solve failed or poor quality - retrying with more markers and learned behavior")
                        
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
                        if context.area:
                            context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                    else:
                        # Already tried robust mode or max iterations reached
                        quality_failure = getattr(tracker, '_solve_quality_failure', False)
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
                    if context.area:
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
                        if context.area:
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
            
            if success:
                import time
                state.report_markers_detected = getattr(_state, "markers_detected", 0)
                state.report_survived_forward = getattr(_state, "survived_forward", 0)
                state.report_survived_backward = getattr(_state, "survived_backward", 0)
                state.report_after_cleanup = getattr(_state, "after_cleanup", 0)
                state.report_gaps_healed = getattr(_state, "gaps_healed", 0)
                state.report_bundles = state.point_count
                state.report_error = state.solve_error
                state.report_total_time = max(0.1, time.time() - getattr(_state, "start_time", time.time()))
            
            manager.sync_to_blender_properties(clip, context.scene)
            
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
    
    def _configure_scene_dimensions(self, context, clip):
        if not clip:
            return
        try:
            # Match render resolution to clip dimensions
            context.scene.render.resolution_x = clip.size[0]
            context.scene.render.resolution_y = clip.size[1]
            
            # Match frame range to clip duration
            context.scene.frame_start = clip.frame_start
            context.scene.frame_end = clip.frame_start + clip.frame_duration - 1
            print(f"AutoSolve: Scene dimensions matched to clip: {clip.size[0]}x{clip.size[1]}, frame range {context.scene.frame_start}-{context.scene.frame_end}")
        except Exception as e:
            print(f"AutoSolve: Failed to configure scene dimensions: {e}")

    def execute(self, context):
        if self.floor_mode == 'MANUAL':
            # User wants to select tracks manually - just cancel
            self.report({'INFO'}, "Select 3+ floor tracks, then click Setup again")
            return {'CANCELLED'}
        
        clip = context.edit_movieclip
        self._configure_scene_dimensions(context, clip)
        
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
        
        self._configure_scene_dimensions(context, clip)
        
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



class AUTOSOLVE_OT_select_high_error(Operator):
    """Select tracks with reprojection error above threshold."""
    bl_idname = "autosolve.select_high_error"
    bl_label = "Select High Error Tracks"
    bl_description = "Select tracks with average reprojection error above the threshold"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        clip = getattr(context, "edit_movieclip", None)
        return clip is not None and len(clip.tracking.tracks) > 0
        
    def execute(self, context):
        clip = context.edit_movieclip
        settings = context.scene.autosolve
        threshold = settings.select_error_threshold
        
        # Deselect all tracks first
        for track in clip.tracking.tracks:
            track.select = False
            
        count = 0
        for track in clip.tracking.tracks:
            if track.has_bundle:
                if track.average_error > threshold:
                    track.select = True
                    count += 1
        self.report({'INFO'}, f"Selected {count} tracks with error > {threshold:.2f}px")
        
        # Force UI redraw
        if context.area:
            context.area.tag_redraw()
        return {'FINISHED'}


class AUTOSOLVE_OT_resolve(Operator):
    """Clean and re-solve camera using existing tracks (no re-tracking)."""
    bl_idname = "autosolve.resolve"
    bl_label = "Clean & Re-Solve"
    bl_description = "Refine existing tracks by filtering outliers and re-solving camera (no new tracking)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        clip = getattr(context, "edit_movieclip", None)
        return clip is not None and len(clip.tracking.tracks) >= 8
        
    def execute(self, context):
        clip = context.edit_movieclip
        settings = context.scene.autosolve
        
        import time
        start_time = time.time()
        
        from .tracker.smart_tracker import SmartTracker
        
        robust = getattr(settings, 'robust_mode', False)
        footage_type = getattr(settings, 'footage_type', 'AUTO')
        quality_preset = getattr(settings, 'quality_preset', 'BALANCED')
        tripod_mode = getattr(settings, 'tripod_mode', False)
        
        # Instantiate SmartTracker
        tracker = SmartTracker(
            clip, 
            robust_mode=robust, 
            footage_type=footage_type,
            quality_preset=quality_preset,
            tripod_mode=tripod_mode
        )
        
        # Configure settings
        tracker.configure_settings()
        
        # Step 1: Cleanup tracks (short tracks, spikes)
        tracker.cleanup_tracks(
            min_frames=tracker.min_lifespan,
            spike_multiplier=8.0,
            jitter_threshold=0.6,
            coherence_threshold=0.4
        )
        
        # Step 2: Smoothing if enabled
        if settings.smooth_tracks:
            try:
                from .tracker.smoothing import smooth_track_markers
                smooth_track_markers(tracker.tracking, settings.track_smooth_factor)
            except Exception as e:
                print(f"AutoSolve: Smooth tracks failed: {e}")
                
        # Step 3: Solve draft to establish errors
        success = tracker.solve_camera(tripod_mode=tripod_mode)
        
        # Step 4: Filter high error tracks (above 2.0px)
        if success:
            tracker.filter_high_error(max_error=2.0)
            
        # Step 5: Sanitize tracks before final solve
        tracker.sanitize_tracks_before_solve()
        
        # Step 6: Select optimal keyframes
        tracker.select_optimal_keyframes()
        
        # Step 7: Final solve
        success = tracker.solve_camera(tripod_mode=tripod_mode)
        
        if not success:
            self.report({'ERROR'}, "Re-solve failed. Not enough tracks or bad configuration.")
            return {'CANCELLED'}
            
        # Get results
        error = tracker.get_solve_error()
        bundles = tracker.get_bundle_count()
        
        # Sync results to scene settings and ClipState
        settings.solve_error = error
        settings.point_count = bundles
        settings.has_solve = True
        
        manager = _get_clip_manager()
        if manager:
            manager.update_from_blender(clip, context.scene)
            state = manager.get_state(clip)
            state.has_solve = True
            state.solve_error = error
            state.point_count = bundles
            
            # Update report metrics
            state.report_markers_detected = len(clip.tracking.tracks)
            state.report_after_cleanup = len(clip.tracking.tracks)
            state.report_bundles = bundles
            state.report_error = error
            state.report_total_time = max(0.1, time.time() - start_time)
            
            manager.sync_to_blender_properties(clip, context.scene)
            
        self.report({'INFO'}, f"Re-solved: {bundles} tracks, {error:.2f}px error")
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL AI ASSISTANT OPERATORS
# ═══════════════════════════════════════════════════════════════════════════

class AUTOSOLVE_OT_install_onnx(bpy.types.Operator):
    """Install onnxruntime into Blender's Python as a fallback for older Blender versions
    that do not support bundled extension wheels."""

    bl_idname  = "autosolve.install_onnx"
    bl_label   = "Install AI Assistant"
    bl_options = {'REGISTER'}

    def execute(self, context):
        self.report({'INFO'}, "AutoSolve: Installing onnxruntime...")

        try:
            from .tracker.onnx_predictor import install_onnx_runtime

            def _progress(msg):
                self.report({'INFO'}, f"AutoSolve: {msg}")

            success = install_onnx_runtime(progress_callback=_progress)

            if success:
                self.report({'INFO'},
                    "AutoSolve: onnxruntime installed! "
                    "Restart Blender once to activate local AI assistant.")
            else:
                self.report({'WARNING'},
                    "AutoSolve: Installation failed — check System Console for details.")

        except Exception as e:
            self.report({'ERROR'}, f"AutoSolve: Unexpected error: {e}")

        return {'FINISHED'}


class AUTOSOLVE_OT_check_turbo_status(bpy.types.Operator):
    """Refresh the Local AI Assistant status panel."""

    bl_idname  = "autosolve.check_turbo_status"
    bl_label   = "Refresh AI Assistant Status"
    bl_options = {'REGISTER'}

    def execute(self, context):
        try:
            from .tracker.onnx_predictor import is_onnx_installed, get_onnx_version, OnnxPredictor
            if is_onnx_installed():
                version = get_onnx_version()
                predictor   = OnnxPredictor.get_instance()
                track_ok    = predictor.track_model_available
                settings_ok = predictor.settings_model_available
                self.report({'INFO'},
                    f"AI Assistant: onnxruntime {version} | "
                    f"Tracking Predictor: {'Active' if track_ok else 'Missing'} | "
                    f"Settings Optimizer: {'Active' if settings_ok else 'Missing'}")
            else:
                self.report({'WARNING'},
                    "AI Assistant: onnxruntime not found. "
                    "Reinstall the addon or check System Console.")
        except Exception as e:
            self.report({'ERROR'}, f"AI Assistant status error: {e}")

        return {'FINISHED'}


class AUTOSOLVE_OT_detect_python(bpy.types.Operator):
    """Auto-detect system Python path."""
    bl_idname = "autosolve.detect_python"
    bl_label = "Auto-Detect Python Path"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.autosolve
        from .worker.client import detect_system_python, check_dependencies
        path = detect_system_python()
        if path:
            settings.external_python_path = path
            self.report({'INFO'}, f"Python path auto-detected: {path}")
            if check_dependencies(path):
                settings.installer_state = 'SUCCESS'
                settings.installer_progress = "AI Packages are already installed."
            else:
                settings.installer_state = 'IDLE'
                settings.installer_progress = "Packages missing. Click Install below."
        else:
            self.report({'WARNING'}, "Could not auto-detect Python on standard paths. Please enter the path manually.")
        return {'FINISHED'}


class AUTOSOLVE_OT_install_deps(bpy.types.Operator):
    """Install required AI packages asynchronously."""
    bl_idname = "autosolve.install_deps"
    bl_label = "Install AI Packages"
    bl_options = {'REGISTER'}

    _timer = None

    def modal(self, context, event):
        if event.type == 'ESC':
            self.report({'INFO'}, "Installation monitoring cancelled. Background process may still be running.")
            return self.cancel_modal(context)

        if event.type == 'TIMER':
            settings = context.scene.autosolve
            from .worker.client import poll_install_status
            completed, success, message = poll_install_status()
            
            # Update progress
            settings.installer_progress = message
            
            if completed:
                if success:
                    settings.installer_state = 'SUCCESS'
                    self.report({'INFO'}, "AI dependencies installed successfully!")
                    try:
                        from .ui import clear_status_cache
                        clear_status_cache()
                    except Exception:
                        pass
                else:
                    settings.installer_state = 'FAILED'
                    self.report({'ERROR'}, f"AI Installation failed: {message}")
                
                # Cleanup timer
                if self._timer:
                    context.window_manager.event_timer_remove(self._timer)
                    self._timer = None
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def cancel_modal(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        settings = context.scene.autosolve
        settings.installer_state = 'FAILED'
        settings.installer_progress = "Installation cancelled."
        return {'CANCELLED'}

    def execute(self, context):
        settings = context.scene.autosolve
        python_path = settings.external_python_path
        
        # If no path specified, try to auto-detect first
        if not python_path:
            from .worker.client import detect_system_python
            python_path = detect_system_python()
            if python_path:
                settings.external_python_path = python_path
                self.report({'INFO'}, f"Auto-detected Python path: {python_path}")
            else:
                self.report({'ERROR'}, "Please specify/detect the External Python Path first.")
                return {'CANCELLED'}

        from .worker.client import run_install_async
        success = run_install_async(python_path)
        
        if success:
            settings.installer_state = 'INSTALLING'
            settings.installer_progress = "Starting installation..."
            
            # Add timer and register modal
            self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Installer is already running.")
            return {'CANCELLED'}


class AUTOSOLVE_OT_start_worker(bpy.types.Operator):
    """Start the background local AI service process."""
    bl_idname = "autosolve.start_worker"
    bl_label = "Start Local AI Service"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.autosolve
        python_path = settings.external_python_path
        
        # Auto-detect if empty
        if not python_path:
            from .worker.client import detect_system_python
            python_path = detect_system_python()
            if python_path:
                settings.external_python_path = python_path
                self.report({'INFO'}, f"Auto-detected Python path: {python_path}")
            else:
                self.report({'ERROR'}, "Please specify/detect the Local Python Path first.")
                return {'CANCELLED'}
            
        from .worker.client import check_dependencies, start_worker
        if not check_dependencies(python_path):
            self.report({'ERROR'}, "Required AI packages (torch, scipy, ultralytics, opencv-python) are missing. Please click Install first.")
            settings.installer_state = 'FAILED'
            settings.installer_progress = "Required AI packages are missing."
            return {'CANCELLED'}
            
        success, msg = start_worker(python_path)
        if success:
            self.report({'INFO'}, f"Local AI Service started: {msg}")
            try:
                from .ui import clear_status_cache
                clear_status_cache()
            except Exception:
                pass
        else:
            self.report({'ERROR'}, f"Failed to start Local AI Service: {msg}")
            
        return {'FINISHED'}


class AUTOSOLVE_OT_stop_worker(bpy.types.Operator):
    """Stop the background local AI service process."""
    bl_idname = "autosolve.stop_worker"
    bl_label = "Stop Local AI Service"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .worker.client import stop_worker
        stop_worker()
        self.report({'INFO'}, "Local AI Service stopped.")
        try:
            from .ui import clear_status_cache
            clear_status_cache()
        except Exception:
            pass
        return {'FINISHED'}


# Registration

classes = (
    AUTOSOLVE_OT_run_solve,
    AUTOSOLVE_OT_setup_scene,
    AUTOSOLVE_OT_smooth_tracks,
    AUTOSOLVE_OT_select_high_error,
    AUTOSOLVE_OT_resolve,
    # Region detection operators
    AUTOSOLVE_OT_detect_inside_annotation,
    AUTOSOLVE_OT_detect_outside_annotation,
    AUTOSOLVE_OT_clear_annotations,
    # Neural Engine
    AUTOSOLVE_OT_install_onnx,
    AUTOSOLVE_OT_check_turbo_status,
    # Worker operators
    AUTOSOLVE_OT_detect_python,
    AUTOSOLVE_OT_install_deps,
    AUTOSOLVE_OT_start_worker,
    AUTOSOLVE_OT_stop_worker,
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
