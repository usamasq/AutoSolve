# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve Headless Tracking & Data Collection Script.
Runs inside Blender to track a clip, record trajectory features, and solve camera.
"""

import sys
import os
import json
import time
import argparse
import bpy

# Add project path to sys.path so we can import autosolve
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.schema import ClipMetadata, TrackingSettings, TrackSample, SolveSample, solve_sample_to_dict
from autosolve.tracker.smart_tracker import SmartTracker


def run_headless_solve(args):
    """Load clip, run tracking pipeline, extract features, and save solve report."""
    print(f"AutoSolve HEADLESS: Loading clip '{args.clip}'...")
    
    if not os.path.exists(args.clip):
        print(f"Error: Clip file not found: {args.clip}")
        sys.exit(1)
        
    # Ensure raw output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Load clip in Blender
    clip = bpy.data.movieclips.load(args.clip)
    
    # Configure Scene properties to match clip
    bpy.context.scene.frame_start = clip.frame_start
    bpy.context.scene.frame_end = clip.frame_start + clip.frame_duration - 1
    bpy.context.scene.render.resolution_x = clip.size[0]
    bpy.context.scene.render.resolution_y = clip.size[1]
    
    # Initialize SmartTracker
    tracker = SmartTracker(
        clip,
        robust_mode=args.robust_mode,
        footage_type=args.footage_type,
        quality_preset=args.quality,
        tripod_mode=args.tripod_mode
    )
    
    start_time = time.time()
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 1: CONFIGURE
    # ═══════════════════════════════════════════════════════════════
    print("AutoSolve HEADLESS: Configuring settings...")
    tracker.configure_settings()
    tracker.clear_tracks()
    
    # Save the settings used for serialization
    settings_dict = tracker.current_settings
    settings_sample = TrackingSettings(
        quality_preset=args.quality,
        footage_type=args.footage_type,
        robust_mode=args.robust_mode,
        tripod_mode=args.tripod_mode,
        pattern_size=settings_dict.get('pattern_size', 15),
        search_size=settings_dict.get('search_size', 71),
        correlation=settings_dict.get('correlation', 0.70),
        threshold=settings_dict.get('threshold', 0.30),
        motion_model=settings_dict.get('motion_model', 'LocRot')
    )
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2: DETECT
    # ═══════════════════════════════════════════════════════════════
    print("AutoSolve HEADLESS: Detecting initial features...")
    optimal_start = tracker.get_optimal_start_frame()
    target = tracker.target_tracks
    markers_per_region = max(1, target // 9)
    
    bpy.context.scene.frame_set(optimal_start)
    num_detected = tracker.detect_features_smart(
        markers_per_region=markers_per_region, 
        use_cached_probe=False
    )
    print(f"AutoSolve HEADLESS: Placed {num_detected} initial markers at frame {optimal_start}")
    tracker.select_all_tracks()
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3: TRACK FORWARD
    # ═══════════════════════════════════════════════════════════════
    frame_start = clip.frame_start
    frame_end = clip.frame_start + clip.frame_duration - 1
    
    print(f"AutoSolve HEADLESS: Tracking forward from {optimal_start} to {frame_end}...")
    for frame in range(optimal_start, frame_end):
        bpy.context.scene.frame_set(frame)
        tracker.track_frame(backwards=False)
        # Monitor/replenish every interval
        if (frame + 1) % tracker.MONITOR_INTERVAL == 0:
            tracker.monitor_and_replenish(frame + 1, backwards=False)
            
    # ═══════════════════════════════════════════════════════════════
    # STEP 4: TRACK BACKWARD
    # ═══════════════════════════════════════════════════════════════
    print(f"AutoSolve HEADLESS: Tracking backward from {frame_end} to {frame_start}...")
    tracker.select_all_tracks()
    for frame in range(frame_end, frame_start, -1):
        bpy.context.scene.frame_set(frame)
        tracker.track_frame(backwards=True)
        # Monitor/replenish every interval
        if (frame - 1) % tracker.MONITOR_INTERVAL == 0:
            tracker.monitor_and_replenish(frame - 1, backwards=True)
            
    # ═══════════════════════════════════════════════════════════════
    # STEP 5: HEAL GAPS
    # ═══════════════════════════════════════════════════════════════
    print("AutoSolve HEADLESS: Filtering spikes and healing gaps...")
    tracker.mark_healing_pending(True)
    tracker.filter_motion_spikes(threshold=5.0)
    tracker.clean_bad_segments(max_error=5.0, min_frames=3)
    tracker.extend_lost_tracks()
    healed = tracker.heal_tracks()
    print(f"AutoSolve HEADLESS: Healed {healed} track gaps")
    tracker.mark_healing_pending(False)
    
    # ═══════════════════════════════════════════════════════════════
    # SNAPSHOT TRACKS BEFORE CLEANUP (To capture negative training data!)
    # ═══════════════════════════════════════════════════════════════
    print("AutoSolve HEADLESS: Snapshotting raw tracks for analysis...")
    raw_tracks_data = []
    for track in clip.tracking.tracks:
        coords = []
        for f in range(1, clip.frame_duration + 1):
            marker = track.markers.find_frame(f)
            if marker and not marker.mute:
                coords.append((marker.co[0], marker.co[1]))
        raw_tracks_data.append({
            'name': track.name,
            'coords': coords
        })
        
    # ═══════════════════════════════════════════════════════════════
    # STEP 6: CLEANUP & FILTERING
    # ═══════════════════════════════════════════════════════════════
    print("AutoSolve HEADLESS: Running track cleanup...")
    tracker.cleanup_tracks(
        min_frames=tracker.min_lifespan,
        spike_multiplier=8.0,
        jitter_threshold=0.6,
        coherence_threshold=0.4
    )
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 7: DRAFT SOLVE
    # ═══════════════════════════════════════════════════════════════
    print("AutoSolve HEADLESS: Running draft solve...")
    success = tracker.solve_camera(tripod_mode=args.tripod_mode)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 8: FILTER ERROR
    # ═══════════════════════════════════════════════════════════════
    if success:
        tracker.filter_high_error(max_error=2.0)
        
    # ═══════════════════════════════════════════════════════════════
    # STEP 9: FINAL SOLVE
    # ═══════════════════════════════════════════════════════════════
    print("AutoSolve HEADLESS: Running final solve...")
    tracker.sanitize_tracks_before_solve()
    tracker.select_optimal_keyframes()
    solve_success = tracker.solve_camera(tripod_mode=args.tripod_mode)
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    solve_error = tracker.get_solve_error() if solve_success else 0.0
    bundle_count = tracker.get_bundle_count() if solve_success else 0
    total_tracks = len(clip.tracking.tracks)
    bundle_ratio = bundle_count / total_tracks if total_tracks > 0 else 0.0
    
    print(f"AutoSolve HEADLESS: Solve success: {solve_success}, Error: {solve_error:.2f}px, Bundles: {bundle_count}/{total_tracks}")
    
    # ═══════════════════════════════════════════════════════════════
    # BUILD DATA SAMPLES
    # ═══════════════════════════════════════════════════════════════
    clip_meta = ClipMetadata(
        clip_name=os.path.basename(args.clip),
        width=clip.size[0],
        height=clip.size[1],
        fps=clip.fps,
        frame_count=clip.frame_duration
    )
    
    final_track_names = {t.name for t in clip.tracking.tracks}
    final_tracks_by_name = {t.name: t for t in clip.tracking.tracks}
    
    track_samples = []
    for snapshot in raw_tracks_data:
        name = snapshot['name']
        coords = snapshot['coords']
        
        # Calculate velocities (deltas between consecutive coordinates)
        velocities = []
        for i in range(1, len(coords)):
            velocities.append((coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1]))
            
        # Calculate jitter (difference in velocities)
        jitter_scores = []
        for i in range(1, len(velocities)):
            dv_x = velocities[i][0] - velocities[i-1][0]
            dv_y = velocities[i][1] - velocities[i-1][1]
            jitter_scores.append((dv_x**2 + dv_y**2)**0.5)
            
        survived = name in final_track_names
        has_bundle = False
        average_error = 0.0
        
        if survived:
            t = final_tracks_by_name[name]
            has_bundle = t.has_bundle
            average_error = t.average_error if t.has_bundle else 0.0
            
        # Classify grid region from first coordinate
        region = "center"
        if coords:
            fx, fy = coords[0]
            ry = "top" if fy > 0.66 else ("bottom" if fy < 0.33 else "mid")
            rx = "left" if fx < 0.33 else ("right" if fx > 0.66 else "center")
            region = "center" if (ry == "mid" and rx == "center") else (f"{ry}-{rx}" if ry != "mid" else f"mid-{rx}")
            
        track_samples.append(TrackSample(
            track_name=name,
            region=region,
            positions=coords,
            velocities=velocities,
            jitter_scores=jitter_scores,
            lifespan=len(coords),
            survived=survived,
            has_bundle=has_bundle,
            average_error=average_error
        ))
        
    solve_sample = SolveSample(
        clip_metadata=clip_meta,
        settings=settings_sample,
        tracks=track_samples,
        solve_success=solve_success,
        solve_error=solve_error,
        bundle_count=bundle_count,
        bundle_ratio=bundle_ratio,
        runtime_seconds=total_time
    )
    
    # Save SolveSample to JSON file
    with open(args.output, 'w') as f:
        json.dump(solve_sample_to_dict(solve_sample), f, indent=4)
        
    print(f"AutoSolve HEADLESS: Successfully wrote dataset sample to '{args.output}'")


if __name__ == "__main__":
    # Get arguments after '--'
    try:
        args_idx = sys.argv.index("--")
        script_args = sys.argv[args_idx + 1:]
    except ValueError:
        script_args = []
        
    parser = argparse.ArgumentParser(description="AutoSolve Headless Dataset Generation Script")
    parser.add_argument("--clip", required=True, help="Path to video clip file")
    parser.add_argument("--quality", default="BALANCED", choices=["FAST", "BALANCED", "QUALITY"])
    parser.add_argument("--footage-type", default="AUTO")
    parser.add_argument("--robust-mode", action="store_true")
    parser.add_argument("--tripod-mode", action="store_true")
    parser.add_argument("--output", required=True, help="Path to save output JSON")
    
    parsed_args = parser.parse_args(script_args)
    run_headless_solve(parsed_args)
