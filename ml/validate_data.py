# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve ML Dataset Validation Tool.
Validates collected raw dataset JSON samples against schema structures and prints stats.
"""

import os
import json
import argparse
from typing import Dict, Any


def validate_json_schema(data: Dict[str, Any], file_path: str) -> bool:
    """Validate JSON fields against the expected SolveSample schema."""
    filename = os.path.basename(file_path)
    
    # SolveSample root keys
    root_keys = {
        "clip_metadata", "settings", "tracks", 
        "solve_success", "solve_error", "bundle_count", 
        "bundle_ratio", "runtime_seconds"
    }
    if not root_keys.issubset(data.keys()):
        missing = root_keys - data.keys()
        print(f"[{filename}] Invalid SolveSample: Missing root keys: {missing}")
        return False
        
    # ClipMetadata keys
    meta = data["clip_metadata"]
    meta_keys = {"clip_name", "width", "height", "fps", "frame_count"}
    if not meta_keys.issubset(meta.keys()):
        missing = meta_keys - meta.keys()
        print(f"[{filename}] Invalid ClipMetadata: Missing keys: {missing}")
        return False
        
    # TrackingSettings keys
    settings = data["settings"]
    settings_keys = {
        "quality_preset", "footage_type", "robust_mode", 
        "tripod_mode", "pattern_size", "search_size", 
        "correlation", "threshold", "motion_model"
    }
    if not settings_keys.issubset(settings.keys()):
        missing = settings_keys - settings.keys()
        print(f"[{filename}] Invalid TrackingSettings: Missing keys: {missing}")
        return False
        
    # TrackSample keys
    tracks = data["tracks"]
    track_keys = {
        "track_name", "region", "positions", "velocities", 
        "jitter_scores", "lifespan", "survived", "has_bundle", 
        "average_error"
    }
    for idx, t in enumerate(tracks):
        if not track_keys.issubset(t.keys()):
            missing = track_keys - t.keys()
            print(f"[{filename}] Invalid TrackSample at index {idx}: Missing keys: {missing}")
            return False
            
    return True


def run_validation(args):
    """Validate all JSON files in the dataset folder and print statistical summary."""
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
        
    json_files = [
        os.path.join(args.data_dir, f) 
        for f in os.listdir(args.data_dir) 
        if f.endswith('.json') and not f.endswith('_video_meta.json')
    ]
    
    total_files = len(json_files)
    if total_files == 0:
        print(f"No JSON dataset files found in '{args.data_dir}'")
        return
        
    print(f"Validating {total_files} solve sample files...\n")
    
    valid_count = 0
    invalid_count = 0
    
    # Statistical aggregates
    total_solved = 0
    total_failed = 0
    solve_errors = []
    runtimes = []
    
    total_tracks_count = 0
    survived_tracks_count = 0
    failed_tracks_count = 0
    bundle_tracks_count = 0
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if validate_json_schema(data, file_path):
                valid_count += 1
                
                # Gather stats from valid sample
                solve_success = data["solve_success"]
                if solve_success:
                    total_solved += 1
                    solve_errors.append(data["solve_error"])
                else:
                    total_failed += 1
                    
                runtimes.append(data["runtime_seconds"])
                
                # Gather track stats
                for track in data["tracks"]:
                    total_tracks_count += 1
                    if track["survived"]:
                        survived_tracks_count += 1
                    else:
                        failed_tracks_count += 1
                        
                    if track["has_bundle"]:
                        bundle_tracks_count += 1
            else:
                invalid_count += 1
                
        except Exception as e:
            print(f"Exception while loading {os.path.basename(file_path)}: {e}")
            invalid_count += 1
            
    # Print summary report
    print("═" * 45)
    print(" AutoSolve ML Dataset Summary Report")
    print("═" * 45)
    print(f"Valid Files:   {valid_count} / {total_files}")
    print(f"Invalid Files: {invalid_count} / {total_files}")
    
    if valid_count > 0:
        print("\nSolve Outcomes:")
        solve_rate = (total_solved / valid_count) * 100
        print(f"  Solved:            {total_solved} ({solve_rate:.1f}%)")
        print(f"  Failed:            {total_failed} ({100 - solve_rate:.1f}%)")
        if solve_errors:
            avg_err = sum(solve_errors) / len(solve_errors)
            print(f"  Avg Solve Error:   {avg_err:.2f} px")
        if runtimes:
            avg_time = sum(runtimes) / len(runtimes)
            print(f"  Avg Runtime:       {avg_time:.1f} s")
            
        print("\nTrack Statistics:")
        print(f"  Total Tracks:      {total_tracks_count}")
        if total_tracks_count > 0:
            survived_rate = (survived_tracks_count / total_tracks_count) * 100
            failed_rate = (failed_tracks_count / total_tracks_count) * 100
            bundle_rate = (bundle_tracks_count / total_tracks_count) * 100
            print(f"  Survived Tracks:   {survived_tracks_count} ({survived_rate:.1f}%) [Positive Samples]")
            print(f"  Filtered Tracks:   {failed_tracks_count} ({failed_rate:.1f}%) [Negative Samples]")
            print(f"  3D Bundles (Z!=0):  {bundle_tracks_count} ({bundle_rate:.1f}%)")
    print("═" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoSolve ML Dataset Validation Tool")
    parser.add_argument("--data-dir", default="ml/data/raw", help="Directory containing JSON files to validate")
    parsed_args = parser.parse_args()
    
    run_validation(parsed_args)
