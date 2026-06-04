# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve Batch Data Collection Runner.
Iterates over video clips in a directory and executes multiple tracking variations
via headless Blender subprocesses.
"""

import sys
import os
import subprocess
import argparse
from typing import List


SUPPORTED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.ogg', '.webm'}


def get_video_files(clips_dir: str) -> List[str]:
    """Find all supported video files in the clips directory."""
    video_files = []
    if not os.path.exists(clips_dir):
        print(f"Error: Clips directory does not exist: {clips_dir}")
        return []
        
    for root, _, files in os.walk(clips_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                video_files.append(os.path.join(root, f))
    return video_files


def run_collection(args):
    """Iterate over clips and execute tracking variations."""
    video_files = get_video_files(args.clips_dir)
    total_clips = len(video_files)
    
    if total_clips == 0:
        print(f"No supported video clips found in '{args.clips_dir}'")
        return
        
    print(f"Found {total_clips} video clips for data collection.")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define settings variations to run per clip
    variations = [
        # (quality, robust_mode, tripod_mode, suffix)
        ("BALANCED", False, False, "balanced_standard"),
        ("FAST", False, False, "fast_standard"),
        ("QUALITY", False, False, "quality_standard"),
        ("BALANCED", True, False, "balanced_robust"),
        ("BALANCED", False, True, "balanced_tripod"),
    ]
    
    total_runs = total_clips * len(variations)
    completed_runs = 0
    failures = 0
    
    print(f"Starting batch tracking collection. Total runs planned: {total_runs}\n")
    
    for idx, video_path in enumerate(video_files):
        clip_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[{idx+1}/{total_clips}] Processing clip: {clip_name}")
        
        for quality, robust, tripod, suffix in variations:
            output_name = f"{clip_name}_{suffix}.json"
            output_path = os.path.join(args.output_dir, output_name)
            
            # Skip if already exists (resume support)
            if os.path.exists(output_path):
                print(f"  -> Skipping existing: {output_name}")
                completed_runs += 1
                continue
                
            cmd = [
                args.blender_bin,
                "--background",
                "--python", "ml/collect_data.py",
                "--",
                "--clip", video_path,
                "--quality", quality,
                "--output", output_path
            ]
            
            if robust:
                cmd.append("--robust-mode")
            if tripod:
                cmd.append("--tripod-mode")
                
            print(f"  -> Running variation {suffix}...")
            
            try:
                # Run headless Blender process
                result = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True,
                    timeout=300 # 5 minutes timeout per run to prevent hanging processes
                )
                
                if result.returncode == 0:
                    print(f"     Success: Saved data to {output_name}")
                else:
                    print(f"     Failed (exit code {result.returncode}): {output_name}")
                    print(f"     Stderr summary: {result.stderr.strip()[-300:]}")
                    failures += 1
                    
            except subprocess.TimeoutExpired:
                print(f"     Timeout: Subprocess hung for {output_name}")
                failures += 1
            except Exception as e:
                print(f"     Error running subprocess: {e}")
                failures += 1
                
            completed_runs += 1
            progress = (completed_runs / total_runs) * 100
            print(f"  Progress: {completed_runs}/{total_runs} ({progress:.1f}%) | Failures: {failures}\n")
            
    print("Batch data collection complete.")
    print(f"Total Runs: {total_runs} | Completed: {completed_runs} | Failures: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoSolve Headless Batch Collection Runner")
    parser.add_argument("--clips-dir", default="ml/clips", help="Directory containing input clips")
    parser.add_argument("--output-dir", default="ml/data/raw", help="Directory to save raw JSON output files")
    parser.add_argument("--blender-bin", default="blender", help="Path to Blender executable")
    
    parsed_args = parser.parse_args()
    run_collection(parsed_args)
