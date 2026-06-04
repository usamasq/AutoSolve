# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve ML Trackability Heatmap / Region Weighting Trainer.
Analyzes the collected dataset to compute empirical survival rates per region
for each footage type, outputting a distilled JSON weights table.
"""

import os
import json
import argparse
from typing import Dict, Any


def generate_prebaked_defaults() -> dict:
    """Generate default empirical region weights based on general VFX/tracking guidelines."""
    print("Generating pre-baked default region trackability weights...")
    
    # 9 standard regions
    regions = [
        'top-left', 'top-center', 'top-right',
        'mid-left', 'center', 'mid-right',
        'bottom-left', 'bottom-center', 'bottom-right'
    ]
    
    # Balanced default (flat 1.0 weight)
    flat_weights = {r: 1.0 for r in regions}
    
    defaults = {
        "AUTO": flat_weights.copy(),
        "INDOOR": {
            "top-left": 0.85, "top-center": 0.80, "top-right": 0.85,
            "mid-left": 0.95, "center": 1.00, "mid-right": 0.95,
            "bottom-left": 0.90, "bottom-center": 0.90, "bottom-right": 0.90
        },
        "OUTDOOR": {
            # Skies at top are uniform or cloud-shifting (avoid)
            "top-left": 0.40, "top-center": 0.15, "top-right": 0.40,
            "mid-left": 0.85, "center": 0.95, "mid-right": 0.85,
            "bottom-left": 1.00, "bottom-center": 1.00, "bottom-right": 1.00
        },
        "DRONE": {
            # Horizon/sky at top, fast panning near sides
            "top-left": 0.10, "top-center": 0.05, "top-right": 0.10,
            "mid-left": 0.70, "center": 0.90, "mid-right": 0.70,
            "bottom-left": 1.00, "bottom-center": 1.00, "bottom-right": 1.00
        },
        "HANDHELD": {
            # High motion blur at edges (prefer center stability)
            "top-left": 0.50, "top-center": 0.60, "top-right": 0.50,
            "mid-left": 0.80, "center": 1.00, "mid-right": 0.80,
            "bottom-left": 0.60, "bottom-center": 0.80, "bottom-right": 0.60
        },
        "GIMBAL": {
            # Stabilized: very uniform distribution works well
            "top-left": 0.90, "top-center": 0.90, "top-right": 0.90,
            "mid-left": 0.95, "center": 1.00, "mid-right": 0.95,
            "bottom-left": 0.95, "bottom-center": 0.95, "bottom-right": 0.95
        },
        "ACTION": {
            # Extreme blur on edges, keep to central focal points
            "top-left": 0.30, "top-center": 0.50, "top-right": 0.30,
            "mid-left": 0.70, "center": 1.00, "mid-right": 0.70,
            "bottom-left": 0.40, "bottom-center": 0.70, "bottom-right": 0.40
        },
        "VFX": {
            # VFX plates: tracker grids often fill center/mid regions
            "top-left": 0.80, "top-center": 0.85, "top-right": 0.80,
            "mid-left": 0.95, "center": 1.00, "mid-right": 0.95,
            "bottom-left": 0.90, "bottom-center": 0.95, "bottom-right": 0.90
        },
        "SCREEN": {
            # Screen recordings: static UI at edges (avoid), content in mid/center
            "top-left": 0.20, "top-center": 0.30, "top-right": 0.20,
            "mid-left": 0.60, "center": 1.00, "mid-right": 0.60,
            "bottom-left": 0.20, "bottom-center": 0.40, "bottom-right": 0.20
        },
        "CINEMATIC": {
            # Anamorphic distortion on edges, shallow depth of field (prefer center focus)
            "top-left": 0.40, "top-center": 0.60, "top-right": 0.40,
            "mid-left": 0.85, "center": 1.00, "mid-right": 0.85,
            "bottom-left": 0.50, "bottom-center": 0.80, "bottom-right": 0.50
        }
    }
    
    return defaults


def train_trackability_model(args):
    """Aggregate track survival counts from JSON dataset and export region weights."""
    if not os.path.exists(args.data_dir):
        print(f"Data directory '{args.data_dir}' not found. Generating default pre-baked weights.")
        weights = generate_prebaked_defaults()
    else:
        json_files = [
            os.path.join(args.data_dir, f) 
            for f in os.listdir(args.data_dir) 
            if f.endswith('.json')
        ]
        
        if not json_files:
            print("No JSON files found in data directory. Generating default pre-baked weights.")
            weights = generate_prebaked_defaults()
        else:
            print(f"Aggregating trackability data from {len(json_files)} samples...")
            
            # Structure to hold counts: {footage_type: {region: {"detected": count, "survived": count}}}
            stats = {}
            
            for file_path in json_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    footage_type = data["settings"]["footage_type"]
                    if footage_type not in stats:
                        stats[footage_type] = {}
                        
                    for track in data["tracks"]:
                        region = track["region"]
                        if region not in stats[footage_type]:
                            stats[footage_type][region] = {"detected": 0, "survived": 0}
                            
                        stats[footage_type][region]["detected"] += 1
                        if track["survived"] and track["has_bundle"]:
                            stats[footage_type][region]["survived"] += 1
                except Exception as e:
                    print(f"Error reading {os.path.basename(file_path)}: {e}")
                    
            # Compute empirical ratios
            weights = {}
            prebaked = generate_prebaked_defaults()
            
            # Use pre-baked keys to ensure all footage types are covered
            for f_type in prebaked.keys():
                weights[f_type] = {}
                prebaked_f_type = prebaked[f_type]
                
                # Check if we collected any data for this footage type
                if f_type in stats and stats[f_type]:
                    f_stats = stats[f_type]
                    for region, r_weights in prebaked_f_type.items():
                        if region in f_stats and f_stats[region]["detected"] > 20:
                            # Enough data to compute meaningful empirical probability
                            det = f_stats[region]["detected"]
                            surv = f_stats[region]["survived"]
                            # Survival rate normalized by dividing by the max survival rate in this footage class
                            weights[f_type][region] = surv / det
                        else:
                            # Not enough data, fall back to default guidelines
                            weights[f_type][region] = r_weights
                            
                    # Normalize weights so that the best region in this class has weight 1.0
                    max_w = max(weights[f_type].values())
                    if max_w > 0:
                        for region in weights[f_type]:
                            weights[f_type][region] = round(weights[f_type][region] / max_w, 3)
                else:
                    # No data, use guidelines directly
                    weights[f_type] = prebaked_f_type
                    
            print("Successfully compiled empirical region weights.")
            
    # Save output to presets directory in the addon
    output_dir = os.path.dirname(args.output_json)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_json, 'w') as f:
        json.dump(weights, f, indent=4)
        
    print(f"Region trackability weights successfully saved to '{args.output_json}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoSolve Region Trackability Weights Aggregator")
    parser.add_argument("--data-dir", default="ml/data/raw", help="Directory containing raw JSON samples")
    parser.add_argument("--output-json", default="autosolve/tracker/presets/region_weights.json", help="Path to save region weights inside the addon presets")
    
    parsed_args = parser.parse_args()
    train_trackability_model(parsed_args)
