# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve ML Model Exporter to Numpy weights format.
Converts trained model weights to pure JSON arrays for numpy-only inference.
Uses only standard python libraries to avoid dependencies during export.
"""

import os
import json
import argparse
import random


def generate_prebaked_default() -> dict:
    """Generate a default set of weights with simple heuristic properties using standard library random."""
    print("Generating pre-baked default model weights (no-dependency mode)...")
    
    # 15 input features, 64 hidden, 32 hidden, 1 output
    # Setup simple weights that penalize jitter and acceleration
    layer1_w = []
    for i in range(64):
        row = []
        for j in range(15):
            val = 0.0
            # Penalize velocity_x_std (2) and velocity_y_std (3)
            if i == 0 and j in (2, 3):
                val = -1.0
            # Penalize accel_x_mean (4) and accel_y_mean (5)
            elif i == 1 and j in (4, 5):
                val = -1.0
            # Penalize dir_change_x (6) and dir_change_y (7)
            elif i == 2 and j in (6, 7):
                val = -0.5
            # Benefit frames_alive (8)
            elif i == 3 and j == 8:
                val = 0.1
            # Benefit distance to nearest neighbor (9)
            elif i == 4 and j == 9:
                val = 0.5
                
            # Add small random noise
            val += random.normalvariate(0.0, 0.05)
            row.append(val)
        layer1_w.append(row)
        
    layer1_b = [0.0] * 64
    
    layer2_w = []
    for _ in range(32):
        row = [random.normalvariate(0.0, 0.1) for _ in range(64)]
        layer2_w.append(row)
    layer2_b = [0.0] * 32
    
    layer3_w = []
    row = [random.normalvariate(0.0, 0.1) for _ in range(32)]
    layer3_w.append(row)
    
    # Sigmoid(1.38) approx 0.8 default survival probability
    layer3_b = [1.38]
    
    input_mean = [0.0] * 15
    input_std = [1.0] * 15
    
    return {
        "layer1_weight": layer1_w,
        "layer1_bias": layer1_b,
        "layer2_weight": layer2_w,
        "layer2_bias": layer2_b,
        "layer3_weight": layer3_w,
        "layer3_bias": layer3_b,
        "input_mean": input_mean,
        "input_std": input_std,
        "activation": "relu"
    }


def export_numpy_model(args):
    """Load PyTorch meta-weights and export to Blender-addon model folder."""
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    if args.input_json and os.path.exists(args.input_json):
        print(f"Loading weights from '{args.input_json}'...")
        with open(args.input_json, 'r') as f:
            meta = json.load(f)
            
        weights = meta["weights"]
        
        # Map PyTorch Sequential keys to layers
        numpy_model = {
            "layer1_weight": weights["network.0.weight"],
            "layer1_bias": weights["network.0.bias"],
            "layer2_weight": weights["network.2.weight"],
            "layer2_bias": weights["network.2.bias"],
            "layer3_weight": weights["network.4.weight"],
            "layer3_bias": weights["network.4.bias"],
            "input_mean": meta["input_mean"],
            "input_std": meta["input_std"],
            "activation": "relu"
        }
        print("Successfully mapped PyTorch weights to numpy-compatible format.")
    else:
        if args.input_json:
            print(f"Input file '{args.input_json}' not found. Falling back to pre-baked defaults.")
        numpy_model = generate_prebaked_default()
        
    # Write to target path
    with open(args.output_path, 'w') as f:
        json.dump(numpy_model, f, indent=4)
        
    print(f"Exported model weights successfully saved to '{args.output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoSolve ML Weights Exporter to Numpy Format")
    parser.add_argument("--input-json", default="ml/runs/track_predictor/model_meta_weights.json", help="Path to PyTorch converted JSON weights")
    parser.add_argument("--output-path", default="autosolve/tracker/models/track_predictor.json", help="Path to save weights for numpy inference in the addon")
    
    parsed_args = parser.parse_args()
    export_numpy_model(parsed_args)
