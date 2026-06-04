# SPDX-FileCopyrightText: 2026 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve ML Settings Optimizer Defaults Exporter.
Grid searches candidate tracking settings using the trained expected reward MLP
and exports recommended preset tables for developer review and manual merge.
"""

import os
import json
import argparse
import math
from typing import List, Dict, Tuple, Any

# Fixed catalogs (matching prepare_dataset.py)
FOOTAGE_TYPES = ['AUTO', 'INDOOR', 'OUTDOOR', 'DRONE', 'HANDHELD', 'GIMBAL', 'ACTION', 'VFX', 'SCREEN', 'CINEMATIC']
MOTION_MODELS = ['Loc', 'LocRot', 'Affine', 'Perspective']
RESOLUTION_CLASSES = ['HD_24fps', 'HD_30fps', 'HD_60fps', '4K_24fps', '4K_30fps']


def get_one_hot(value: str, catalog: List[str]) -> List[float]:
    one_hot = [0.0] * len(catalog)
    if value in catalog:
        one_hot[catalog.index(value)] = 1.0
    else:
        one_hot[0] = 1.0
    return one_hot


def matmul(A: List[List[float]], B_T: List[List[float]]) -> List[List[float]]:
    N = len(A)
    K = len(A[0])
    M = len(B_T)
    out = [[0.0] * M for _ in range(N)]
    for i in range(N):
        for j in range(M):
            s = 0.0
            for k in range(K):
                s += A[i][k] * B_T[j][k]
            out[i][j] = s
    return out


def add_bias(A: List[List[float]], b: List[float]) -> List[List[float]]:
    for i in range(len(A)):
        for j in range(len(b)):
            A[i][j] += b[j]
    return A


def relu(A: List[List[float]]) -> List[List[float]]:
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = max(0.0, A[i][j])
    return A


def sigmoid(A: List[List[float]]) -> List[List[float]]:
    for i in range(len(A)):
        for j in range(len(A[0])):
            val = max(-20.0, min(20.0, A[i][j]))
            A[i][j] = 1.0 / (1.0 + math.exp(-val))
    return A


def predict_batch(X: List[List[float]], weights: Dict[str, Any]) -> List[float]:
    """Runs forward pass on a batch of feature vectors."""
    if not X:
        return []
    # Layer 1
    x1 = matmul(X, weights["network.0.weight"])
    x1 = add_bias(x1, weights["network.0.bias"])
    x1 = relu(x1)
    
    # Layer 2
    x2 = matmul(x1, weights["network.2.weight"])
    x2 = add_bias(x2, weights["network.2.bias"])
    x2 = relu(x2)
    
    # Layer 3
    x3 = matmul(x2, weights["network.4.weight"])
    x3 = add_bias(x3, weights["network.4.bias"])
    x3 = sigmoid(x3)
    
    return [row[0] for row in x3]


def get_resolution_metadata(res_class: str) -> Tuple[float, float, float, float]:
    """Return width, height, fps, frame_count characteristics for resolution classes."""
    meta_map = {
        'HD_24fps': (1920.0, 1080.0, 24.0, 250.0),
        'HD_30fps': (1920.0, 1080.0, 30.0, 250.0),
        'HD_60fps': (1920.0, 1080.0, 60.0, 250.0),
        '4K_24fps': (3840.0, 2160.0, 24.0, 250.0),
        '4K_30fps': (3840.0, 2160.0, 30.0, 250.0),
    }
    return meta_map.get(res_class, (1920.0, 1080.0, 30.0, 250.0))


def export_defaults(args):
    """Perform grid search for optimal settings per clip type and save to review JSON."""
    if not os.path.exists(args.model_path):
        print(f"Model file '{args.model_path}' not found. Please run train_settings_model.py first.")
        return
        
    with open(args.model_path, 'r') as f:
        meta = json.load(f)
        
    means = meta["input_mean"]
    stds = meta["input_std"]
    weights = meta["weights"]
    
    # Define search parameter grid
    grid_patterns = [11, 15, 19, 25, 31, 41, 55]
    grid_searches = [51, 71, 91, 121, 181, 251]
    grid_correlations = [0.55, 0.65, 0.72, 0.80]
    grid_thresholds = [0.15, 0.25, 0.35, 0.45]
    grid_models = ['Loc', 'LocRot', 'Affine']
    
    # Build list of all candidate settings dictionaries (7*6*4*4*3 = 2,016 candidates)
    candidates = []
    for pattern in grid_patterns:
        for search in grid_searches:
            for corr in grid_correlations:
                for thresh in grid_thresholds:
                    for model in grid_models:
                        candidates.append({
                            "pattern_size": pattern,
                            "search_size": search,
                            "correlation": corr,
                            "threshold": thresh,
                            "motion_model": model,
                            "robust_mode": False,
                            "tripod_mode": False
                        })
                        
    print(f"Candidate settings grid size: {len(candidates)}")
    
    recommendations = {}
    
    # Recommend optimal settings for each Resolution class + Footage type combo
    for res_class in RESOLUTION_CLASSES:
        width, height, fps, frame_count = get_resolution_metadata(res_class)
        recommendations[res_class] = {}
        
        for f_type in FOOTAGE_TYPES:
            # Construct batch of feature vectors
            features_batch = []
            for cand in candidates:
                # 1. Clip features (4)
                clip_feats = [width, height, fps, frame_count]
                # 2. Boolean switches (2) (both False for default presets)
                switches = [0.0, 0.0]
                # 3. Settings features (4)
                settings_feats = [
                    float(cand["pattern_size"]),
                    float(cand["search_size"]),
                    cand["correlation"],
                    cand["threshold"]
                ]
                # 4. One-hots (14)
                f_type_oh = get_one_hot(f_type, FOOTAGE_TYPES)
                m_model_oh = get_one_hot(cand["motion_model"], MOTION_MODELS)
                
                raw_vector = clip_feats + switches + settings_feats + f_type_oh + m_model_oh
                # Normalize
                norm_vector = [(val - m) / s for val, m, s in zip(raw_vector, means, stds)]
                features_batch.append(norm_vector)
                
            # Predict expected rewards in a single batch
            rewards = predict_batch(features_batch, weights)
            
            # Find candidate with max reward
            max_idx = rewards.index(max(rewards))
            best_settings = candidates[max_idx].copy()
            best_reward = rewards[max_idx]
            
            # Remove unused keys for preset export
            best_settings.pop("robust_mode")
            best_settings.pop("tripod_mode")
            best_settings["expected_reward"] = round(best_reward, 4)
            
            recommendations[res_class][f_type] = best_settings
            
    # Output to target path
    output_dir = os.path.dirname(args.output_json)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_json, 'w') as f:
        json.dump(recommendations, f, indent=4)
        
    print(f"\nRecommended presets report successfully generated: '{args.output_json}'")
    print("="*80)
    print("SAMPLE RECOMMENDATIONS FOR 'HD_30fps':")
    print("="*80)
    for f_type in ['AUTO', 'INDOOR', 'OUTDOOR', 'DRONE']:
        rec = recommendations['HD_30fps'][f_type]
        print(f"  Footage Type: {f_type:<10} | Pattern: {rec['pattern_size']:<2} | Search: {rec['search_size']:<3} | Corr: {rec['correlation']:.2f} | Thresh: {rec['threshold']:.2f} | Model: {rec['motion_model']:<6} | Reward: {rec['expected_reward']:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoSolve Settings Preset Recommendation Generator")
    parser.add_argument("--model-path", default="ml/runs/settings_optimizer/model_meta_weights.json", help="Path to model meta weights file")
    parser.add_argument("--output-json", default="ml/runs/settings_optimizer/recommended_defaults.json", help="Path to save recommended settings presets")
    
    parsed_args = parser.parse_args()
    export_defaults(parsed_args)
