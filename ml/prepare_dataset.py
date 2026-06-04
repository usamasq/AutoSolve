# SPDX-FileCopyrightText: 2026 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve ML Settings Optimizer Dataset Preparation.
Parses raw JSON solve samples, computes expected rewards, and exports normalized datasets.
"""

import os
import json
import argparse
import random
import math
from typing import Dict, List, Tuple, Any

# Fixed catalog of footage types and motion models for one-hot encoding
FOOTAGE_TYPES = ['AUTO', 'INDOOR', 'OUTDOOR', 'DRONE', 'HANDHELD', 'GIMBAL', 'ACTION', 'VFX', 'SCREEN', 'CINEMATIC']
MOTION_MODELS = ['Loc', 'LocRot', 'Affine', 'Perspective']


def get_one_hot(value: str, catalog: List[str]) -> List[float]:
    """Helper to generate a one-hot encoding list."""
    one_hot = [0.0] * len(catalog)
    if value in catalog:
        one_hot[catalog.index(value)] = 1.0
    else:
        # Fallback to the first item (e.g. AUTO or LocRot)
        one_hot[0] = 1.0
    return one_hot


def extract_sample_features(sample: Dict[str, Any]) -> List[float]:
    """
    Extracts a 24-dimensional feature vector from a SolveSample dictionary.
    Vector structure:
    [0:4] clip: width, height, fps, frame_count
    [4] tripod_mode
    [5] robust_mode
    [6:10] settings: pattern_size, search_size, correlation, threshold
    [10:20] footage_type one-hot
    [20:24] motion_model one-hot
    """
    meta = sample["clip_metadata"]
    settings = sample["settings"]
    
    # 1. Clip features (4)
    clip_feats = [
        float(meta.get("width", 1920)),
        float(meta.get("height", 1080)),
        float(meta.get("fps", 24.0)),
        float(meta.get("frame_count", 250))
    ]
    
    # 2. Boolean switches (2)
    switches = [
        1.0 if settings.get("tripod_mode", False) else 0.0,
        1.0 if settings.get("robust_mode", False) else 0.0
    ]
    
    # 3. Settings features (4)
    setting_feats = [
        float(settings.get("pattern_size", 17)),
        float(settings.get("search_size", 71)),
        float(settings.get("correlation", 0.70)),
        float(settings.get("threshold", 0.30))
    ]
    
    # 4. One-hots (14)
    f_type_oh = get_one_hot(settings.get("footage_type", "AUTO"), FOOTAGE_TYPES)
    m_model_oh = get_one_hot(settings.get("motion_model", "LocRot"), MOTION_MODELS)
    
    return clip_feats + switches + setting_feats + f_type_oh + m_model_oh


def calculate_reward(sample: Dict[str, Any]) -> float:
    """
    Compute solve reward:
    reward = success * (1 - clamp(error/5, 0, 1)) * bundle_ratio
    """
    success = 1.0 if sample.get("solve_success", False) else 0.0
    error = sample.get("solve_error", 10.0)
    bundle_ratio = sample.get("bundle_ratio", 0.0)
    
    # Clamp error contribution
    error_penalty = max(0.0, min(1.0, error / 5.0))
    reward = success * (1.0 - error_penalty) * bundle_ratio
    return round(reward, 4)


def generate_simulated_dataset() -> List[Dict[str, Any]]:
    """Generates a synthetic list of SolveSamples to simulate the data collection pipeline."""
    print("Generating simulated SolveSamples for testing settings optimizer pipeline...")
    samples = []
    
    # Generate 15 fake clips
    clips = []
    for i in range(15):
        f_type = random.choice(FOOTAGE_TYPES)
        w, h = random.choice([(1920, 1080), (3840, 2160), (1280, 720)])
        clips.append({
            "clip_name": f"synthetic_clip_{i:03d}",
            "width": w,
            "height": h,
            "fps": random.choice([23.976, 24.0, 25.0, 29.97, 30.0, 60.0]),
            "frame_count": random.randint(60, 450),
            "footage_type": f_type
        })
        
    # For each clip, try multiple settings variations
    for clip in clips:
        # Expected optimal settings for this footage type to add some signal to the data
        optimal_pattern = 17
        optimal_search = 71
        if clip["width"] >= 3840:
            optimal_pattern = 55
            optimal_search = 231
        if clip["footage_type"] == "DRONE":
            optimal_search = 121
            
        for _ in range(25):  # 25 solve attempts per clip
            pattern = random.choice([11, 15, 17, 21, 31, 55])
            search = random.choice([51, 71, 91, 121, 231])
            corr = random.uniform(0.45, 0.85)
            thresh = random.uniform(0.1, 0.5)
            m_model = random.choice(MOTION_MODELS)
            tripod = random.random() < 0.1
            robust = random.random() < 0.3
            
            # Calculate distance from optimal settings
            dist = abs(pattern - optimal_pattern) / 50.0 + abs(search - optimal_search) / 200.0
            
            # Higher distance = lower probability of success
            success_prob = max(0.1, 0.85 - dist)
            success = random.random() < success_prob
            
            error = 99.0
            bundle_ratio = 0.0
            if success:
                error = max(0.1, random.normalvariate(0.6 + dist * 2.0, 0.3))
                bundle_ratio = max(0.2, min(0.95, random.uniform(0.4, 0.9) - dist * 0.3))
                
            sample = {
                "clip_metadata": {
                    "clip_name": clip["clip_name"],
                    "width": clip["width"],
                    "height": clip["height"],
                    "fps": clip["fps"],
                    "frame_count": clip["frame_count"]
                },
                "settings": {
                    "quality_preset": random.choice(["FAST", "BALANCED", "QUALITY"]),
                    "footage_type": clip["footage_type"],
                    "robust_mode": robust,
                    "tripod_mode": tripod,
                    "pattern_size": pattern,
                    "search_size": search,
                    "correlation": corr,
                    "threshold": thresh,
                    "motion_model": m_model
                },
                "solve_success": success,
                "solve_error": error,
                "bundle_count": int(bundle_ratio * 40),
                "bundle_ratio": bundle_ratio,
                "runtime_seconds": random.uniform(5.0, 60.0)
            }
            samples.append(sample)
            
    return samples


def prepare_dataset(args):
    """Load, split, normalize, and save the settings optimizer dataset."""
    raw_samples = []
    
    if os.path.exists(args.data_dir):
        json_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.json')]
        for fp in json_files:
            try:
                with open(fp, 'r') as f:
                    raw_samples.append(json.load(f))
            except Exception as e:
                print(f"Error loading {fp}: {e}")
                
    if not raw_samples:
        print(f"No raw files found in '{args.data_dir}'. Generating synthetic dataset.")
        raw_samples = generate_simulated_dataset()
        
    print(f"Total samples loaded: {len(raw_samples)}")
    
    # 1. Group samples by clip_name for clean splitting
    by_clip = {}
    for sample in raw_samples:
        clip_name = sample["clip_metadata"]["clip_name"]
        by_clip.setdefault(clip_name, []).append(sample)
        
    clip_names = list(by_clip.keys())
    random.shuffle(clip_names)
    
    # Split 80% train / 20% validation by clip
    split_idx = int(len(clip_names) * 0.8)
    train_clips = set(clip_names[:split_idx])
    val_clips = set(clip_names[split_idx:])
    
    train_raw = []
    val_raw = []
    for c_name, samples in by_clip.items():
        if c_name in train_clips:
            train_raw.extend(samples)
        else:
            val_raw.extend(samples)
            
    print(f"Split results: {len(train_clips)} train clips ({len(train_raw)} samples), "
          f"{len(val_clips)} val clips ({len(val_raw)} samples)")
          
    # 2. Extract features and targets
    train_X = [extract_sample_features(s) for s in train_raw]
    train_y = [calculate_reward(s) for s in train_raw]
    
    val_X = [extract_sample_features(s) for s in val_raw]
    val_y = [calculate_reward(s) for s in val_raw]
    
    # 3. Compute normalization parameters (means & stds) from training set only
    num_features = len(train_X[0])
    means = [0.0] * num_features
    stds = [1.0] * num_features
    
    num_samples = len(train_X)
    for j in range(num_features):
        col_sum = sum(train_X[i][j] for i in range(num_samples))
        means[j] = col_sum / num_samples
        
        variance = sum((train_X[i][j] - means[j]) ** 2 for i in range(num_samples)) / num_samples
        stds[j] = math.sqrt(variance) if variance > 1e-8 else 1.0
        
    # Normalize datasets
    def normalize_X(X):
        X_norm = []
        for row in X:
            norm_row = [(val - m) / s for val, m, s in zip(row, means, stds)]
            X_norm.append(norm_row)
        return X_norm
        
    train_X_norm = normalize_X(train_X)
    val_X_norm = normalize_X(val_X)
    
    # 4. Save processed dataset
    output_dir = os.path.dirname(args.output_json)
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = {
        "train": {
            "X": train_X_norm,
            "y": train_y
        },
        "val": {
            "X": val_X_norm,
            "y": val_y
        },
        "input_mean": means,
        "input_std": stds
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Processed dataset successfully saved to '{args.output_json}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoSolve Prepare Settings Dataset")
    parser.add_argument("--data-dir", default="ml/data/raw", help="Directory containing raw JSON samples")
    parser.add_argument("--output-json", default="ml/data/processed/settings_dataset.json", help="Path to save processed dataset")
    
    parsed_args = parser.parse_args()
    prepare_dataset(parsed_args)
