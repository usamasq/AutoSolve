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

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


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


def compute_dynamic_pixel_ratio(semantic_fp: str) -> float | None:
    """Returns mean dynamic object area ratio across frames, or None on failure."""
    try:
        with open(semantic_fp, 'r', encoding='utf-8') as f:
            sem_d = json.load(f)
        frames = sem_d.get("frames", [])
        if not frames:
            return None
        total = 0.0
        for frame in frames:
            area = 0.0
            for det in frame.get("detections", []):
                if det.get("is_dynamic", False):
                    box = det.get("box", [0, 0, 0, 0])
                    area += max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            total += min(1.0, area)
        return round(total / len(frames), 4)
    except Exception as e:
        print(f"Warning: dynamic ratio calculation failed ({e})")
        return None


SKIP_SUFFIXES = {
    '_video_meta.json', 'settings_dataset.json', 'recommended_defaults.json',
    'track_predictor.json', '_semantic_meta.json', 'defaults.json', '_base_trajectories.json'
}


def is_solve_json(filename: str) -> bool:
    """Check if the JSON file is a simulated solve attempt log."""
    return filename.endswith('.json') and not any(filename.endswith(s) for s in SKIP_SUFFIXES)


def extract_sample_features(sample: Dict[str, Any], video_meta: Dict[str, float] = None) -> List[float]:
    """
    Extracts a 29-dimensional feature vector from a SolveSample dictionary.
    Vector structure:
    [0:4] clip: width, height, fps, frame_count
    [4] tripod_mode
    [5] robust_mode
    [6:10] settings: pattern_size, search_size, correlation, threshold
    [10:20] footage_type one-hot
    [20:24] motion_model one-hot
    [24:29] video features: mean_motion, zoom_divergence, distortion_factor, grain_noise, dynamic_area_ratio
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
    
    # 5. Video features (5)
    if video_meta is None:
        video_meta = {}
    v_feats = [
        float(video_meta.get("mean_motion", 0.5)),
        float(video_meta.get("zoom_divergence", 0.0)),
        float(video_meta.get("distortion_factor", 0.0)),
        float(video_meta.get("grain_noise", video_meta.get("noise_ratio", 0.003))),
        float(video_meta.get("dynamic_area_ratio", 0.0))
    ]
    
    return clip_feats + switches + setting_feats + f_type_oh + m_model_oh + v_feats


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


def generate_simulated_dataset(video_metas: Dict[str, Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Generates a synthetic list of SolveSamples to simulate the data collection pipeline using video features."""
    print("Generating simulated SolveSamples for settings optimizer pipeline using video features...")
    samples = []
    
    clips = []
    if video_metas:
        # Use real video metadata from extracted features
        for c_name, meta in video_metas.items():
            f_type = "AUTO"
            if meta.get("mean_motion", 0.0) > 3.0:
                f_type = "ACTION"
            elif meta.get("zoom_divergence", 0.0) > 0.4:
                f_type = "DRONE" if random.random() < 0.5 else "CINEMATIC"
            elif meta.get("distortion_factor", 0.0) > 50.0:
                f_type = "CINEMATIC"
                
            clips.append({
                "clip_name": c_name,
                "width": meta.get("width", 1920),
                "height": meta.get("height", 1080),
                "fps": meta.get("fps", 24.0),
                "frame_count": meta.get("frame_count", 250),
                "footage_type": f_type,
                "mean_motion": meta.get("mean_motion", 0.5),
                "zoom_divergence": meta.get("zoom_divergence", 0.0),
                "distortion_factor": meta.get("distortion_factor", 0.0),
                "noise_ratio": meta.get("noise_ratio", 0.003)
            })
    else:
        # Generate 15 fake clips
        for i in range(15):
            f_type = random.choice(FOOTAGE_TYPES)
            w, h = random.choice([(1920, 1080), (3840, 2160), (1280, 720)])
            clips.append({
                "clip_name": f"synthetic_clip_{i:03d}",
                "width": w,
                "height": h,
                "fps": random.choice([23.976, 24.0, 25.0, 29.97, 30.0, 60.0]),
                "frame_count": random.randint(60, 450),
                "footage_type": f_type,
                "mean_motion": random.uniform(0.1, 4.0),
                "zoom_divergence": random.uniform(0.0, 1.2),
                "distortion_factor": random.uniform(0.0, 100.0),
                "noise_ratio": random.uniform(0.001, 0.08)
            })
        
    # For each clip, try multiple settings variations
    for clip in clips:
        # Expected optimal settings determined by physical video features to guide training
        optimal_pattern = 17
        if clip["width"] >= 3840:
            optimal_pattern = 55
        elif clip.get("noise_ratio", 0.0) > 0.03:
            optimal_pattern = 31
            
        optimal_search = 71
        if clip.get("mean_motion", 0.5) > 2.0:
            optimal_search += 100
        if clip.get("zoom_divergence", 0.0) > 0.3:
            optimal_search += 50
            
        optimal_corr = 0.70
        if clip.get("noise_ratio", 0.0) > 0.03:
            optimal_corr = 0.55
            
        optimal_thresh = 0.30
        if clip.get("mean_motion", 0.5) > 2.0:
            optimal_thresh = 0.15
            
        for _ in range(25):  # 25 solve attempts per clip
            pattern = random.choice([11, 15, 17, 21, 31, 55])
            search = random.choice([51, 71, 91, 121, 231])
            corr = random.uniform(0.45, 0.85)
            thresh = random.uniform(0.1, 0.5)
            m_model = random.choice(MOTION_MODELS)
            tripod = random.random() < 0.1
            robust = random.random() < 0.3
            
            # Calculate distance from optimal settings
            dist = (
                abs(pattern - optimal_pattern) / 50.0 + 
                abs(search - optimal_search) / 200.0 +
                abs(corr - optimal_corr) +
                abs(thresh - optimal_thresh)
            )
            
            # Higher distance = lower probability of success
            success_prob = max(0.1, 0.90 - dist)
            success = random.random() < success_prob
            
            error = 99.0
            bundle_ratio = 0.0
            if success:
                error = max(0.1, random.normalvariate(0.5 + dist * 1.5, 0.2))
                bundle_ratio = max(0.2, min(0.95, random.uniform(0.5, 0.9) - dist * 0.25))
                
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
    video_meta_lookup = {}
    real_samples = []
    
    if os.path.exists(args.data_dir):
        # 1. Load video feature files first
        video_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('_video_meta.json')]
        for fp in video_files:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    c_name = meta.get("clip_name")
                    if c_name:
                        # Compute dynamic_pixel_ratio if semantic features exist
                        semantic_fp = os.path.join(args.data_dir, f"{c_name}_semantic_meta.json")
                        meta["dynamic_area_ratio"] = 0.0
                        if os.path.exists(semantic_fp):
                            ratio = compute_dynamic_pixel_ratio(semantic_fp)
                            if ratio is not None:
                                meta["dynamic_area_ratio"] = ratio
                        video_meta_lookup[c_name] = meta
            except Exception as e:
                print(f"Error loading video meta {fp}: {e}")

        # 2. Load actual solve sample JSONs if present
        base_files = [f for f in os.listdir(args.data_dir) if f.endswith('_base_trajectories.json')]
        
        if base_files:
            print(f"Found {len(base_files)} base trajectory files. Simulating variations in-memory...")
            try:
                import sys
                project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if project_path not in sys.path:
                    sys.path.append(project_path)
                from ml.extract_cotracker_trajectories import simulate_variation
            except ImportError as ie:
                print(f"Warning: Failed to import simulate_variation: {ie}")
                simulate_variation = None
                
            if simulate_variation:
                for f in base_files:
                    fp = os.path.join(args.data_dir, f)
                    try:
                        with open(fp, 'r', encoding='utf-8') as fh:
                            base_data = json.load(fh)
                        
                        meta = base_data["clip_metadata"]
                        base_trajectories = [t["positions"] for t in base_data["tracks"]]
                        
                        # Run the 5 preset simulations in-memory
                        variations = [
                            ("BALANCED", False, False),
                            ("FAST", False, False),
                            ("QUALITY", False, False),
                            ("BALANCED", True, False),
                            ("BALANCED", False, True),
                        ]
                        for quality, robust, tripod in variations:
                            solve_sample = simulate_variation(base_trajectories, meta, quality, robust, tripod)
                            real_samples.append(solve_sample)
                    except Exception as e:
                        print(f"Error processing base trajectories from {fp}: {e}")
            else:
                print("Could not import simulate_variation. Skipping base trajectories simulation.")
        else:
            # Fallback for old multi-file JSONs
            for f in os.listdir(args.data_dir):
                if is_solve_json(f):
                    fp = os.path.join(args.data_dir, f)
                    try:
                        with open(fp, 'r', encoding='utf-8') as fh:
                            sample = json.load(fh)
                            if isinstance(sample, dict) and "clip_metadata" in sample and "settings" in sample:
                                real_samples.append(sample)
                    except Exception as e:
                        print(f"Error loading solve sample {fp}: {e}")

    if real_samples:
        print(f"Loaded {len(real_samples)} actual solve samples from '{args.data_dir}'.")
        raw_samples = real_samples
    else:
        print(f"No actual solve samples found in '{args.data_dir}'. Falling back to simulated dataset.")
        raw_samples = generate_simulated_dataset(video_meta_lookup)
        
    print(f"Total samples loaded: {len(raw_samples)}")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # 1. Group samples by clip_name for clean splitting
    by_clip = {}
    for sample in raw_samples:
        clip_name = sample["clip_metadata"]["clip_name"]
        by_clip.setdefault(clip_name, []).append(sample)
        
    clip_names = list(by_clip.keys())
    
    train_raw = []
    val_raw = []
    
    if len(clip_names) > 1:
        random.shuffle(clip_names)
        # Split 80% train / 20% validation by clip
        split_idx = max(1, int(len(clip_names) * 0.8))
        train_clips = set(clip_names[:split_idx])
        val_clips = set(clip_names[split_idx:])
        
        for c_name, samples in by_clip.items():
            if c_name in train_clips:
                train_raw.extend(samples)
            else:
                val_raw.extend(samples)
        print(f"Split results: {len(train_clips)} train clips ({len(train_raw)} samples), "
              f"{len(val_clips)} val clips ({len(val_raw)} samples)")
    else:
        # Fallback for single clip: split samples within the clip randomly
        all_samples = by_clip[clip_names[0]]
        random.shuffle(all_samples)
        split_idx = max(1, int(len(all_samples) * 0.8))
        train_raw = all_samples[:split_idx]
        val_raw = all_samples[split_idx:]
        print(f"Single clip split results: 1 clip, {len(train_raw)} train samples, {len(val_raw)} val samples")
          
    # 2. Extract features and targets
    train_X = [extract_sample_features(s, video_meta_lookup.get(s["clip_metadata"]["clip_name"])) for s in train_raw]
    train_y = [calculate_reward(s) for s in train_raw]
    
    val_X = [extract_sample_features(s, video_meta_lookup.get(s["clip_metadata"]["clip_name"])) for s in val_raw]
    val_y = [calculate_reward(s) for s in val_raw]
    
    # 3. Compute normalization parameters (means & stds) from training set only
    if not train_X:
        raise ValueError("Training set is empty. Cannot prepare dataset.")
        
    num_features = len(train_X[0])
    if NUMPY_AVAILABLE:
        X_arr = np.array(train_X, dtype=np.float32)
        means = X_arr.mean(axis=0).tolist()
        stds = X_arr.std(axis=0)
        stds[stds < 1e-8] = 1.0
        stds = stds.tolist()
        
        # Reset means/stds to 0.0/1.0 for binary/one-hot columns
        for j in range(num_features):
            if j in (4, 5) or (10 <= j < 24):
                means[j] = 0.0
                stds[j] = 1.0
    else:
        # Fallback manual loop
        means = [0.0] * num_features
        stds = [1.0] * num_features
        num_samples = len(train_X)
        for j in range(num_features):
            if j in (4, 5) or (10 <= j < 24):
                continue
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
    output_dir = os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else "."
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    parsed_args = parser.parse_args()
    prepare_dataset(parsed_args)
