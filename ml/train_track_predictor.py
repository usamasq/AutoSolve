# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve ML Track Quality Predictor Training Script.
Extracts training samples from collected JSON database, trains a PyTorch MLP,
and outputs model evaluation stats.
"""

import os
import sys
import json
import argparse
import random
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Map footage type strings to indices (matching properties.py order)
FOOTAGE_TYPE_MAP = {
    'AUTO': 0, 'INDOOR': 1, 'OUTDOOR': 2, 'DRONE': 3, 'HANDHELD': 4,
    'GIMBAL': 5, 'ACTION': 6, 'VFX': 7, 'SCREEN': 8, 'CINEMATIC': 9
}


if TORCH_AVAILABLE:
    class TrackMLP(nn.Module):
        """3-layer Multi-Layer Perceptron for track survival prediction."""
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(20, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )
            
        def forward(self, x):
            return self.network(x)
else:
    class TrackMLP:
        """Fallback class when PyTorch is not available."""
        pass


def load_dataset(data_dir: str):
    """Load JSON files and extract feature/label samples."""
    import math
    from ml.features.track_features import extract_features_from_history
    print(f"Loading data from '{data_dir}'...")
    
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"Data directory '{data_dir}' not found or empty. Generating synthetic track predictor samples...")
        samples_by_clip = {}
        for i in range(10): # 10 dummy clips
            clip_name = f"dummy_clip_{i}"
            samples_by_clip[clip_name] = []
            for _ in range(50): # 50 tracks per clip
                features = np.random.randn(20).astype(np.float32).tolist()
                label = 1.0 if random.random() > 0.4 else 0.0
                samples_by_clip[clip_name].append((features, label))
        clips = list(samples_by_clip.keys())
        return samples_by_clip, clips
        
    samples_by_clip = {}
    solve_samples_to_process = []
    
    # 1. Look for base trajectories first
    base_files = [f for f in os.listdir(data_dir) if f.endswith('_base_trajectories.json')]
    
    if base_files:
        print(f"Found {len(base_files)} base trajectory files. Simulating balanced_standard in-memory...")
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
                fp = os.path.join(data_dir, f)
                try:
                    with open(fp, 'r', encoding='utf-8') as fh:
                        base_data = json.load(fh)
                    meta = base_data["clip_metadata"]
                    base_trajectories = [t["positions"] for t in base_data["tracks"]]
                    # Simulate balanced_standard (quality="BALANCED", robust=False, tripod=False)
                    solve_sample = simulate_variation(base_trajectories, meta, "BALANCED", False, False)
                    solve_samples_to_process.append((f, solve_sample))
                except Exception as e:
                    print(f"Error processing base trajectories from {fp}: {e}")
        else:
            print("Could not import simulate_variation. Skipping base trajectories.")
            
    # 2. If no base files processed, look for old balanced_standard.json files on disk
    if not solve_samples_to_process:
        json_files = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('_balanced_standard.json')
        ]
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    solve_data = json.load(f)
                solve_samples_to_process.append((os.path.basename(file_path), solve_data))
            except Exception as e:
                print(f"Error reading {os.path.basename(file_path)}: {e}")
                
    if not solve_samples_to_process:
        print("No training files found. Generating synthetic track predictor samples...")
        samples_by_clip = {}
        for i in range(10):
            clip_name = f"dummy_clip_{i}"
            samples_by_clip[clip_name] = []
            for _ in range(50):
                features = np.random.randn(20).astype(np.float32).tolist()
                label = 1.0 if random.random() > 0.4 else 0.0
                samples_by_clip[clip_name].append((features, label))
        clips = list(samples_by_clip.keys())
        return samples_by_clip, clips
    
    for fname, solve_data in solve_samples_to_process:
        try:
            clip_name = solve_data["clip_metadata"]["clip_name"]
            if clip_name not in samples_by_clip:
                samples_by_clip[clip_name] = []
                
            settings = solve_data["settings"]
            footage_idx = FOOTAGE_TYPE_MAP.get(settings["footage_type"], 0)
            robust_mode = settings["robust_mode"]
            
            # Find the base clip name to load corresponding semantic features
            base_clip_name = clip_name
            for suffix in ["_balanced_standard", "_fast_standard", "_quality_standard", "_balanced_robust", "_balanced_tripod"]:
                if base_clip_name.endswith(suffix):
                    base_clip_name = base_clip_name[:-len(suffix)]
                    break
                    
            # Load semantic meta JSON if exists
            semantic_fp = os.path.join(data_dir, f"{base_clip_name}_semantic_meta.json")
            frame_detections = {}
            if os.path.exists(semantic_fp):
                try:
                    with open(semantic_fp, 'r') as sfh:
                        sem_d = json.load(sfh)
                        for f_meta in sem_d.get("frames", []):
                            frame_detections[f_meta["frame_idx"]] = f_meta.get("detections", [])
                except Exception as se:
                    print(f"Warning: failed to load semantic meta {semantic_fp}: {se}")
            
            tracks = solve_data["tracks"]
            if not tracks:
                continue
                
            # Precompute active track positions/velocities per frame to optimize lookup from O(N^2) to O(N)
            max_len = max(len(t["positions"]) for t in tracks)
            frame_neighbors = [[] for _ in range(max_len)]
            
            for t in tracks:
                coords = t["positions"]
                t_name = t["track_name"]
                for f in range(len(coords)):
                    curr = coords[f]
                    prev = coords[f-1] if f > 0 else curr
                    frame_neighbors[f].append((t_name, [prev, curr]))
            
            for t in tracks:
                name = t["track_name"]
                coords = t["positions"]
                survived = t["survived"] and t["has_bundle"]
                
                # Determine starting region index (0-8)
                region_name = t["region"]
                # Convert region string back to index
                regions_list = [
                    'top-left', 'top-center', 'top-right',
                    'mid-left', 'center', 'mid-right',
                    'bottom-left', 'bottom-center', 'bottom-right'
                ]
                region_idx = regions_list.index(region_name) if region_name in regions_list else 4
                
                # Extract features at multiple check frames during the track's life
                # Needs at least 6 frames to extract trajectory deltas
                if len(coords) < 6:
                    continue
                    
                for frame_idx in range(5, len(coords), 10):
                    # History coords up to current check frame
                    sub_coords = coords[:frame_idx+1]
                    curr_pos = coords[frame_idx]
                    
                    # Gather neighbor coords active at this same frame
                    neighbors = [item for n_name, item in frame_neighbors[frame_idx] if n_name != name]
                    
                    # Calculate flow consensus average
                    vels_x = [item[-1][0] - item[-2][0] for item in neighbors]
                    vels_y = [item[-1][1] - item[-2][1] for item in neighbors]
                    avg_vel_x = np.mean(vels_x) if vels_x else 0.0
                    avg_vel_y = np.mean(vels_y) if vels_y else 0.0
                    my_vel_x = coords[frame_idx][0] - coords[frame_idx-1][0]
                    my_vel_y = coords[frame_idx][1] - coords[frame_idx-1][1]
                    flow_consensus_deviation = math.sqrt((my_vel_x - avg_vel_x)**2 + (my_vel_y - avg_vel_y)**2)
                    
                    # Calculate proximity to dynamic semantic mask
                    dist_to_dynamic_mask = 1.0
                    on_dynamic_mask = 0.0
                    
                    detections = frame_detections.get(frame_idx, [])
                    if detections:
                        min_dist = 999.0
                        for det in detections:
                            if det.get("is_dynamic", False):
                                box = det.get("box", [0.0, 0.0, 0.0, 0.0])
                                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                                if x1 <= curr_pos[0] <= x2 and y1 <= curr_pos[1] <= y2:
                                    on_dynamic_mask = 1.0
                                    min_dist = 0.0
                                    break
                                else:
                                    dx = max(x1 - curr_pos[0], 0.0, curr_pos[0] - x2)
                                    dy = max(y1 - curr_pos[1], 0.0, curr_pos[1] - y2)
                                    dist = math.sqrt(dx**2 + dy**2)
                                    if dist < min_dist:
                                        min_dist = dist
                        if min_dist < 998.0:
                            dist_to_dynamic_mask = min_dist
                    
                    features = extract_features_from_history(
                        coords=sub_coords,
                        neighbors_coords=neighbors,
                        region_idx=region_idx,
                        footage_idx=footage_idx,
                        robust_mode=robust_mode,
                        dist_to_dynamic_mask=dist_to_dynamic_mask,
                        on_dynamic_mask=on_dynamic_mask,
                        flow_consensus_deviation=flow_consensus_deviation
                    )
                    
                    # Solver feedback target: y = survived * exp(-0.5 * MRE)
                    if survived:
                        mre = t.get("average_error", 10.0)
                        label = float(math.exp(-0.5 * mre))
                    else:
                        label = 0.0
                    samples_by_clip[clip_name].append((features, label))
                    
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")
            
    # Flatten samples list by clip for split
    clips = list(samples_by_clip.keys())
    random.shuffle(clips)
    
    return samples_by_clip, clips


def train_model(args):
    """Train PyTorch model using SolveSamples."""
    if not NUMPY_AVAILABLE:
        print("Error: NumPy (numpy) is not installed. NumPy is required to train the model.")
        print("Please install numpy: pip install numpy")
        return
        
    if not TORCH_AVAILABLE:
        print("Error: PyTorch (torch) is not installed. PyTorch is required to train the model.")
        print("Please install torch: pip install torch")
        return
        
    samples_by_clip, clips = load_dataset(args.data_dir)
    if not clips:
        print("No samples collected. Exiting.")
        return
        
    # Split clips (by clip_id to prevent data leakage)
    split_idx = int(len(clips) * args.train_split)
    train_clips = clips[:split_idx]
    val_clips = clips[split_idx:]
    
    print(f"Split: {len(train_clips)} train clips, {len(val_clips)} validation clips")
    
    # Collect train/val vectors
    X_train_raw = []
    y_train = []
    for c in train_clips:
        for feat, lbl in samples_by_clip[c]:
            X_train_raw.append(feat)
            y_train.append(lbl)
            
    X_val_raw = []
    y_val = []
    for c in val_clips:
        for feat, lbl in samples_by_clip[c]:
            X_val_raw.append(feat)
            y_val.append(lbl)
            
    if not X_train_raw:
        print("No training samples found.")
        return
        
    X_train_raw = np.array(X_train_raw, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val_raw = np.array(X_val_raw, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    
    print(f"Training dataset size:   {len(X_train_raw)} samples")
    print(f"Validation dataset size: {len(X_val_raw)} samples")
    print(f"Positive ratio (train):  {np.mean(y_train):.1%}")
    
    # Compute mean and standard deviation for normalization
    input_mean = np.mean(X_train_raw, axis=0)
    input_std = np.std(X_train_raw, axis=0)
    # Prevent divide by zero
    input_std[input_std < 1e-6] = 1.0
    
    # Normalize features
    X_train = (X_train_raw - input_mean) / input_std
    X_val = (X_val_raw - input_mean) / input_std if len(X_val_raw) > 0 else np.zeros((0, 15), dtype=np.float32)
    
    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Setup model, loss, optimizer
    model = TrackMLP()
    
    # Load pretrained weights if specified
    if hasattr(args, 'pretrained') and args.pretrained:
        if os.path.exists(args.pretrained):
            print(f"Loading pretrained weights from '{args.pretrained}'...")
            try:
                if args.pretrained.endswith('.json'):
                    with open(args.pretrained, 'r') as fh:
                        meta_data = json.load(fh)
                    weights_dict = {}
                    for k, v in meta_data["weights"].items():
                        weights_dict[k] = torch.tensor(v, dtype=torch.float32)
                    model.load_state_dict(weights_dict)
                    print("Successfully loaded weights from metadata JSON file.")
                else:
                    model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
                    print("Successfully loaded weights from PyTorch checkpoint (.pt) file.")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                print("Proceeding with fresh training.")
        else:
            print(f"Pretrained weight file not found at '{args.pretrained}'. Proceeding with fresh training.")

    pos_ratio = np.mean(y_train > 0.5) if len(y_train) > 0 else 0.5
    pos_weight = torch.tensor([(1.0 - pos_ratio) / max(1e-5, pos_ratio)], dtype=torch.float32)
    print(f"Class balance (quality > 0.5): {pos_ratio:.1%}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training Loop
    print("\nTraining MLP model...")
    best_loss = 9999.0
    best_weights = None
    best_train_thresh = 0.5
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Data Augmentation: add low-magnitude Gaussian noise to continuous features
            if model.training:
                noise = torch.randn_like(batch_x) * 0.01
                # Zero out noise for region_idx, footage_idx, robust_mode, on_dynamic_mask (12, 13, 14, 16)
                for col_idx in range(batch_x.shape[1]):
                    if col_idx in (12, 13, 14, 16):
                        noise[:, col_idx] = 0.0
                batch_x_augmented = batch_x + noise
            else:
                batch_x_augmented = batch_x
                
            logits = model(batch_x_augmented)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
            
        epoch_loss /= len(X_train)
        scheduler.step()
        
        # Validation evaluation
        if len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.tensor(X_val))
                val_loss = criterion(val_logits, torch.tensor(y_val).unsqueeze(1)).item()
                
                # Check metrics (accuracy) using Sigmoid on logits
                probs = torch.sigmoid(val_logits)
                probs_np = probs.numpy().flatten()
                
                # Find the best decision threshold on training predictions (no leakage)
                train_logits = model(torch.tensor(X_train))
                train_probs = torch.sigmoid(train_logits).numpy().flatten()
                y_train_bin = (y_train > 0.5).astype(np.float32)
                
                best_thresh = 0.5
                best_tr_acc = 0.0
                for t_val in np.linspace(0.1, 0.9, 81):
                    tr_acc = np.mean((train_probs > t_val) == y_train_bin)
                    if tr_acc > best_tr_acc:
                        best_tr_acc = tr_acc
                        best_thresh = t_val
                
                y_val_bin = (y_val > 0.5).astype(np.float32)
                
                # Standard threshold 0.5 evaluation
                acc_05 = np.mean((probs_np > 0.5) == y_val_bin)
                
                # Optimized threshold evaluation
                preds_opt = (probs_np > best_thresh).astype(np.float32)
                acc_opt = np.mean(preds_opt == y_val_bin)
                
                # Compute F1 and Balanced Accuracy
                tp = np.sum((y_val_bin == 1.0) & (preds_opt == 1.0))
                tn = np.sum((y_val_bin == 0.0) & (preds_opt == 0.0))
                fp = np.sum((y_val_bin == 0.0) & (preds_opt == 1.0))
                fn = np.sum((y_val_bin == 1.0) & (preds_opt == 0.0))
                
                sens = tp / max(1, tp + fn)
                spec = tn / max(1, tn + fp)
                bal_acc = (sens + spec) / 2.0
                f1_score = 2.0 * tp / max(1e-8, 2.0 * tp + fp + fn)
                
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Acc (0.5): {acc_05:.1%} | Val Acc (Opt @ {best_thresh:.2f}): {acc_opt:.1%} | "
                      f"Val BalAcc: {bal_acc:.1%} | Val F1: {f1_score:.1%}")
                
            # Track best model weights
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                best_train_thresh = best_thresh
        else:
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {epoch_loss:.4f}")
                
    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"\nBest Validation Loss: {best_loss:.4f}")
        
        # Shift final layer bias to align optimal threshold with 0.5
        import math
        logit_shift = math.log(best_train_thresh / max(1e-5, 1.0 - best_train_thresh))
        if "network.8.bias" in best_weights:
            model.network[8].bias.data -= logit_shift
            print(f"Shifted final layer bias by {logit_shift:.4f} to align optimal threshold ({best_train_thresh:.2f}) with 0.5 for exporting.")
        
    # Save PyTorch checkpoints
    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.out_dir, "track_predictor_pytorch.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved PyTorch weights checkpoint to '{checkpoint_path}'")
    
    # Save the normalization stats along with weights for export
    meta_weights = {
        "weights": {k: v.numpy().tolist() for k, v in model.state_dict().items()},
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist()
    }
    
    meta_weights_path = os.path.join(args.out_dir, "model_meta_weights.json")
    with open(meta_weights_path, 'w') as f:
        json.dump(meta_weights, f, indent=4)
    print(f"Saved numpy-compatible meta JSON weights to '{meta_weights_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoSolve ML Track Predictor Trainer")
    parser.add_argument("--data-dir", default="ml/data/raw", help="Directory containing raw JSON samples")
    parser.add_argument("--out-dir", default="ml/runs/track_predictor", help="Directory to save run results")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--train-split", type=float, default=0.8, help="Ratio for training vs validation split")
    parser.add_argument("--pretrained", default=None, help="Path to pretrained model weight file (.pt or metadata JSON) to resume/fine-tune training")
    
    parsed_args = parser.parse_args()
    train_model(parsed_args)
