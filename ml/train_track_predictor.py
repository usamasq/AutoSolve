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
import numpy as np

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features.track_features import extract_features_from_history

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
                nn.Linear(15, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            return self.network(x)
else:
    class TrackMLP:
        """Fallback class when PyTorch is not available."""
        pass


def load_dataset(data_dir: str):
    """Load JSON files and extract feature/label samples."""
    print(f"Loading data from '{data_dir}'...")
    json_files = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.json')
    ]
    
    if not json_files:
        print("No JSON files found.")
        return [], [], []
        
    samples_by_clip = {}
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                solve_data = json.load(f)
                
            clip_name = solve_data["clip_metadata"]["clip_name"]
            if clip_name not in samples_by_clip:
                samples_by_clip[clip_name] = []
                
            settings = solve_data["settings"]
            footage_idx = FOOTAGE_TYPE_MAP.get(settings["footage_type"], 0)
            robust_mode = settings["robust_mode"]
            
            tracks = solve_data["tracks"]
            # Track coordinates dictionary by name for easy neighbor lookup
            track_coords = {t["track_name"]: t["positions"] for t in tracks}
            
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
                # For training, sample every 10 frames of the track coordinates to simulate online checks
                # Needs at least 6 frames to extract trajectory deltas
                if len(coords) < 6:
                    continue
                    
                for frame_idx in range(5, len(coords), 10):
                    # History coords up to current check frame
                    sub_coords = coords[:frame_idx+1]
                    
                    # Gather neighbor coords at this same frame
                    neighbors = []
                    for n_name, n_coords in track_coords.items():
                        if n_name != name and len(n_coords) > frame_idx:
                            # Neighbor is active at this frame index
                            neighbors.append(n_coords[:frame_idx+1])
                            
                    features = extract_features_from_history(
                        coords=sub_coords,
                        neighbors_coords=neighbors,
                        region_idx=region_idx,
                        footage_idx=footage_idx,
                        robust_mode=robust_mode
                    )
                    
                    # Label: survived to solve (1.0) or died/filtered (0.0)
                    label = 1.0 if survived else 0.0
                    samples_by_clip[clip_name].append((features, label))
                    
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")
            
    # Flatten samples list by clip for split
    clips = list(samples_by_clip.keys())
    random.shuffle(clips)
    
    return samples_by_clip, clips


def train_model(args):
    """Train PyTorch model using SolveSamples."""
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
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training Loop
    print("\nTraining MLP model...")
    best_loss = 9999.0
    best_weights = None
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
            
        epoch_loss /= len(X_train)
        
        # Validation evaluation
        if len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(torch.tensor(X_val))
                val_loss = criterion(val_preds, torch.tensor(y_val).unsqueeze(1)).item()
                
                # Check metrics (accuracy)
                bin_preds = (val_preds.numpy() > 0.5).astype(np.float32)
                acc = np.mean(bin_preds == y_val.reshape(-1, 1))
                
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.1%}")
                
            # Track best model weights
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {epoch_loss:.4f}")
                
    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"\nBest Validation Loss: {best_loss:.4f}")
        
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
    
    parsed_args = parser.parse_args()
    train_model(parsed_args)
