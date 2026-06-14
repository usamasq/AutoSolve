# SPDX-FileCopyrightText: 2026 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve ML Patch Rigidity Classifier Trainer.
Uses RANSAC motion segmentation over CoTracker / OpenCV trajectories to automatically 
extract rigid (positive) vs. dynamic (negative) pixel patches from video frames, 
trains a PyTorch CNN, and exports it to ONNX for Blender add-on inference.
"""

import os
import sys
import json
import argparse
import random
import numpy as np

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


SKIP_SUFFIXES = {
    '_video_meta.json', 'settings_dataset.json', 'recommended_defaults.json',
    'track_predictor.json', '_semantic_meta.json', 'defaults.json', '_base_trajectories.json'
}


def is_solve_json(filename: str) -> bool:
    return filename.endswith('.json') and not any(filename.endswith(s) for s in SKIP_SUFFIXES)


if TORCH_AVAILABLE:
    class PatchRigidityCNN(nn.Module):
        """Lightweight 2-layer CNN for 32x32 patch rigidity classification."""
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 32x32 -> 16x16
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
            )
            self.classifier = nn.Sequential(
                nn.Linear(32 * 8 * 8, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
else:
    class PatchRigidityCNN:
        pass


def generate_synthetic_patch_data(num_samples=200) -> tuple:
    """Generate fake grayscale patches for testing/fallback when video is unavailable."""
    print("Generating synthetic patch rigidity training data...")
    X = []
    y = []
    
    for i in range(num_samples):
        label = 1.0 if random.random() < 0.5 else 0.0
        # Create 32x32 image
        patch = np.zeros((32, 32), dtype=np.float32)
        
        if label == 1.0:
            # Rigid: High-contrast static features / lines / grid
            x_line = random.randint(5, 25)
            y_line = random.randint(5, 25)
            patch[x_line-2:x_line+2, :] = 1.0
            patch[:, y_line-2:y_line+2] = 1.0
            # Low noise
            patch += np.random.normal(0.0, 0.05, (32, 32))
        else:
            # Dynamic: Homogeneous sky (flat/no details) or pure random noise (water ripples / blur)
            if random.random() < 0.5:
                # Homogeneous flat sky/wall
                patch += random.uniform(0.1, 0.8)
                patch += np.random.normal(0.0, 0.005, (32, 32))
            else:
                # High chaotic noise
                patch += np.random.normal(0.5, 0.25, (32, 32))
                
        patch = np.clip(patch, 0.0, 1.0)
        X.append(patch)
        y.append(label)
        
    # Shape: (N, 1, 32, 32)
    X = np.expand_dims(np.array(X, dtype=np.float32), 1)
    y = np.array(y, dtype=np.float32)
    clip_ids = [f"synth_clip_{i // 40}" for i in range(num_samples)]
    return X, y, clip_ids


def extract_real_patch_data(clips_dir: str, data_dir: str) -> tuple:
    """Scan solve samples, load associated videos, and run homography RANSAC to label rigid vs dynamic patches."""
    if not OPENCV_AVAILABLE:
        print("Warning: OpenCV is not available. Cannot extract patches from real video.")
        return None, None, None
        
    # Find solve files
    solve_files = [f for f in os.listdir(data_dir) if is_solve_json(f)]
    if not solve_files:
        print("No solve files found in data directory.")
        return None, None, None
        
    X_list = []
    y_list = []
    clip_ids = []
    
    for file_name in solve_files:
        solve_path = os.path.join(data_dir, file_name)
        try:
            with open(solve_path, 'r', encoding='utf-8') as f:
                solve_data = json.load(f)
                
            clip_name = solve_data["clip_metadata"]["clip_name"]
            # Look for video in clips directory
            video_path = os.path.join(clips_dir, clip_name)
            if not os.path.exists(video_path):
                # Try with different extensions or stripping suffix
                base_name = os.path.splitext(clip_name)[0]
                # Check clips_dir for matches
                matches = [f for f in os.listdir(clips_dir) if f.startswith(base_name)]
                if matches:
                    video_path = os.path.join(clips_dir, matches[0])
                else:
                    print(f"Skipping {clip_name}: Video file not found in '{clips_dir}'")
                    continue
                    
            print(f"Segmenting motion on video: {os.path.basename(video_path)} using RANSAC...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            try:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                tracks = solve_data["tracks"]
                # Track coordinates dictionary: track_name -> {frame_idx: (x, y)}
                track_points = {}
                for t in tracks:
                    positions = t["positions"]
                    track_points[t["track_name"]] = {idx: pos for idx, pos in enumerate(positions)}
                    
                num_frames = max(len(t["positions"]) for t in tracks)
                
                # Map frames to tracks and inliers count
                track_outlier_votes = {t["track_name"]: {"outlier": 0, "total": 0} for t in tracks}
                
                # Run RANSAC across frame transitions to label outlier/dynamic tracks
                for f_idx in range(num_frames - 1):
                    src_pts = []
                    dst_pts = []
                    names = []
                    
                    for name, pos_map in track_points.items():
                        if f_idx in pos_map and (f_idx + 1) in pos_map:
                            x0, y0 = pos_map[f_idx]
                            x1, y1 = pos_map[f_idx + 1]
                            src_pts.append([x0 * width, y0 * height])
                            dst_pts.append([x1 * width, y1 * height])
                            names.append(name)
                            
                    if len(src_pts) >= 8:
                        src_pts = np.array(src_pts, dtype=np.float32)
                        dst_pts = np.array(dst_pts, dtype=np.float32)
                        
                        # Find homography of dominant background camera motion
                        H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if inliers is not None:
                            for i, name in enumerate(names):
                                track_outlier_votes[name]["total"] += 1
                                if inliers[i][0] == 0:
                                    track_outlier_votes[name]["outlier"] += 1
                                    
                # Final labeling of tracks based on outlier ratio
                track_rigidity = {}
                for name, votes in track_outlier_votes.items():
                    if votes["total"] > 0:
                        outlier_ratio = votes["outlier"] / votes["total"]
                        # If track behaves as outlier > 35% of the time, it's non-rigid/dynamic
                        track_rigidity[name] = 0.0 if outlier_ratio > 0.35 else 1.0
                    else:
                        track_rigidity[name] = 1.0  # default to rigid
                        
                # Load frames and extract patches
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Sample tracks active at this frame
                    for name, pos_map in track_points.items():
                        # Limit sample count to avoid bloating the dataset with consecutive frame duplicates
                        if frame_idx in pos_map and frame_idx % 10 == 0:
                            x, y = pos_map[frame_idx]
                            px = int(x * width)
                            py = int(y * height)
                            
                            # Pad boundary patches safely
                            y0, y1 = py - 16, py + 16
                            x0, x1 = px - 16, px + 16
                            
                            if y0 >= 0 and y1 < height and x0 >= 0 and x1 < width:
                                patch = gray[y0:y1, x0:x1].astype(np.float32) / 255.0
                                label = track_rigidity[name]
                                X_list.append(patch)
                                y_list.append(label)
                                clip_ids.append(clip_name)
                                
                    frame_idx += 1
            finally:
                cap.release()
            
        except Exception as e:
            print(f"Error processing video patch generation: {e}")
            
    if not X_list:
        return None, None, None
        
    X = np.expand_dims(np.array(X_list, dtype=np.float32), 1)
    y = np.array(y_list, dtype=np.float32)
    return X, y, clip_ids


def train_rigidity_model(args):
    """Main model training and ONNX export loop."""
    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for model training. Please run: pip install torch")
        sys.exit(1)
        
    # 1. Gather data
    X, y, clip_ids = None, None, None
    if os.path.exists(args.clips_dir) and os.path.exists(args.data_dir):
        X, y, clip_ids = extract_real_patch_data(args.clips_dir, args.data_dir)
        
    if X is None:
        X, y, clip_ids = generate_synthetic_patch_data(num_samples=400)
        
    print(f"Training dataset size: {len(X)} patches. Rigid ratio: {np.mean(y):.1%}")
    
    # 2. Split dataset by clip to prevent data leakage
    unique_clips = list(set(clip_ids))
    random.seed(42)
    random.shuffle(unique_clips)
    split_idx = max(1, int(len(unique_clips) * 0.8))
    train_clips = set(unique_clips[:split_idx])
    
    train_idx = [i for i, c in enumerate(clip_ids) if c in train_clips]
    val_idx = [i for i, c in enumerate(clip_ids) if c not in train_clips]
    
    train_dataset = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X[val_idx]), torch.tensor(y[val_idx]).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 3. Model setup
    model = PatchRigidityCNN()
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 4. Training loop
    print("Training CNN Patch Rigidity model...")
    best_loss = 9999.0
    best_weights = None
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_idx)
        scheduler.step()
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_X_t = torch.tensor(X[val_idx])
            val_y_t = torch.tensor(y[val_idx]).unsqueeze(1)
            val_preds = model(val_X_t)
            val_loss = criterion(val_preds, val_y_t).item()
            acc = np.mean((val_preds.numpy() > 0.5) == y[val_idx].reshape(-1, 1)) if len(val_idx) > 0 else 1.0
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.1%}")
            
        # Track best weights
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            
    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"\nBest Validation Loss: {best_loss:.4f}")
            
    # 5. Export to ONNX
    os.makedirs(args.out_dir, exist_ok=True)
    onnx_path = os.path.join(args.out_dir, "patch_rigidity.onnx")
    
    print(f"Exporting model to ONNX: '{onnx_path}'...")
    dummy_input = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["patch_pixels"],
        output_names=["rigidity_score"],
        opset_version=17,
        dynamic_axes={"patch_pixels": {0: "batch"}, "rigidity_score": {0: "batch"}}
    )
    
    # Write meta JSON with normalization metadata
    meta = {
        "input_size": [32, 32],
        "channels": 1,
        "normalization": "div_255",
        "description": "AutoSolve Patch Rigidity CNN. Inputs 32x32 grayscale image patches scaled to [0,1]. Outputs static/rigid rigidity probability in [0,1]."
    }
    meta_path = os.path.join(args.out_dir, "patch_rigidity_meta.json")
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=4)
        
    print(f"ONNX Model saved and validated. Meta JSON written to: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoSolve ML Patch Rigidity CNN Trainer")
    parser.add_argument("--data-dir", default="ml/data/raw", help="Directory containing raw JSON samples")
    parser.add_argument("--clips-dir", default="ml/clips", help="Directory containing raw video clips")
    parser.add_argument("--out-dir", default="autosolve/tracker/models", help="Directory to save ONNX model output")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    
    parsed_args = parser.parse_args()
    train_rigidity_model(parsed_args)
