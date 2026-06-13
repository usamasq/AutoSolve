# SPDX-FileCopyrightText: 2026 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve ML Settings Optimizer Training Script.
Trains a PyTorch MLP to predict the expected tracking reward from clip and settings features.
"""

import os
import sys
import json
import argparse
import random
import math

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


if TORCH_AVAILABLE:
    class SettingsMLP(nn.Module):
        """3-layer Multi-Layer Perceptron for expected reward prediction."""
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(28, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Reward is in [0, 1] range
            )
            
        def forward(self, x):
            return self.network(x)
else:
    class SettingsMLP:
        """Fallback class when PyTorch is not available."""
        pass
 
 
def generate_fallback_weights() -> dict:
    """Generate mock weights using python standard libraries if PyTorch is not available."""
    print("Generating heuristic weights for settings optimizer fallback...")
    
    # 28 input features, 64 hidden, 32 hidden, 1 output
    # Setup simple weights that reward matching resolution/pattern sizes
    layer1_w = []
    for i in range(64):
        row = []
        for j in range(28):
            val = 0.0
            # Example heuristic weight connections:
            # Connect width (0) and pattern_size (6)
            if i == 0 and j in (0, 6):
                val = 0.5
            # Connect search_size (7) and correlation (8)
            elif i == 1 and j in (7, 8):
                val = -0.5
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
    
    # Sigmoid(0.0) = 0.5 default expected reward
    layer3_b = [0.0]
    
    return {
        "network.0.weight": layer1_w,
        "network.0.bias": layer1_b,
        "network.2.weight": layer2_w,
        "network.2.bias": layer2_b,
        "network.4.weight": layer3_w,
        "network.4.bias": layer3_b
    }


def train_model(args):
    """Load preprocessed dataset, train PyTorch MLP, and save weights."""
    if not os.path.exists(args.data_path):
        print(f"Dataset path '{args.data_path}' not found. Please run prepare_dataset.py first.")
        return
        
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)
        
    os.makedirs(args.out_dir, exist_ok=True)
    
    if not TORCH_AVAILABLE:
        print("WARNING: PyTorch is not available. Exporting heuristic weights.")
        weights = generate_fallback_weights()
        meta = {
            "input_mean": dataset["input_mean"],
            "input_std": dataset["input_std"],
            "weights": weights,
            "training_notes": "fallback generated without PyTorch"
        }
        output_path = os.path.join(args.out_dir, "model_meta_weights.json")
        with open(output_path, 'w') as f:
            json.dump(meta, f, indent=4)
        print(f"Fallback model meta weights saved to '{output_path}'")
        return
        
    print("Training settings optimizer MLP model using PyTorch...")
    
    # Convert lists to PyTorch Tensors
    train_X = torch.tensor(dataset["train"]["X"], dtype=torch.float32)
    train_y = torch.tensor(dataset["train"]["y"], dtype=torch.float32).unsqueeze(1)
    val_X = torch.tensor(dataset["val"]["X"], dtype=torch.float32)
    val_y = torch.tensor(dataset["val"]["y"], dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    model = SettingsMLP()
    
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

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Simple training loop
    best_val_loss = float('inf')
    best_weights = None
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Data Augmentation: add low-magnitude Gaussian noise to continuous features
            if model.training:
                noise = torch.randn_like(batch_x) * 0.01
                # Zero out noise for binary switches and one-hot encodings (4, 5, 10-23)
                for col_idx in range(batch_x.shape[1]):
                    if col_idx in (4, 5) or (10 <= col_idx < 24):
                        noise[:, col_idx] = 0.0
                batch_x_augmented = batch_x + noise
            else:
                batch_x_augmented = batch_x
                
            outputs = model(batch_x_augmented)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs, val_y).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save deep copy of state dict
            best_weights = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{args.epochs:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
    print(f"Training complete. Best Validation MSE Loss: {best_val_loss:.6f}")
    
    # Save metadata along with weights
    meta = {
        "input_mean": dataset["input_mean"],
        "input_std": dataset["input_std"],
        "weights": best_weights,
        "best_val_loss": best_val_loss,
        "epochs_trained": args.epochs
    }
    
    output_path = os.path.join(args.out_dir, "model_meta_weights.json")
    with open(output_path, 'w') as f:
        json.dump(meta, f, indent=4)
        
    print(f"Model meta weights successfully saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoSolve Train Settings Optimizer Model")
    parser.add_argument("--data-path", default="ml/data/processed/settings_dataset.json", help="Path to processed JSON dataset")
    parser.add_argument("--out-dir", default="ml/runs/settings_optimizer", help="Directory to save trained model files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--pretrained", default=None, help="Path to pretrained model weight file (.pt or metadata JSON) to resume/fine-tune training")
    
    parsed_args = parser.parse_args()
    train_model(parsed_args)
