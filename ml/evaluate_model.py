# SPDX-FileCopyrightText: 2026 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve ML Settings Optimizer Evaluation.
Evaluates settings expected reward MLP predictions on the validation set.
"""

import os
import json
import argparse
import math
from typing import List, Dict, Any


def matmul(A: List[List[float]], B_T: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication A @ B_T where B_T is shape (M, K) (transpose of W)."""
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
    """In-place element-wise bias addition."""
    for i in range(len(A)):
        for j in range(len(b)):
            A[i][j] += b[j]
    return A


def relu(A: List[List[float]]) -> List[List[float]]:
    """In-place Rectified Linear Unit activation."""
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = max(0.0, A[i][j])
    return A


def sigmoid(A: List[List[float]]) -> List[List[float]]:
    """In-place Sigmoid activation."""
    for i in range(len(A)):
        for j in range(len(A[0])):
            val = max(-20.0, min(20.0, A[i][j]))
            A[i][j] = 1.0 / (1.0 + math.exp(-val))
    return A


def batchnorm1d(X: List[List[float]], weight: List[float], bias: List[float], running_mean: List[float], running_var: List[float], eps: float = 1e-5) -> List[List[float]]:
    N = len(X)
    C = len(X[0])
    out = [[0.0] * C for _ in range(N)]
    for i in range(N):
        for j in range(C):
            out[i][j] = (X[i][j] - running_mean[j]) / math.sqrt(running_var[j] + eps) * weight[j] + bias[j]
    return out


def run_inference(X: List[List[float]], weights: Dict[str, Any]) -> List[float]:
    """Runs dependency-free feedforward inference over input matrix X."""
    # Layer 1: Linear -> BatchNorm -> ReLU
    w1 = weights["network.0.weight"]
    b1 = weights["network.0.bias"]
    x1 = matmul(X, w1)
    x1 = add_bias(x1, b1)
    x1 = batchnorm1d(
        x1,
        weights["network.1.weight"],
        weights["network.1.bias"],
        weights["network.1.running_mean"],
        weights["network.1.running_var"]
    )
    x1 = relu(x1)
    
    # Layer 2: Linear -> BatchNorm -> ReLU
    w2 = weights["network.4.weight"]
    b2 = weights["network.4.bias"]
    x2 = matmul(x1, w2)
    x2 = add_bias(x2, b2)
    x2 = batchnorm1d(
        x2,
        weights["network.5.weight"],
        weights["network.5.bias"],
        weights["network.5.running_mean"],
        weights["network.5.running_var"]
    )
    x2 = relu(x2)
    
    # Layer 3: Linear
    w3 = weights["network.8.weight"]
    b3 = weights["network.8.bias"]
    x3 = matmul(x2, w3)
    x3 = add_bias(x3, b3)
    
    return [max(0.0, min(1.0, row[0])) for row in x3]


def evaluate_model(args):
    """Load model and dataset, calculate predictions, and print performance metrics."""
    if not os.path.exists(args.data_path):
        print(f"Dataset path '{args.data_path}' not found.")
        return
        
    if not os.path.exists(args.model_path):
        print(f"Model path '{args.model_path}' not found.")
        return
        
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)
        
    with open(args.model_path, 'r') as f:
        meta = json.load(f)
        
    val_X = dataset["val"]["X"]
    val_y = dataset["val"]["y"]
    
    if not val_X:
        print("Validation dataset is empty. Cannot evaluate.")
        return
        
    # Predict rewards
    predictions = run_inference(val_X, meta["weights"])
    
    # Compute overall performance stats
    mae = sum(abs(p - y) for p, y in zip(predictions, val_y)) / len(val_y)
    mse = sum((p - y) ** 2 for p, y in zip(predictions, val_y)) / len(val_y)
    rmse = math.sqrt(mse)
    
    # Compare expected vs actual rewards
    print("\n" + "="*50)
    print("Settings Optimizer Evaluation Report")
    print("="*50)
    print(f"Validation Samples: {len(val_y)}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE):  {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-"*50)
    
    # Top 5 predictions compared to actuals
    print("Sample predictions:")
    for i in range(min(5, len(val_y))):
        print(f"  Sample {i+1:02d}: Predicted Expected Reward = {predictions[i]:.4f} | Actual Reward = {val_y[i]:.4f}")
        
    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoSolve Settings Optimizer Evaluator")
    parser.add_argument("--data-path", default="ml/data/processed/settings_dataset.json", help="Path to processed JSON dataset")
    parser.add_argument("--model-path", default="ml/runs/settings_optimizer/model_meta_weights.json", help="Path to model meta weights file")
    
    parsed_args = parser.parse_args()
    evaluate_model(parsed_args)
