# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Track Quality Predictor inference module.
Uses pure numpy (bundled with Blender) to run inference on the trained MLP model.
"""

import os
import json
import numpy as np


class TrackPredictor:
    """Numpy-only Multi-Layer Perceptron (MLP) inference for track survival prediction."""
    
    def __init__(self):
        self.model = None
        self.input_dim = 15
        self.load_model()
        
    def load_model(self):
        """Load JSON weights from models directory and convert to numpy arrays."""
        try:
            # Load track predictor JSON model
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "tracker", "models", "track_predictor.json"
            )
            
            if os.path.exists(model_path):
                with open(model_path, 'r') as f:
                    self.model = json.load(f)
                    
                # Convert to numpy arrays for speed
                self.layer1_w = np.array(self.model["layer1_weight"], dtype=np.float32)
                self.layer1_b = np.array(self.model["layer1_bias"], dtype=np.float32)
                self.layer2_w = np.array(self.model["layer2_weight"], dtype=np.float32)
                self.layer2_b = np.array(self.model["layer2_bias"], dtype=np.float32)
                self.layer3_w = np.array(self.model["layer3_weight"], dtype=np.float32)
                self.layer3_b = np.array(self.model["layer3_bias"], dtype=np.float32)
                self.input_mean = np.array(self.model["input_mean"], dtype=np.float32)
                self.input_std = np.array(self.model["input_std"], dtype=np.float32)
                
                # Expose input dimension dynamically
                self.input_dim = len(self.input_mean)
                
                print(f"AutoSolve: Successfully loaded Track Quality Predictor model from {model_path} (input_dim={self.input_dim})")
            else:
                print(f"AutoSolve: No Track Quality Predictor model found at {model_path}. Predictions disabled.")
                self.model = None
        except Exception as e:
            print(f"AutoSolve: Failed to load track predictor model: {e}")
            self.model = None
            
    def predict_survival(self, features: np.ndarray) -> np.ndarray:
        """
        Predict survival probability for a batch of track features.
        
        Args:
            features: 2D numpy array of shape (N, 15)
            
        Returns:
            np.ndarray of shape (N,) containing survival probabilities in [0, 1] range.
        """
        if self.model is None or features.size == 0:
            return np.ones(features.shape[0], dtype=np.float32)
            
        try:
            # 1. Normalize input features
            x = (features - self.input_mean) / self.input_std
            
            # 2. Forward pass Layer 1 (Linear + ReLU)
            x = x @ self.layer1_w.T + self.layer1_b
            x = np.maximum(0.0, x)
            
            # 3. Forward pass Layer 2 (Linear + ReLU)
            x = x @ self.layer2_w.T + self.layer2_b
            x = np.maximum(0.0, x)
            
            # 4. Forward pass Layer 3 (Linear + Sigmoid)
            logits = x @ self.layer3_w.T + self.layer3_b
            probs = 1.0 / (1.0 + np.exp(-logits))
            
            return probs.flatten()
        except Exception as e:
            print(f"AutoSolve: Error during track quality prediction inference: {e}")
            return np.ones(features.shape[0], dtype=np.float32)
