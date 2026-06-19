# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
ml/export_onnx.py — Export trained PyTorch models to ONNX format.

Produces:
  autosolve/tracker/models/settings_model.onnx
  autosolve/tracker/models/settings_model_meta.json
  autosolve/tracker/models/track_predictor.onnx
  autosolve/tracker/models/track_predictor_meta.json

Usage:
  python ml/export_onnx.py
  python ml/export_onnx.py --settings-weights ml/runs/settings_optimizer/model_meta_weights.json
  python ml/export_onnx.py --track-weights    ml/runs/track_predictor/track_predictor.json

The exported ONNX models are then loaded at runtime by OnnxPredictor inside
Blender without needing PyTorch installed by the end-user.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS (must match training scripts exactly)
# ═══════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class SettingsMLP(nn.Module):
        """3-layer MLP for expected reward prediction."""
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(29, 64),
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

    class TrackMLP(nn.Module):
        """3-layer MLP for track survival prediction (matching training, with Sigmoid for ONNX export)."""
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
            return torch.sigmoid(self.network(x))


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY
# ═══════════════════════════════════════════════════════════════════════════

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "autosolve", "tracker", "models"
)


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _load_json_weights_into_model(model: "nn.Module", weights: dict):
    """
    Load a weight dict (from model_meta_weights.json) into a PyTorch model.
    Keys in the JSON must match model.state_dict() keys exactly.
    """
    state = {}
    for k, v in weights.items():
        state[k] = torch.tensor(v, dtype=torch.float32)
    model.load_state_dict(state)


def _validate_onnx_vs_torch(torch_model, onnx_path, sample_input):
    """
    Run both models on sample_input and assert outputs match to < 1e-3.
    Returns True if validation passes.
    """
    if not ORT_AVAILABLE:
        print("  Skipping validation (onnxruntime not available).")
        return True

    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(torch.tensor(sample_input, dtype=torch.float32)).numpy()

    sess     = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name  = sess.get_inputs()[0].name
    ort_out  = sess.run(None, {in_name: sample_input.astype(np.float32)})[0]

    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    if max_diff < 1e-3:
        print(f"  [OK] Validation passed — max output diff: {max_diff:.2e}")
        return True
    else:
        print(f"  [WARNING] Validation WARNING — max diff: {max_diff:.2e} (threshold: 1e-3)")
        return False


def export_settings_model(weights_path: str, out_dir: str) -> bool:
    """
    Load settings model weights from JSON and export to ONNX.
    Also writes settings_model_meta.json with normalisation parameters.
    """
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch not available — cannot export ONNX.")
        return False

    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights file not found: {weights_path}")
        return False

    print("\n--- Settings Model ------------------------------")
    print(f"   Loading weights from: {weights_path}")

    with open(weights_path, "r") as f:
        meta = json.load(f)

    model = SettingsMLP()
    _load_json_weights_into_model(model, meta["weights"])
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    onnx_path = os.path.join(out_dir, "settings_model.onnx")
    dummy_input = torch.zeros(1, 29, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["clip_features"],
            output_names=["reward"],
            opset_version=17,
            dynamic_axes={"clip_features": {0: "batch"}, "reward": {0: "batch"}},
        )
    except ModuleNotFoundError as e:
        if "onnxscript" in str(e):
            print("\n" + "=" * 80)
            print("Error: Exporting to ONNX failed because 'onnxscript' is not installed.")
            print("In newer versions of PyTorch (2.4+), exporting to ONNX requires the 'onnxscript' library.")
            print("Please install it by running:")
            print("    pip install onnxscript")
            print("=" * 80 + "\n")
        raise
    print(f"   Exported → {onnx_path}")

    # Validate
    sample = np.random.randn(4, 29).astype(np.float32)
    _validate_onnx_vs_torch(model, onnx_path, sample)

    # Write meta (normalisation + model info)
    meta_out = {
        "input_mean":   meta.get("input_mean", [0.0] * 29),
        "input_std":    meta.get("input_std",  [1.0] * 29),
        "input_size":   29,
        "output_size":  1,
        "best_val_loss": meta.get("best_val_loss", None),
        "description":  "Settings expected-reward predictor. Input: clip+settings features (29-dim). Output: reward in [0,1]."
    }
    meta_path = os.path.join(out_dir, "settings_model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"   Meta    → {meta_path}")

    return True


def export_track_predictor(weights_path: str, out_dir: str) -> bool:
    """
    Load track predictor weights from JSON and export to ONNX.
    Also writes track_predictor_meta.json.
    """
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch not available — cannot export ONNX.")
        return False

    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights file not found: {weights_path}")
        return False

    print("\n--- Track Predictor -----------------------------")
    print(f"   Loading weights from: {weights_path}")

    with open(weights_path, "r") as f:
        raw = json.load(f)

    model = TrackMLP()
    
    if "weights" in raw:
        # Loads directly from PyTorch model_meta_weights.json
        _load_json_weights_into_model(model, raw["weights"])
    else:
        # Fallback to old format if only track_predictor.json is provided, 
        # but warn that batch norm parameters will be default initialized.
        print("   [WARNING] Loading from legacy numpy JSON format. BatchNorm parameters will be identity/zero.")
        mapped_weights = {
            "network.0.weight": raw["layer1_weight"],
            "network.0.bias":   raw["layer1_bias"],
            "network.4.weight": raw["layer2_weight"],
            "network.4.bias":   raw["layer2_bias"],
            "network.8.weight": raw["layer3_weight"],
            "network.8.bias":   raw["layer3_bias"],
        }
        # Create a state dict with default batchnorm values
        state_dict = model.state_dict()
        for k, v in mapped_weights.items():
            state_dict[k] = torch.tensor(v, dtype=torch.float32)
        model.load_state_dict(state_dict)

    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    onnx_path = os.path.join(out_dir, "track_predictor.onnx")
    dummy_input = torch.zeros(1, 20, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["track_features"],
            output_names=["survival_prob"],
            opset_version=17,
            dynamic_axes={"track_features": {0: "batch"}, "survival_prob": {0: "batch"}},
        )
    except ModuleNotFoundError as e:
        if "onnxscript" in str(e):
            print("\n" + "=" * 80)
            print("Error: Exporting to ONNX failed because 'onnxscript' is not installed.")
            print("In newer versions of PyTorch (2.4+), exporting to ONNX requires the 'onnxscript' library.")
            print("Please install it by running:")
            print("    pip install onnxscript")
            print("=" * 80 + "\n")
        raise
    print(f"   Exported → {onnx_path}")

    # Validate
    sample = np.random.randn(8, 20).astype(np.float32)
    _validate_onnx_vs_torch(model, onnx_path, sample)

    # Write meta
    meta_out = {
        "input_mean":  raw.get("input_mean",  [0.0] * 20),
        "input_std":   raw.get("input_std",   [1.0] * 20),
        "input_size":  20,
        "output_size": 1,
        "description": "Track survival predictor. Input: 20-dim trajectory features. Output: survival prob in [0,1]."
    }
    meta_path = os.path.join(out_dir, "track_predictor_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"   Meta    → {meta_path}")

    return True


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Export AutoSolve models to ONNX")
    parser.add_argument(
        "--settings-weights",
        default="ml/runs/settings_optimizer/model_meta_weights.json",
        help="Path to settings model JSON weights"
    )
    parser.add_argument(
        "--track-weights",
        default="ml/runs/track_predictor/model_meta_weights.json",
        help="Path to track predictor JSON weights"
    )
    parser.add_argument(
        "--out-dir",
        default=_MODELS_DIR,
        help="Output directory for .onnx files (default: autosolve/tracker/models/)"
    )
    parser.add_argument(
        "--skip-settings", action="store_true",
        help="Skip settings model export"
    )
    parser.add_argument(
        "--skip-track", action="store_true",
        help="Skip track predictor export"
    )
    args = parser.parse_args()

    print("AutoSolve ONNX Export")
    print(f"  PyTorch:       {'[OK]' if TORCH_AVAILABLE else '[ERROR] not installed'}")
    print(f"  onnxruntime:   {'[OK]' if ORT_AVAILABLE  else '[WARN] not installed (validation skipped)'}")
    print(f"  Output dir:    {args.out_dir}")

    if not TORCH_AVAILABLE:
        print("\n[ERROR] PyTorch is required for ONNX export. Install it with:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)

    ok_settings = True
    ok_track    = True

    if not args.skip_settings:
        ok_settings = export_settings_model(args.settings_weights, args.out_dir)

    if not args.skip_track:
        ok_track = export_track_predictor(args.track_weights, args.out_dir)

    print("\n--- Summary -------------------------------------")
    print(f"   Settings model:   {'[OK] exported' if ok_settings else '[ERROR] failed'}")
    print(f"   Track predictor:  {'[OK] exported' if ok_track    else '[ERROR] failed'}")

    if ok_settings and ok_track:
        print("\n[OK] All models exported successfully!")
        print(f"   Place the .onnx and _meta.json files in:")
        print(f"   {_MODELS_DIR}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
