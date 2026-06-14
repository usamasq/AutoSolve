# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
OnnxPredictor — Neural net inference using ONNX Runtime inside Blender.

Strategy:
  1. Try to import onnxruntime (present from bundled wheels, or installed fallback)
  2. Fall back gracefully to the existing numpy JSON-weight forward pass
  3. Cache InferenceSession so loading only happens once per Blender session

Models bundled in autosolve/tracker/models/:
  - settings_model.onnx   → maps (29 clip features) → expected reward [0,1]
  - track_predictor.onnx  → maps (15 track features) → survival prob [0,1]

Both are exported from the PyTorch checkpoints via ml/export_onnx.py.
"""

import os
import json
import numpy as np
from typing import Optional, Dict, Any

# ─── Optional ONNX Runtime import ─────────────────────────────────────────────
try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

PATCH_MODEL_ONNX    = os.path.join(_MODELS_DIR, "patch_rigidity.onnx")
SETTINGS_MODEL_ONNX = os.path.join(_MODELS_DIR, "settings_model.onnx")
TRACK_MODEL_ONNX    = os.path.join(_MODELS_DIR, "track_predictor.onnx")
PATCH_META_JSON     = os.path.join(_MODELS_DIR, "patch_rigidity_meta.json")
SETTINGS_META_JSON  = os.path.join(_MODELS_DIR, "settings_model_meta.json")
TRACK_META_JSON     = os.path.join(_MODELS_DIR, "track_predictor_meta.json")


# ═══════════════════════════════════════════════════════════════════════════
# ONNX SESSION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class _OnnxSession:
    """Thin wrapper around a single ONNX InferenceSession with normalisation."""

    def __init__(self, onnx_path: str, meta_path: str):
        self.session: Optional[Any] = None
        self.input_name: str = ""
        self.output_name: str = ""
        self.input_mean: Optional[np.ndarray] = None
        self.input_std:  Optional[np.ndarray] = None

        self._load(onnx_path, meta_path)

    def _load(self, onnx_path: str, meta_path: str):
        if not _ONNX_AVAILABLE:
            return
        if not os.path.exists(onnx_path):
            print(f"AutoSolve OnnxPredictor: model not found at {onnx_path}")
            return

        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(onnx_path, sess_options=opts,
                                                providers=["CPUExecutionProvider"])
            self.input_name  = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"AutoSolve OnnxPredictor: loaded {os.path.basename(onnx_path)}")

            # Load normalisation metadata
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self.input_mean = np.array(meta.get("input_mean", [0.0]), dtype=np.float32)
                self.input_std  = np.array(meta.get("input_std",  [1.0]), dtype=np.float32)
                # Guard against zero std
                self.input_std = np.where(self.input_std == 0, 1.0, self.input_std)

        except Exception as e:
            print(f"AutoSolve OnnxPredictor: failed to load {onnx_path}: {e}")
            self.session = None

    @property
    def available(self) -> bool:
        return self.session is not None

    def run(self, features: np.ndarray) -> np.ndarray:
        """
        Normalise features and run the ONNX session.

        Args:
            features: (N, D) float32 array

        Returns:
            (N,) float32 output array
        """
        if not self.available:
            raise RuntimeError("ONNX session is not available")

        x = features.astype(np.float32)
        
        # Validate input dimensions
        try:
            expected_shape = self.session.get_inputs()[0].shape
            if expected_shape and len(expected_shape) > 1:
                expected_dim = expected_shape[-1]
                # If shape is dynamic or text like 'batch_size', skip dimension check, otherwise validate
                if isinstance(expected_dim, int) and expected_dim > 0:
                    if x.shape[-1] != expected_dim:
                        raise ValueError(
                            f"ONNX Model input size mismatch: expected last dimension to be {expected_dim}, "
                            f"but got input of shape {x.shape}."
                        )
        except Exception as shape_err:
            if isinstance(shape_err, ValueError):
                raise shape_err
            print(f"AutoSolve OnnxPredictor: shape validation warning: {shape_err}")

        # Apply normalization if defined in metadata and shape matches
        if self.input_mean is not None and self.input_mean.shape == x.shape[1:]:
            x = (x - self.input_mean) / self.input_std

        output = self.session.run([self.output_name], {self.input_name: x})
        return np.array(output[0], dtype=np.float32).flatten()




# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC PREDICTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════

class OnnxPredictor:
    """
    Unified ONNX-based predictor for:
      - Track survival probability  (replaces TrackPredictor numpy MLP)
      - Expected tracking reward    (replaces SettingsPredictor heuristic)
      - Patch rigidity classification (semantic masking)

    Falls back silently to None returns when ONNX is unavailable so that
    callers can handle the fallback with their existing code paths.
    """

    _instance: Optional["OnnxPredictor"] = None  # Module-level singleton

    def __init__(self):
        self._track_session    = _OnnxSession(TRACK_MODEL_ONNX,    TRACK_META_JSON)
        self._settings_session = _OnnxSession(SETTINGS_MODEL_ONNX, SETTINGS_META_JSON)
        self._patch_session    = _OnnxSession(PATCH_MODEL_ONNX,    PATCH_META_JSON)

        if _ONNX_AVAILABLE:
            print(f"AutoSolve OnnxPredictor: onnxruntime {ort.__version__} ready. "
                  f"Track={self._track_session.available}, "
                  f"Settings={self._settings_session.available}, "
                  f"Patch={self._patch_session.available}")
        else:
            print("AutoSolve OnnxPredictor: onnxruntime not installed — using numpy fallback")

    # ── Singleton access ─────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "OnnxPredictor":
        """Return the module-level singleton, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Force reload of sessions (e.g. after installing onnxruntime)."""
        cls._instance = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def onnx_available(self) -> bool:
        return _ONNX_AVAILABLE

    @property
    def track_model_available(self) -> bool:
        return self._track_session.available

    @property
    def settings_model_available(self) -> bool:
        return self._settings_session.available

    @property
    def patch_model_available(self) -> bool:
        return self._patch_session.available

    def predict_patch_rigidity(self, patch: np.ndarray) -> Optional[float]:
        """
        Predict the rigidity of a 32x32 grayscale image patch.

        Args:
            patch: (32, 32) float32 numpy array with values in [0, 1]

        Returns:
            Rigidity score in [0, 1] (higher = static/rigid, lower = dynamic)
            or None if the patch model is not available.
        """
        if not self._patch_session.available:
            return None
        try:
            # Reshape to (1, 1, 32, 32) matching Conv2D input
            x = patch.reshape(1, 1, 32, 32).astype(np.float32)
            result = self._patch_session.run(x)
            return float(np.clip(result[0], 0.0, 1.0))
        except Exception as e:
            print(f"AutoSolve OnnxPredictor: patch rigidity inference failed: {e}")
            return None

    def predict_track_survival(self, features: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict survival probability for a batch of track feature vectors.

        Args:
            features: (N, 15) float32 array of track trajectory features

        Returns:
            (N,) float32 array of survival probabilities in [0, 1],
            or None if the ONNX model is unavailable (caller should fall back
            to the existing TrackPredictor numpy path).
        """
        if not self._track_session.available:
            return None
        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            probs = self._track_session.run(features)
            return np.clip(probs, 0.0, 1.0)
        except Exception as e:
            print(f"AutoSolve OnnxPredictor: track inference failed: {e}")
            return None

    def predict_expected_reward(self, clip_features: np.ndarray) -> Optional[float]:
        """
        Predict expected tracking reward from clip+settings feature vector.

        Args:
            clip_features: (29,) float32 vector of clip and settings features

        Returns:
            float reward in [0, 1], or None if model unavailable.
        """
        if not self._settings_session.available:
            return None
        try:
            x = clip_features.reshape(1, -1).astype(np.float32)
            result = self._settings_session.run(x)
            return float(np.clip(result[0], 0.0, 1.0))
        except Exception as e:
            print(f"AutoSolve OnnxPredictor: settings inference failed: {e}")
            return None

    def rank_settings_candidates(
        self,
        candidates: list,          # list of (settings_dict)
        feature_fn,                # callable: settings_dict → np.ndarray (29,)
    ) -> list:
        """
        Given a list of candidate settings dicts, score each with the ONNX
        model and return them sorted by predicted reward (best first).

        If the ONNX model is unavailable, returns candidates unchanged.

        Args:
            candidates:  List of settings dicts to rank
            feature_fn:  Function that converts a settings dict → feature vector

        Returns:
            Sorted list of (reward_score, settings_dict) tuples, best first.
        """
        if not self._settings_session.available or not candidates:
            return [(0.5, c) for c in candidates]

        try:
            features = np.stack([feature_fn(c) for c in candidates], axis=0)
            rewards  = self._settings_session.run(features)
            ranked   = sorted(zip(rewards.tolist(), candidates),
                               key=lambda x: x[0], reverse=True)
            return ranked
        except Exception as e:
            print(f"AutoSolve OnnxPredictor: ranking failed: {e}")
            return [(0.5, c) for c in candidates]


# ═══════════════════════════════════════════════════════════════════════════
# ONNX INSTALLATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def is_onnx_installed() -> bool:
    """Return True if onnxruntime is importable."""
    return _ONNX_AVAILABLE


def get_onnx_version() -> Optional[str]:
    """Return the installed onnxruntime version string, or None."""
    if _ONNX_AVAILABLE:
        return ort.__version__
    return None


def install_onnx_runtime(progress_callback=None) -> bool:
    """
    Silently install onnxruntime into Blender's Python using pip.

    Args:
        progress_callback: Optional callable(message: str) for UI feedback

    Returns:
        True if installation succeeded, False otherwise.
    """
    import subprocess
    import sys

    def _log(msg):
        print(f"AutoSolve Neural Engine: {msg}")
        if progress_callback:
            progress_callback(msg)

    _log("Installing onnxruntime (this is a one-time ~6MB download)...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "onnxruntime>=1.16.0",
             "--quiet", "--no-warn-script-location"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            _log("onnxruntime installed successfully!")
            # Force a reimport attempt
            OnnxPredictor.reset_instance()
            return True
        else:
            _log(f"pip install failed: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        _log("Installation timed out after 120s. Check your internet connection.")
        return False
    except Exception as e:
        _log(f"Installation error: {e}")
        return False
