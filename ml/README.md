# AutoSolve Model Training Guide

This guide describes how to run the data collection pipeline and train/evaluate the machine learning models used in AutoSolve.

All training scripts support **zero-dependency fallback modes** which generate heuristic defaults if PyTorch/NumPy are missing from your global environment.

---

## Workflow Overview

```mermaid
graph TD
    Clips[1. Reference Clips] -->|ml/run_collection.py| RawJSON[2. Raw JSON Samples]
    RawJSON -->|ml/train_trackability_model.py| RegionWeights[3. region_weights.json]
    RawJSON -->|ml/train_track_predictor.py| PyTorchTrack[4. Track Predictor Weights]
    PyTorchTrack -->|ml/export_numpy_model.py| PredictorJSON[5. track_predictor.json]
    RawJSON -->|ml/prepare_dataset.py| ProcessedDataset[6. Settings Dataset]
    ProcessedDataset -->|ml/train_settings_model.py| PyTorchSettings[7. Settings Model Weights]
    PyTorchSettings -->|ml/evaluate_model.py| Evaluation[8. Performance Evaluation]
    PyTorchSettings -->|ml/export_defaults.py| PresetsReport[9. recommended_defaults.json]
```

---

## 1. Setup Reference Clips
To build a dataset, you must gather a set of CC-licensed tracking reference clips (covering different movement characteristics, panning, and resolutions). Refer to the guidelines in [ml/clips/README.md](file:///c:/Users/usama/OneDrive/Desktop/AutoSolve/ml/clips/README.md) to categorize your footage. Place clips in a local directory (e.g. `ml/clips/`).

---

## 2. Simulated Data Collection
Run the batch simulation runner. This script launches Blender headlessly, performs simulated tracking sweeps with varying parameters, and logs solve results.

```bash
# Run batch tracking simulation over clips directory
python ml/run_collection.py --clips-dir ml/clips/ --out-dir ml/data/raw/
```

### Validate Raw Dataset
Verify the integrity of the collected solve logs:
```bash
python ml/validate_data.py --data-dir ml/data/raw/
```

---

## 3. Train Trackability Heatmap / Region Weights (Milestone 5)
Aggregate region survival rates from the collected tracking runs to generate the region trackability table:

```bash
# Aggregates raw logs and writes to the addon's presets directory
python ml/train_trackability_model.py --data-dir ml/data/raw/ --output-json autosolve/tracker/presets/region_weights.json
```

---

## 4. Train Track Quality Predictor (Milestone 4)
Trains the neural network that predicts if an active track is likely to fail in the next 20 frames.

### Step 4a: Train PyTorch MLP
```bash
# Trains 15 -> 64 -> 32 -> 1 classification MLP
python ml/train_track_predictor.py --data-dir ml/data/raw/ --out-dir ml/runs/track_predictor/ --epochs 100
```

### Step 4b: Export to NumPy JSON Weights
Export the trained state dict to standard JSON arrays for native NumPy runtime inference:
```bash
python ml/export_numpy_model.py --input-json ml/runs/track_predictor/model_meta_weights.json --output-path autosolve/tracker/models/track_predictor.json
```

---

## 5. Train Settings Optimizer (Milestone 6)
Fits the expected reward model that predicts solve quality given candidate parameters.

### Step 5a: Preprocess and Scale Dataset
Extracts 24-dimensional feature vectors, calculates solve rewards, and normalizes inputs:
```bash
python ml/prepare_dataset.py --data-dir ml/data/raw/ --output-json ml/data/processed/settings_dataset.json
```

### Step 5b: Train Expectation Model
Trains the expected reward MLP:
```bash
python ml/train_settings_model.py --data-path ml/data/processed/settings_dataset.json --out-dir ml/runs/settings_optimizer/ --epochs 50
```

### Step 5c: Evaluate Model Performance
Validate predictions and check MAE/MSE accuracy on the validation split:
```bash
python ml/evaluate_model.py --data-path ml/data/processed/settings_dataset.json --model-path ml/runs/settings_optimizer/model_meta_weights.json
```

### Step 5d: Run Presets Grid Search & Recommendation
Performs parameter search grid sweeps using the MLP predictor to select optimal presets per clip type:
```bash
python ml/export_defaults.py --model-path ml/runs/settings_optimizer/model_meta_weights.json --output-json ml/runs/settings_optimizer/recommended_defaults.json
```
*Note: Developers should manually review recommendations inside `recommended_defaults.json` and merge them into `PRETRAINED_DEFAULTS` inside `autosolve/tracker/constants.py`.*
