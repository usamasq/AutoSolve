# AutoSolve Development Log

This document chronicles the development journey of AutoSolve from its initial cleanup phase to a robust, intelligent, and entirely offline Blender-native camera tracking assistant.

---

## Milestone 1: Telemetry Removal & Public Cleanup
* **Goal**: Surgical removal of user-facing telemetry and inflated claims, transitioning to a deterministic, offline-first codebase.
* **Key Achievements**:
  * Deleted `session_recorder.py` and `behavior_recorder.py` to stop all user tracking and background session logging.
  * Restructured the repository by moving purely algorithmic modules (`track_healer.py` and `failure_diagnostics.py`) out of the learning directory to `tracker/`.
  * Removed telemetry-related UI components, operator registration, and property descriptors (e.g. `record_edits`).
  * Updated all project documentation (`README.md`, `ARCHITECTURE.md`, `CONTRIBUTING_DATA.md`, `TRAINING_DATA.md`) to reflect the new layout and honest positioning.

---

## Milestone 2: Usability & UX Improvements
* **Goal**: Enable direct, actionable controls for tracking refining, scene setup, and footage presets.
* **Key Achievements**:
  * **Solve Summary Report**: Placed an active report panel inside the Step 2 scene setup panel, showing metrics like count, survival rate, final reprojection error, and a computed star rating.
  * **Select High Error Tracks Operator**: Implemented a slider-based selector to instantly select tracks exceeding a pixel error threshold (e.g. `> 1.5px`) for easy deletion.
  * **Clean & Re-Solve Operator**: Added a button that performs automated cleanups and resolves the camera without needing a full re-track sequence.
  * **Resolution Matching**: Refactored the scene setup operator to automatically synchronize render dimensions and frame ranges to the source movie clip characteristics.
  * **SCREEN and CINEMATIC presets**: Registered new footage type preset tables handling screen captures and cinematic lens characteristics.

---

## Milestone 3: Developer Data Collection Pipeline
* **Goal**: Establish an offline developer-side data collection pipeline to train model weights without user telemetry.
* **Key Achievements**:
  * Defined data serialization schemas (`ml/schema.py`) detailing clips, settings, trajectory positions, and solve success outcomes.
  * Created `ml/collect_data.py` to run headless simulation tracks inside Blender, recording samples.
  * Built `ml/run_collection.py` to batch run clips using settings variations and generate datasets.
  * Implemented validation and summary scripts (`ml/validate_data.py`) to verify collection output.

---

## Milestone 4: Track Quality Predictor
* **Goal**: Build a machine learning module to predict track failure and proactively replenish tracks.
* **Key Achievements**:
  * **Trajectory Feature Engineering**: Extracted coordinate-based properties (velocities, standard deviations/jitter proxies, accelerations, nearest neighbors, etc.).
  * **PyTorch MLP**: Designed and trained a 3-layer MLP network (15 -> 64 -> 32 -> 1) predicting the probability of track survival in the next 20 frames.
  * **Numpy Inference Engine**: Ported feedforward model weights to a lightweight JSON structure (`track_predictor.json`) loaded at runtime using Blender's built-in `numpy` library.
  * **Proactive Replenishment**: Intercepted the tracking loop to predict weak tracks, muting and replacing them before they degrade the camera solve.

---

## Milestone 5: Content-Aware Feature Placement
* **Goal**: Direct marker placement to highly-textured regions using image complexity metrics.
* **Key Achievements**:
  * **Texture Quality Estimation**: Wrote a sparse pixel grid sampler that loads a temporary frame buffer, divides the screen into 3x3 grids, and computes local luminance variance on 8x8 sample points.
  * **Skipping Uniform Areas**: Configured target calculations to automatically skip regions with near-uniform complexity (such as clear skies or flat solid walls).
  * **Dynamic Count Scaling**: Integrated dynamic quality estimation with distilled empirical track survival weights (`region_weights.json`) to scale region target counts on the fly.
  * **Memory Safety**: Designed strict cleanup handlers that purge all temporary image datablocks (`__autosolve_temp_*`) to avoid memory leaks.

---

## Milestone 6: Settings Optimizer
* **Goal**: Predict optimal tracking settings combinations using expected reward estimation.
* **Key Achievements**:
  * **Dataset Preprocessor**: Developed `ml/prepare_dataset.py` to extract 24-dimensional feature vectors and calculate solve rewards: `success * (1 - clamp(error/5, 0, 1)) * bundle_ratio`.
  * **PyTorch Trainer**: Implemented `ml/train_settings_model.py` to predict expected reward and output weights.
  * **Dependency-Free Evaluation**: Wrote standard Python matrix multiplication routines in `ml/evaluate_model.py` to run inference without needing NumPy/PyTorch.
  * **Presets Grid Search**: Built `ml/export_defaults.py` to search a parameter grid of 2,016 options per clip type, writing recommendations to `recommended_defaults.json` for developer review.

---

## Developer Environment Dependency Fallbacks
All developer-side training and preprocessing pipelines inside the `ml/` directory support zero-dependency fallback:
* **Trackability Trainer**: Reverts to pre-baked default weights if no raw tracking logs are present.
* **Dataset Preprocessing**: Synthesizes structured data samples for testing.
* **Expected Reward MLP**: Generates heuristic-based weights when PyTorch is not available.
* **Evaluation & Exporter**: Runs list-based feedforward calculations natively without requiring any external matrix library.
