# AutoSolve Development Log

This document chronicles the development journey of AutoSolve from its initial cleanup phase to a robust, intelligent, and entirely offline Blender-native camera tracking assistant.

---

## Milestone 1: Telemetry Removal & Public Cleanup
* **Goal**: Surgical removal of user-facing telemetry and inflated claims, transitioning to a deterministic, offline-first codebase.
* **Key Achievements**:
  * Deleted `session_recorder.py` and `behavior_recorder.py` to stop all user tracking and background session logging.
  * Restructured the repository by moving purely algorithmic modules (`track_healer.py` and `failure_diagnostics.py`) out of the learning directory to `tracker/`.
  * Removed telemetry-related UI components, operator registration, and property descriptors (e.g. `record_edits`).
  * Updated project documentation (`README.md`, `ARCHITECTURE.md`) to reflect the new layout and honest positioning.

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

---

## Milestone 7: QA Audit, ML Upgrades & Mixin Refactoring
* **Goal**: Refactor the massive `smart_tracker.py` file, improve machine learning quality, fold batch normalization, and resolve all systematic bugs.
* **Key Achievements**:
  * **SmartTracker Modularization**: De-bloated the god object `smart_tracker.py` (originally >3,900 lines) by extracting functional domains into five clean mixin classes: `probe_cache.py`, `detection.py`, `strategic.py`, `learning.py`, and `cleanup.py`.
  * **ML Data Leakage Prevention**: fit scaling parameters (`StandardScaler`) strictly on the training folds and grouped train/validation splits by Clip ID to prevent evaluation correlation.
  * **Training Loop Optimization**: Upgraded PyTorch training configurations across standalone scripts and the Jupyter Notebook:
    * Standardized models (`SettingsMLP`, `TrackMLP`, `PatchRigidityCNN`) with `nn.BatchNorm` and `nn.Dropout`.
    * Implemented `AdamW` (weight decay `1e-4`), gradient clipping (`max_norm=1.0`), and `CosineAnnealingLR` learning rate scheduling.
    * Added Gaussian noise ($\sigma=0.01$) to continuous features as a data regularizer.
  * **Numpy Parameter Folding**: Developed folding math inside the exporters to collapse BatchNorm parameters directly into linear layer weights and biases, allowing the addon's numpy inference engine to run the upgraded PyTorch models with zero overhead.
  * **Algorithmic Enhancements**: Upgraded `track_healer.py` to Cubic Hermite Spline interpolation for smooth velocity-preserving track gap filling. Added relative adaptive velocity spike thresholds in `validation.py` to support fast camera pans.
  * **Verification Suite**: Created [test_smart_tracker.py](file:///c:/Users/usama/OneDrive/Desktop/AutoSolve/scratch/test_smart_tracker.py) with mocked Blender namespaces to verify mixin binding and neural engine prediction correctness. All tests compile and execute successfully.

---

## Milestone 8: Hollywood-Grade Matchmoving
* **Goal**: Implement deep-learning dense tracking (CoTracker v3), semantic masking (YOLOv8), and a custom bundle adjuster (SciPy) via an out-of-process background worker with offline bundling.
* **Key Achievements**:
  * **Client-Worker IPC Server:** Implemented a multi-threaded TCP/JSON-IPC socket server (`autosolve/worker/server.py`) and Blender client (`autosolve/worker/client.py`) running in the system Python to avoid blocking Blender's single-thread UI.
  * **CoTracker v3 Offline Tracking:** Bundled the CoTracker source repository and v3 offline model weights (`cotracker3_offline.pth` ~100MB) locally, loading the model offline via PyTorch Hub (`source="local"`) with a custom key-prefix mapping adapter.
  * **YOLOv8 Semantic Masking:** Bundled `yolov8n-seg.pt` (~7MB) for dynamic object masking (people, vehicles, animals), preventing tracks from instantiating on non-rigid elements.
  * **Precision SciPy Solver:** Created a multi-pass bundle adjustment solver (`autosolve/worker/ba_solver.py`) using `scipy.optimize.least_squares` with robust Cauchy loss via Trust Region Reflective (`trf`) optimization.
  * **Offline Wheels Bundling:** Restored pre-compiled `onnxruntime` wheels under `wheels/` and re-declared them in `blender_manifest.toml` for automatic zero-download installation at addon setup time.
  * **Verification:** Built `test_worker.py` to verify the IPC server, precision solver convergence, and clean shutdown protocol. All tests pass successfully.

---

## Milestone 9: Stability Audit & Native Blender Integration Gaps
* **Goal**: Conduct an audit of data portability, process stability, and coordinate precision between the out-of-process worker and Blender's native tracking system, and fix all identified integration gaps.
* **Key Achievements**:
  * **Timeline Frame Alignment**: Fixed a 1-frame offset bug in `import_external_trajectories` where 0-indexed trajectory points were shifted by -1 frame on the Blender timeline.
  * **Y-Coordinate Inversion**: Resolved the coordinate space mismatch (CoTracker's top-down vs Blender's bottom-up) by inverting normalized y-coordinates (`1.0 - ny`).
  * **Aspect-Ratio Preserving Normalization**: Modified the frame loading and resizing loops in CoTracker to pad widescreen (16:9) and vertical (9:16) footage with black letterbox/pillarbox margins (fitting `288x384`), and updated normalization math to reverse these padding offsets (`nx = (px - pad_w) / new_w` and `ny = (py - pad_h) / new_h`) for distortion-free tracking. This logic was also integrated into the dataset generation scripts (`ml/extract_cotracker_trajectories.py`) and the Google Colab training notebook (`ml/AutoSolve_Training.ipynb`) to ensure coordinate representation parity between training and runtime.
  * **VFR & Framerate Validation**: Added a non-blocking check during operator execution comparing clip FPS with scene render FPS, displaying a Blender warning banner on mismatch to alert users of potential solve drift.
  * **Background Task Robustness**: Implemented 90-second timeouts and clean TCP/process termination (`kill_worker`) for SAM2 and CoTracker background workers to ensure Blender's UI never hangs.
