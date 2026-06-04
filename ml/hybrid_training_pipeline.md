# AutoSolve Hybrid ML Pipeline: Deep Training & Addon Testing

This architecture details the hybrid loop where models are trained directly on video files using deep learning libraries in Jupyter, and then tested/validated inside Blender using the AutoSolve addon.

---

## 🔁 The Hybrid Closed-Loop Architecture

```mermaid
graph TD
    %% Training Phase (Jupyter / Colab)
    subgraph Phase 1: Deep Training (Jupyter)
        Videos[Raw Video Library] -->|RAFT Optical Flow| Zoom[Zoom & Speed Divergence]
        Videos -->|Hough Transform| Distortion[Lens Distortion Curvature]
        Videos -->|SuperPoint / CoTracker| Trajectories[Deep Track Dynamics]
        
        Zoom & Distortion & Trajectories -->|PyTorch training| PyTorchModels[Trained Weights]
        PyTorchModels -->|export_onnx.py| ONNXModels[settings_model.onnx & track_predictor.onnx]
    End

    %% Testing & Validation Phase (Blender)
    subgraph Phase 2: Addon Testing (Blender & AutoSolve)
        ONNXModels -->|Deploy to Addon| AutoSolve[AutoSolve Neural Engine]
        Videos -->|run_collection.py| HeadlessBlender[Headless Blender Solve]
        AutoSolve -->|Steers solver settings| HeadlessBlender
        HeadlessBlender -->|Compute actual metrics| SolveStats[Solve Stats: Reprojection Error, Bundle Ratio]
    End

    %% Feedback Loop
    SolveStats -->|Benchmark Feedback| Phase 1
```

---

## 🛠️ Phase 1: Deep Training (Jupyter Notebook)

By bypassing Blender for dataset generation, we train on raw video files at 10x speed using specialized computer vision models:

1.  **Divergence Analysis (Zooms):**
    *   Compute dense optical flow using **RAFT** or **Farneback**.
    *   Calculate vector divergence:
        $$\text{Div} = \frac{\partial V_x}{\partial x} + \frac{\partial V_y}{\partial y}$$
    *   High divergence indicates camera zoom, signaling the need for variable focal length solving.

2.  **Curvature Analysis (Lens Distortion):**
    *   Use OpenCV **Canny + HoughLinesP** to detect linear structures.
    *   Measure line curvature near image borders to quantify barrel/pincushion lens distortion.

3.  **Trajectory Survival (Occlusions):**
    *   Generate tracker points using **CoTracker** to simulate multi-frame trajectories.
    *   Inject synthetic noise and occlusions directly in the pixel coordinate space.

---

## 🧪 Phase 2: Testing and Validation (AutoSolve in Blender)

To verify the trained models actually translate to better camera tracks, we use AutoSolve inside Blender as our benchmarking suite.

1.  **Headless Solving:**
    *   Launch Blender in background mode using `run_collection.py`.
    *   AutoSolve loads the newly trained `settings_model.onnx` to predict and apply tracking parameters.
    *   AutoSolve runs `track_predictor.onnx` to monitor and prune bad tracks during solver execution.

2.  **Performance Metrics:**
    The solve returns three concrete ground-truth metrics:
    *   **Average Reprojection Error (px):** Target $< 0.5\text{px}$.
    *   **Bundle Ratio:** The percentage of tracked markers successfully reconstructed in 3D (Target $> 60\%$).
    *   **Solve Time (s):** Benchmark speed and efficiency.

3.  **Regression Benchmarking:**
    We compare the new model's solve stats against:
    *   Traditional heuristic trackers.
    *   Previous iterations of the neural engine.
    If the solve stats show regression, we refine features or hyper-parameters in the Jupyter Phase.
