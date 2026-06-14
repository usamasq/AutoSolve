# AutoSolve Architecture & Technical Design

AutoSolve is a professional matchmoving assistant for Blender. It features a dual-engine architecture: a **zero-dependency, native Blender engine** and an **out-of-process, GPU-accelerated AI worker** running completely offline.

---

## 1. Architectural Core

AutoSolve is built as a **Blender Extension (Addon)** using the Blender Python API (`bpy`). The architecture separates the lightweight UI/orchestration from heavy execution tasks to avoid blocking Blender's single-threaded UI loop:

```
┌──────────────────────────────────────────────────────────────────┐
│                          Blender Process                         │
│  ┌───────────────────────┐              ┌─────────────────────┐  │
│  │     Addon UI & Ops    │              │    Local Predict    │  │
│  │ (autosolve/operators) │              │  (onnx_predictor)   │  │
│  └───────────┬───────────┘              └──────────▲──────────┘  │
│              │                                     │             │
│              │ Spawns & Polls                      │ Fallback    │
│              ▼                                     │             │
│  ┌───────────────────────┐              ┌──────────┴──────────┐  │
│  │    JSON-IPC Client    │─────────────>│  ONNX Runtime /     │  │
│  │  (autosolve/client)   │  (Imports)   │   Numpy Engine      │  │
│  └───────────┬───────────┘              └─────────────────────┘  │
└──────────────┼───────────────────────────────────────────────────┘
               │
               │ JSON-IPC over TCP (localhost:47832)
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     System Python Process                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                     Background Worker                       │  │
│  │                (autosolve/worker/server.py)                 │  │
│  └───────────┬─────────────────────────────────────┬───────────┘  │
│              │                                     │              │
│              ▼                                     ▼              │
│  ┌───────────────────────┐              ┌─────────────────────┐  │
│  │     AI Inference      │              │    SciPy Solver     │  │
│  │   (CoTracker, YOLO)   │              │   (least_squares)   │  │
│  └───────────────────────┘              └─────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Key Core Layers:
1. **Blender Client Engine:** Implemented in pure Python. It collects Blender tracking datablocks, serializes coordinates to JSON, manages worker subprocess life cycles, and runs a non-blocking modal timer loop that polls socket outputs.
2. **Asynchronous TCP Worker:** A lightweight, multi-threaded JSON-IPC socket server (`autosolve/worker/server.py`) written in standard Python. It listens on port `47832` and coordinates PyTorch, OpenCV, and SciPy calls.
3. **Local ONNX Predictors:** High-performance, lightweight models running directly inside Blender to estimate track quality, settings expected rewards, and patch rigidity.

---

## 2. Technology Stack & Model Specifications

AutoSolve uses a curated stack of machine learning frameworks, libraries, and custom solvers:

### A. Background Worker (AI Backends)
* **PyTorch (2.12.0) & OpenCV (4.13.0):** Used for tensor computations and frame extraction/resize operations.
* **CoTracker v3 Offline (`cotracker3_offline.pth` ~100MB):** A transformer-based dense tracking checkpoint. It tracks points jointly over 60-frame sliding windows. The source repository is bundled locally in `autosolve/worker/cotracker_src/`, allowing PyTorch Hub to initialize the model completely offline with `source="local"`. A prefix adapter handles mapping the checkpoint keys to the internal class attributes. To prevent optics distortion during scaling, video frames are padded with black margins (letterbox/pillarbox) to fit the `288x384` target canvas. During output coordinate mapping, the padding offsets are subtracted and normalized relative to the aspect-ratio-scaled dimensions, returning 100% distortion-free coordinates.
* **YOLOv8 Segmentation (`yolov8n-seg.pt` ~7MB):** Ultralytics segmenter used frame-by-frame. It isolates moving elements (people, vehicles, animals) and returns normalized bounding boxes. The client filters out any tracking coordinates falling inside these boxes.
* **SciPy Bundle Adjuster (`ba_solver.py`):** Uses non-linear least-squares optimization (`scipy.optimize.least_squares`) with robust **Cauchy loss** via the **Trust Region Reflective (TRF)** method. It resolves camera poses in a multi-pass routine (poses → focal length → radial distortion → global adjustment).

### B. Blender Internal Engine (Local Predictors)
* **ONNX Runtime (1.26.0):** Executed inside Blender's Python. Pre-compiled wheel files (`.whl`) for Windows, macOS, and Linux are bundled under `wheels/` and automatically installed during addon registration.
* **Numpy Fallback Engine:** A custom, dependency-free matrix computation library written in pure Python/NumPy. It automatically executes forward passes of the local MLP networks if ONNX Runtime is missing.

---

## 3. Local Machine Learning Models

AutoSolve implements three lightweight local networks trained to automate heuristic settings and validation:

| Model | Architecture | Inputs | Output | Purpose |
| --- | --- | --- | --- | --- |
| **Track Quality Predictor** | 3-Layer MLP | 15 trajectory features (velocity, jitter, neighborhood displacement) | Survival probability $P(\text{survival})$ | Proactively terminates and replenishes drifting tracks every 10 frames. |
| **Settings Optimizer** | 3-Layer MLP | 28 clip & configuration features | Expected Solve Reward | Ranks 2,016 parameter combinations to select the best tracking preset. |
| **Patch Rigidity Classifier** | Convolutional NN | 32x32 grayscale image patch | Rigidity Score $[0, 1]$ | Detects local non-rigid regions (foliage, water) when YOLOv8 is not in use. |

---

## 4. ML Training & ONNX Pipeline

### Data Collection & Training
1. **Telemetry-Free Collection:** Developers use `ml/collect_data.py` to run headless simulation tracks inside Blender, saving telemetry data (trajectories, presets, solve errors, and gap statistics) into structured JSON files defined by `ml/schema.py`.
2. **Training Notebook:** The PyTorch pipeline ([AutoSolve_Training.ipynb](file:///c:/Users/usama/OneDrive/Desktop/AutoSolve/ml/AutoSolve_Training.ipynb)) trains the networks with `AdamW` regularization, learning rate scheduling (`CosineAnnealingLR`), and validation splits grouped by Clip ID to prevent data leakage.
3. **Export to ONNX:** The trained PyTorch modules are exported to `.onnx` and JSON weights format.

### BatchNorm Parameter Folding
To run the PyTorch models inside the zero-dependency Numpy fallback engine, the exporters mathematically **fold** Batch Normalization parameters directly into the weights and biases of their preceding Linear (or Conv) layers.

For a linear layer $Y = W \cdot X + B$ followed by a batch norm layer:
$$Y_{\text{norm}} = \gamma \cdot \frac{Y - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

We fold the mean ($\mu$), variance ($\sigma^2$), scale factor ($\gamma$), shift ($\beta$), and epsilon ($\epsilon$) into folded weights ($W_{\text{fold}}$) and biases ($B_{\text{fold}}$):

$$W_{\text{fold}} = W \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}$$

$$B_{\text{fold}} = (B - \mu) \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

This mathematical simplification allows the addon's Numpy engine to execute the fully regularized model with zero computational overhead and no library dependencies.

---

## 5. System Execution Lifecycle

```
[Blender UI]                 [operators.py]               [client.py]             [worker/server.py]
     │                             │                           │                           │
     │ Click "Auto-Track & Solve"  │                           │                           │
     ├────────────────────────────>│                           │                           │
     │                             │ Validate Scene/Clip FPS   │                           │
     │                             │ (Report warning mismatch) │                           │
     │                             │ Spawn Worker (Subprocess) │                           │
     │                             ├──────────────────────────>│                           │
     │                             │                           │ TCP Bind & Listen         │
     │                             │                           ├──────────────────────────>│
     │                             │ WAITING_FOR_WORKER_TRACK  │                           │
     │                             ├──────────────────────────>│                           │
     │                             │                           │ Send "cotrack" + Clip path│
     │                             │                           ├──────────────────────────>│
     │                             │                           │                           │ Run CoTracker v3
     │                             │                           │                           │ (offline pth)
     │                             │                           │                           │ Extract trajectories
     │                             │                           │ Return Trajectories JSON  │
     │                             │                           │<──────────────────────────┤
     │                             │ Import tracks to Blender  │                           │
     │                             │<──────────────────────────┤                           │
     │                             │                           │                           │
     │                             │ WAITING_FOR_WORKER_SOLVE  │                           │
     │                             ├──────────────────────────>│                           │
     │                             │                           │ Send "precision_solve"    │
     │                             │                           ├──────────────────────────>│
     │                             │                           │                           │ Run SciPy TRF
     │                             │                           │                           │ Bundle Adjustment
     │                             │                           │ Return Solved Poses JSON  │
     │                             │                           │<──────────────────────────┤
     │                             │ Apply camera animation    │                           │
     │                             │<──────────────────────────┤                           │
     │                             │                           │                           │
     │                             │ Stop Worker               │                           │
     │                             ├──────────────────────────>│                           │
     │                             │                           │ Send "shutdown"           │
     │                             │                           ├──────────────────────────>│
     │                             │                           │                           │ Terminate process
     │ Solve Complete              │                           │                           │
     │<────────────────────────────┤                           │                           │
```
