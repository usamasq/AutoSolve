# Training Data Guide

> **How AutoSolve learns and improves from your tracking sessions**

> [!IMPORTANT] > **🧪 Research Beta** - Your data contributions directly improve AutoSolve for everyone.
> By sharing anonymized tracking data, you help train the model that predicts optimal settings.
> **[How to contribute](#contributing-your-data)**

---

## Overview

AutoSolve uses an **adaptive learning system** that collects anonymous telemetry from each tracking session and uses it to predict optimal settings for future footage. The more you use it, the smarter it gets.

---

## Current vs. Future AI

It is important to understand exactly how AutoSolve learns, as it distinguishes between what happens **now** and what is being built for the **future**.

### 1. Current System: Statistical Learning (Implemented)

The "AI" currently running on your machine is a **statistical feedback loop**, not a deep neural network. It uses a technique called **Hindsight Experience Replay (HER)**.

- **How it learns:** It records the outcome of every session. If a specific settings configuration (e.g., `Search Size: 101px`) consistently produces success for a specific footage type (e.g., `4K_24fps`), it reinforces that configuration.
- **What it predicts:** It calculates weighted averages of the most successful settings from your history.
- ** Privacy:** All learning happens locally on your machine.

### 2. Future System: Neural Network (Research Phase)

We are currently collecting rich telemetry data (optical flow patterns, trajectory shapes, error gradients) to train a **Deep Neural Network**.

- **Why collect this data?** While the current statistical model is effective, a neural network can learn complex, non-linear relationships—like "shaky handheld footage with motion blur requires specific settings that differ from smooth drone shots."

- **The Goal:** To replace the current statistical heuristics with a trained model that can "see" the motion in your footage and understand it like a human tracker would.

---

## Practical Applications (What can we build?)

The data schema allows for three tiers of AI advancement based on data volume:

### ✅ Tier 1: Statistical Heuristics (< 500 sessions)

**What you have now.**

- **Lookup Tables:** "If 4K_24fps, try pattern=19".
- **Rule-based Logic:** "If velocity > 0.03, assume drone/high-motion".
- **Region Scoring:** "Center tracks survive 80% longer than edges".

### 🟡 Tier 2: Shallow Machine Learning (500-5k sessions)

**Practical Next Step.**

- **Settings Predictor (XGBoost/LightGBM):** Input clip metadata and motion probe results → Output optimal 5 tracking parameters.
- **Dead Zone Classifier (Random Forest):** Input region texture/density → Output probability of tracking failure.
- **Track Quality Filter (Logistic Regression):** Filter "good" vs "bad" tracks based on jitter, velocity, and correlation confidence.

### 🔴 Tier 3: Deep Learning (10k+ sessions)

**Long-term Goal.**

- **LSTM/Transformer:** Analyze full 100-frame trajectory sequences to predict exact frame of failure.
- **End-to-End Control:** Neural network that actively drives the tracker, adjusting parameters per-frame.
- **Visual Saliency (CNN):** Input raw pixels → Output exact feature placement map (requires image data export).

---

## Data Storage Location

| Platform    | Path                                                                   |
| ----------- | ---------------------------------------------------------------------- |
| **Windows** | `%APPDATA%/Blender Foundation/Blender/[version]/datafiles/AutoSolve/`  |
| **macOS**   | `~/Library/Application Support/Blender/[version]/datafiles/AutoSolve/` |
| **Linux**   | `~/.config/blender/[version]/datafiles/AutoSolve/`                     |

**Files:**

```
AutoSolve/
├── model.json           # Learned patterns & HER experiences
├── sessions/
│   └── *.json           # Individual session records
└── behavior/
    └── *.json           # User behavior records (new)
```

**Bundled with addon:**

```
autosolve/tracker/learning/
└── pretrained_model.json  # Community defaults (fallback)
```

---

## Data Schema

### Session Record (`sessions/*.json`)

Each tracking session generates a JSON file with this structure:

```json
{
  "schema_version": 1,
  "timestamp": "2025-12-07T19:30:00",
  "clip_name": "footage_001",
  "iteration": 1,
  "duration_seconds": 45.2,

  "resolution": [1920, 1080],
  "fps": 30,
  "frame_count": 240,

  "settings": {
    "pattern_size": 15,
    "search_size": 71,
    "correlation": 0.7,
    "threshold": 0.3,
    "motion_model": "LocRot"
  },

  "success": true,
  "solve_error": 0.42,
  "total_tracks": 45,
  "successful_tracks": 38,
  "bundle_count": 35,

  "camera_intrinsics": {
    "focal_length_mm": 35.0,
    "focal_length_px": 2100.0,
    "sensor_width_mm": 36.0,
    "pixel_aspect": 1.0,
    "principal_point": [0.5, 0.5],
    "distortion_model": "POLYNOMIAL",
    "k1": -0.05,
    "k2": 0.02,
    "k3": 0.0
  },

  "clip_fingerprint": "a4f3c89b2e71d6f0",
  "motion_class": "MEDIUM",

  "global_motion_vector": [0.012, -0.003],
  "motion_consistency": 0.85,
  "failure_type": "NONE",
  "frame_of_failure": null,

  "flow_direction_histogram": [0.12, 0.08, 0.05, 0.15, 0.25, 0.18, 0.1, 0.07],
  "flow_magnitude_histogram": [0.1, 0.3, 0.35, 0.2, 0.05],

  "tracks": [
    {
      "name": "Track.001",
      "lifespan": 180,
      "start_frame": 1,
      "end_frame": 180,
      "region": "center",
      "avg_velocity": 0.0023,
      "jitter_score": 0.12,
      "success": true,
      "contributed_to_solve": true,
      "reprojection_error": 0.38,
      "trajectory": [
        [0.52, 0.48],
        [0.53, 0.47],
        [0.54, 0.46]
      ],
      "trajectory_sample_rate": 5
    }
  ],

  "track_failures": [
    {
      "track_name": "Track.015",
      "frame": 87,
      "position": [0.12, 0.85],
      "reason": "DRIFT"
    }
  ],

  "region_stats": {
    "center": { "total_tracks": 8, "successful_tracks": 7 },
    "top-left": { "total_tracks": 4, "successful_tracks": 2 }
  },

  "dead_zones": ["top-right"],
  "sweet_spots": ["center", "bottom-center"],

  "motion_probe_results": {
    "motion_class": "MEDIUM",
    "texture_class": "HIGH",
    "best_regions": ["center", "mid-left"]
  },

  "optical_flow": {
    "velocity_mean": 0.0234,
    "velocity_std": 0.0089,
    "velocity_max": 0.067,
    "parallax_score": 0.32,
    "dominant_direction": [0.98, -0.12],
    "direction_entropy": 0.15,
    "velocity_acceleration": 0.0012,
    "track_dropout_rate": 0.18
  },

  "visual_features": {
    "clip_fingerprint": "a4f3c89b2e71d6f0",
    "motion_class": "MEDIUM",
    "motion_magnitude": 0.015,
    "motion_variance": 0.008,
    "edge_density": {
      "center": 0.85,
      "top-left": 0.42,
      "top-right": 0.38
    },
    "edge_density_mean": 0.55,
    "contrast_stats": {
      "center": {
        "estimated_contrast": 0.85,
        "track_count": 8,
        "success_rate": 0.875
      },
      "top-left": {
        "estimated_contrast": 0.42,
        "track_count": 4,
        "success_rate": 0.5
      }
    },
    "temporal_motion_profile": [0.012, 0.015, 0.018, 0.02, 0.019]
  },

  // Track Healing (NEW v5): Anchor-based gap interpolation
  "anchor_tracks": [
    { "name": "Track.003", "start_frame": 1, "end_frame": 180, "quality": 0.92 }
  ],
  "healing_attempts": [
    {
      "gap_start_frame": 50,
      "gap_end_frame": 65,
      "gap_start_pos": [0.52, 0.48],
      "gap_end_pos": [0.54, 0.46],
      "anchor_count": 4,
      "interpolated_positions": [
        [0.52, 0.47],
        [0.53, 0.47],
        [0.53, 0.46]
      ],
      "method_used": "anchor_weighted",
      "post_heal_error": 0.42,
      "heal_success": true
    }
  ],
  "healing_stats": {
    "candidates_found": 5,
    "heals_attempted": 3,
    "heals_successful": 2,
    "avg_gap_frames": 15.0,
    "avg_match_score": 0.82
  }
}
```

---

### Behavior Record (`behavior/*.json`)

User behavior is recorded separately for ML training. **This is THE KEY data** for learning how experts improve tracking.

```json
{
  "schema_version": 1,
  "session_id": "20251209_093000",
  "timestamp": "2025-12-09T09:35:00",

  // Session linkage (for multi-attempt analysis)
  "clip_fingerprint": "a7f3c89b2e71d6f0",
  "previous_session_id": "20251209_091500",
  "iteration": 3,
  "contributor_id": "x7f2k9a1",

  "editing_duration_seconds": 45.0,

  "settings_adjustments": {
    "search_size": {
      "before": 71,
      "after": 100,
      "delta": 29
    },
    "correlation": {
      "before": 0.7,
      "after": 0.6,
      "delta": -0.1
    }
  },

  "re_solve": {
    "attempted": true,
    "error_before": 1.8,
    "error_after": 0.9,
    "improvement": 0.9,
    "improved": true
  },

  // THE KEY: Track additions (what pros ADD to improve tracking)
  "track_additions": [
    {
      "track_name": "Track.042",
      "region": "center",
      "initial_frame": 45,
      "position": [0.52, 0.48],
      "lifespan_achieved": 145,
      "had_bundle": true,
      "reprojection_error": 0.32
    }
  ],

  "track_deletions": [
    {
      "track_name": "Track.005",
      "region": "top-left",
      "lifespan": 23,
      "had_bundle": true,
      "reprojection_error": 3.2,
      "inferred_reason": "high_error"
    }
  ],

  "marker_refinements": [
    {
      "track_name": "Track.008",
      "frame": 45,
      "old_position": [0.52, 0.48],
      "new_position": [0.524, 0.477],
      "displacement_px": 2.3
    }
  ],

  // Quality metrics
  "net_track_change": 2,
  "region_additions": { "center": 2, "bottom-left": 1 }
}
```

---

## Track Telemetry Fields

| Field                    | Type        | Description                             |
| ------------------------ | ----------- | --------------------------------------- |
| `name`                   | string      | Marker identifier                       |
| `lifespan`               | int         | Number of frames tracked                |
| `initial_position`       | [x,y]       | Position at first frame (0-1)           |
| `feature_quality_score`  | float       | 0-1 score (edges=low, center=high)      |
| `start_frame`            | int         | First frame of track                    |
| `end_frame`              | int         | Last frame of track                     |
| `region`                 | string      | Frame region (9 zones)                  |
| `avg_velocity`           | float       | Average movement per frame (normalized) |
| `jitter_score`           | float       | Movement variance (0=smooth, 1=erratic) |
| `success`                | bool        | Track lifespan ≥ 5 frames               |
| `contributed_to_solve`   | bool        | Track has 3D bundle                     |
| `reprojection_error`     | float       | Pixel error after solve                 |
| `trajectory`             | [[x,y],...] | Sampled positions for ML (RNN input)    |
| `trajectory_sample_rate` | int         | Frames between samples (default: 5)     |

---

## Camera Intrinsics Fields

| Field              | Type   | Description                               |
| ------------------ | ------ | ----------------------------------------- |
| `focal_length_mm`  | float  | Focal length in millimeters               |
| `focal_length_px`  | float  | Focal length in pixels                    |
| `sensor_width_mm`  | float  | Sensor width in millimeters               |
| `pixel_aspect`     | float  | Pixel aspect ratio                        |
| `principal_point`  | [x, y] | Optical center (normalized 0-1)           |
| `distortion_model` | string | POLYNOMIAL, DIVISION, NUKE, or BROWN      |
| `k1`, `k2`, `k3`   | float  | Polynomial radial distortion coefficients |
| `division_k1/k2`   | float  | Division model coefficients               |
| `nuke_k1/k2`       | float  | Nuke-compatible coefficients              |
| `brown_k1-k4`      | float  | Brown-Conrady radial coefficients         |
| `brown_p1/p2`      | float  | Brown-Conrady tangential coefficients     |

---

## Global Motion & Failure Fields

| Field                  | Type     | Description                                      |
| ---------------------- | -------- | ------------------------------------------------ |
| `global_motion_vector` | [dx, dy] | Average motion across all tracks (normalized)    |
| `motion_consistency`   | float    | 0-1 score (1=all tracks move consistently)       |
| `failure_type`         | string   | NONE, BLUR, CONTRAST, CUT, DRIFT, INSUFFICIENT   |
| `frame_of_failure`     | int/null | Frame where track count first dropped critically |

---

## Optical Flow Fields (ML-Ready)

Continuous metrics for neural network training:

| Field                   | Type     | Description                                     |
| ----------------------- | -------- | ----------------------------------------------- |
| `velocity_mean`         | float    | Average track velocity (normalized to diagonal) |
| `velocity_std`          | float    | Velocity standard deviation                     |
| `velocity_max`          | float    | Maximum velocity observed                       |
| `parallax_score`        | float    | 0.0=uniform, 1.0=strong depth variation         |
| `dominant_direction`    | [dx, dy] | Unit vector of camera motion                    |
| `direction_entropy`     | float    | 0.0=all same direction, 1.0=random              |
| `velocity_acceleration` | float    | Change in velocity over clip                    |
| `track_dropout_rate`    | float    | Fraction of tracks that fail early              |

## Motion Probe & Adaptation Fields (NEW)

| Field                  | Type   | Description                                      |
| ---------------------- | ------ | ------------------------------------------------ |
| `motion_probe_results` | object | Cached motion analysis from 20-frame probe       |
| `motion_class`         | string | LOW, MEDIUM, or HIGH based on velocity analysis  |
| `texture_class`        | string | LOW, MEDIUM, or HIGH based on feature count      |
| `best_regions`         | [str]  | Regions with >50% survival during probe          |
| `adaptation_history`   | [obj]  | List of mid-session settings adaptations         |
| `region_confidence`    | Dict   | Probabilistic confidence scores per region (0-1) |
| `frame_samples`        | [obj]  | Per-frame statistics for temporal ML (v1+)       |

## Zoom Analysis Fields (NEW)

Zoom/dolly detection computed from track trajectory data:

| Field                | Type    | Description                                         |
| -------------------- | ------- | --------------------------------------------------- |
| `is_zoom_detected`   | bool    | True if >5% scale change detected                   |
| `zoom_direction`     | string  | ZOOM_IN, ZOOM_OUT, or NONE                          |
| `scale_timeline`     | [float] | Temporal scale at each trajectory sample [1.0, ...] |
| `estimated_fl_ratio` | float   | Final/initial scale ratio (>1=out, <1=in)           |
| `scale_variance`     | float   | Low=zoom (uniform), high=dolly (parallax)           |
| `is_uniform_scale`   | bool    | True if zoom-like, False if dolly-like              |
| `radial_convergence` | float   | -1=converging (in), +1=diverging (out)              |
| `confidence`         | float   | Detection confidence (0-1)                          |

**Interpretation:**

| Scale Timeline     | Variance | Interpretation   |
| ------------------ | -------- | ---------------- |
| `[1.0, 1.1, 1.2]`↑ | Low      | ZOOM OUT (lens)  |
| `[1.0, 0.9, 0.8]`↓ | Low      | ZOOM IN (lens)   |
| `[1.0, 1.1, 1.2]`↑ | High     | DOLLY OUT (move) |
| `[1.0, 0.9, 0.8]`↓ | High     | DOLLY IN (move)  |

### Frame Samples (v1 Schema)

Per-frame telemetry for RNN/LSTM training:

```json
"frame_samples": [
  {"frame": 1, "active_tracks": 27, "tracks_lost": 0, "avg_velocity": 0.012},
  {"frame": 10, "active_tracks": 25, "tracks_lost": 2, "avg_velocity": 0.015},
  {"frame": 20, "active_tracks": 23, "tracks_lost": 0, "avg_velocity": 0.018}
]
```

| Field           | Type  | Description                           |
| --------------- | ----- | ------------------------------------- |
| `frame`         | int   | Frame number                          |
| `active_tracks` | int   | Number of active tracks at this frame |
| `tracks_lost`   | int   | Tracks lost since previous sample     |
| `avg_velocity`  | float | Average velocity across active tracks |

### Adaptation Record Structure

Each entry in `adaptation_history` contains:

- `iteration`: Which adaptation pass (1, 2, 3)
- `survival_rate`: Track survival rate that triggered adaptation
- `changes`: Human-readable descriptions of setting changes
- `old_settings` / `new_settings`: Before/after settings

### Frame Regions

```
┌───────────┬───────────┬───────────┐
│ top-left  │ top-center│ top-right │
├───────────┼───────────┼───────────┤
│ mid-left  │  center   │ mid-right │
├───────────┼───────────┼───────────┤
│bottom-left│bottom-ctr │bottom-right
└───────────┴───────────┴───────────┘
```

---

## Learning Model (`model.json`)

The aggregated model tracks performance using HER (Hindsight Experience Replay):

```json
{
  "version": 1,
  "global_stats": {
    "total_sessions": 47,
    "successful_sessions": 42
  },
  "footage_classes": {
    "HD_30fps": {
      "sample_count": 23,
      "success_count": 21,
      "avg_success_rate": 0.91,
      "experiences": [
        {
          "settings": {"pattern_size": 17, "search_size": 91},
          "outcome": "SUCCESS",
          "reward": 0.85,
          "solve_error": 0.7,
          "failure_type": null
        }
      ],
      "best_settings": {...}
    }
  },
  "region_models": {
    "center": {
      "total_tracks": 234,
      "successful_tracks": 198,
      "avg_lifespan": 145
    }
  },
  "behavior_patterns": {
    "HD_30fps:search_size_increase": {
      "count": 5,
      "confidence": 0.8,
      "avg_delta": 29
    }
  }
}
```

### Footage Classification

Clips are classified by resolution and frame rate:

| Class      | Resolution | FPS   |
| ---------- | ---------- | ----- |
| `SD_24fps` | <1920      | <28   |
| `SD_30fps` | <1920      | 28-49 |
| `HD_24fps` | 1920-3839  | <28   |
| `HD_30fps` | 1920-3839  | 28-49 |
| `HD_60fps` | 1920-3839  | 50+   |
| `4K_24fps` | 3840+      | <28   |
| `4K_30fps` | 3840+      | 28-49 |
| `4K_60fps` | 3840+      | 50+   |

### Motion Classification

The system runs a 20-frame motion probe to classify footage:

| Motion Class | Velocity Threshold | Settings Applied                     |
| ------------ | ------------------ | ------------------------------------ |
| **HIGH**     | >0.03              | Pattern: 25px, Search: 141px, Affine |
| **MEDIUM**   | 0.01-0.03          | Pattern: 19px, Search: 101px, LocRot |
| **LOW**      | <0.01              | Pattern: 15px, Search: 71px, Loc     |

**How the probe works:**

1. Place 1 marker in 5 random regions
2. Track forward for 20 frames
3. Measure velocity per track: `displacement / frames`
4. Average across all tracks → classify motion

### Per-Clip Learning

The system now supports exact clip recognition via fingerprinting:

| Feature               | Description                                           |
| --------------------- | ----------------------------------------------------- |
| **Clip Fingerprint**  | MD5 hash of filepath + resolution + fps + duration    |
| **Per-clip Settings** | Best settings stored per fingerprint                  |
| **Priority Order**    | 1) Same clip → 2) Motion sub-class → 3) Footage class |

When you re-track the same clip, AutoSolve uses the exact settings that worked before.

### Motion Sub-Classification

Footage classes are now subdivided by motion level:

- `HD_30fps_LOW_MOTION` - Tripod/static shots
- `HD_30fps_MEDIUM_MOTION` - Standard camera movement
- `HD_30fps_HIGH_MOTION` - Action/fast movement

This provides more specific settings than base footage class alone.

### Visual Features for NN Training

When "Learn from My Edits" is enabled, these features are extracted:

| Feature              | Description                                       | ML Purpose                 |
| -------------------- | ------------------------------------------------- | -------------------------- |
| **Feature Density**  | Count of trackable features per region            | Dead zone prediction       |
| **Density Timeline** | Density at 25%, 50%, 75% of clip                  | Temporal change/occlusion  |
| **Edge Density**     | Track success rate per region (proxy for texture) | Region quality prediction  |
| **Contrast Stats**   | Success/track counts per region                   | Dead zone prediction       |
| **Flow Histograms**  | 8-bin direction + 5-bin magnitude                 | Motion pattern recognition |
| **Temporal Profile** | Velocity at evenly-spaced frames                  | RNN/LSTM training          |

**Feature Density Extraction:**

- Runs `detect_features` with low threshold (0.4)
- Samples 3 consecutive frames at each timeline point (25%, 50%, 75%) for robustness
- Counts potential markers in each 3x3 region
- Saves counts without storing images (privacy safe)
- Used to identify textureless or sky regions before tracking

---

## How Learning Works

### 1. Data Collection

```
User clicks "Auto-Track & Solve"
         │
         ▼
SessionRecorder.start_session()
         │
    [Tracking runs]
         │
         ▼
SessionRecorder.record_tracks()
         │
         ▼
SessionRecorder.end_session()
         │
         ▼
SettingsPredictor.update_model()
```

### 2. Model Update Algorithm

```python
def update_model(session_data):
    footage_class = classify(resolution, fps)

    if success:
        # Record successful settings
        settings_history.append({
            settings, solve_error, success_rate
        })

        # Keep best 20 sessions
        settings_history = settings_history[-20:]

        # Best = lowest error
        best_settings = min(history, key=solve_error)

    # Update success rates
    avg_success_rate = success_count / sample_count
```

### 3. Prediction for New Footage

```python
def predict_settings(clip):
    footage_class = classify(clip)

    if footage_class in model and sample_count >= 3:
        # Use learned settings
        return best_settings
    else:
        # Fall back to heuristics
        return heuristic_settings(resolution, fps)
```

---

## Settings Parameters

| Parameter      | Range             | Purpose                       |
| -------------- | ----------------- | ----------------------------- |
| `pattern_size` | 5-31 (odd)        | Template matching window      |
| `search_size`  | 25-151 (odd)      | Motion search area            |
| `correlation`  | 0.5-0.95          | Match confidence threshold    |
| `threshold`    | 0.1-0.5           | Feature detection sensitivity |
| `motion_model` | Loc/LocRot/Affine | Motion complexity             |

### Tiered Presets

| Tier         | When Used           | Characteristics               |
| ------------ | ------------------- | ----------------------------- |
| `aggressive` | Success rate <30%   | Large search, low correlation |
| `moderate`   | Success rate 30-50% | Balanced settings             |
| `balanced`   | Success rate 50-70% | Default settings              |
| `selective`  | Success rate >70%   | Strict filtering              |

---

## Using Training Data

### Export Data

```
Clip Editor → AutoSolve → Training Data → Export
```

Creates a ZIP archive containing:

```
autosolve_training_YYYYMMDD.zip
├── manifest.json      # Export metadata
├── model.json         # Learned patterns
├── sessions/          # Session records
│   └── *.json
└── behavior/          # User behavior records
    └── *.json
```

### Import Data

```
Clip Editor → AutoSolve → Training Data → Import
```

Options:

- **Merge**: Combine with existing data
- **Replace**: Overwrite existing data

### Reset Data

```
Clip Editor → AutoSolve → Training Data → Reset
```

Clears all learned data, returns to pretrained defaults.

---

## Behavior Learning

AutoSolve learns from your corrections between solves. This is **enabled by default** via "Learn from My Edits" in the Training Data panel.

### How It Works

```
1. Run AutoSolve → Solve completes (error = 1.2)
2. User adjusts settings (search_size 71 → 100)
3. Run AutoSolve again → Solve completes (error = 0.8)
4. System learns: "search_size increase helped for this footage type"
```

### What is Learned

| Data                        | Value                                                     |
| --------------------------- | --------------------------------------------------------- |
| **Settings changes**        | Which adjustments improved error for which footage types  |
| **Track deletions**         | Regions where users consistently delete tracks            |
| **Improvement correlation** | Only builds confidence when changes actually reduce error |

### Safety Guarantees

AutoSolve will **never degrade** performance:

- **3+ observations required** — No adjustment applied until seen multiple times
- **0.7+ confidence required** — Changes must have improved error in most cases
- **Only proven adjustments** — Changes that made error worse are noted but never applied

### Console Feedback

```
AutoSolve: Monitoring for user edits and behavior changes
AutoSolve: Captured user behavior - will learn after new solve
AutoSolve: Learned from user behavior (error improved: 1.20→0.80)
AutoSolve: Applied learned behavior adjustments: search_size: 71.0→95.0
```

---

## Contributing Training Data

### Data Privacy

All session data is **anonymized**:

- ❌ No file paths
- ❌ No user identifiers
- ❌ No frame content
- ✅ Only numerical metrics

### Sharing Your Data

**[Read the Full Contribution Guide](CONTRIBUTING_DATA.md)**

To help improve AutoSolve for everyone:

1. Export your training data via the UI
2. Upload to: **[HuggingFace dataset](https://huggingface.co/datasets/UsamaSQ/autosolve-telemetry)**
3. Discuss on: **[Discord community](https://discord.gg/qUvrXHP9PU)**

### Community Model

Future versions may include a **community model** aggregated from user submissions to provide better defaults for everyone.

---

## Advanced: Custom Training

### Analyzing Your Data

```python
import json
from pathlib import Path

# Load all sessions
sessions_dir = Path("~/.config/blender/.../autosolve/sessions")
sessions = []
for f in sessions_dir.glob("*.json"):
    with open(f) as fp:
        sessions.append(json.load(fp))

# Analyze success rates by footage class
from collections import defaultdict
by_class = defaultdict(list)
for s in sessions:
    width = s['resolution'][0]
    fps = s['fps']
    cls = f"{'HD' if width >= 1920 else 'SD'}_{int(fps)}fps"
    by_class[cls].append(s['success'])

for cls, results in by_class.items():
    rate = sum(results) / len(results)
    print(f"{cls}: {rate:.1%} success ({len(results)} sessions)")
```

### Identifying Problem Regions

```python
# Find regions that consistently fail
region_stats = defaultdict(lambda: {'total': 0, 'success': 0})

for session in sessions:
    for track in session['tracks']:
        region = track['region']
        region_stats[region]['total'] += 1
        if track['contributed_to_solve']:
            region_stats[region]['success'] += 1

print("Region Performance:")
for region, stats in region_stats.items():
    rate = stats['success'] / max(stats['total'], 1)
    status = "🔴" if rate < 0.5 else ("🟡" if rate < 0.7 else "🟢")
    print(f"  {status} {region}: {rate:.1%}")
```

---

## Failure Diagnostics

When tracking fails, the system analyzes the session data to diagnose the cause:

### Failure Patterns

| Pattern                   | Symptoms                      | Automatic Fix                 |
| ------------------------- | ----------------------------- | ----------------------------- |
| **Motion Blur**           | High jitter + short lifespan  | Larger pattern, larger search |
| **Rapid Motion**          | High velocity tracks          | Double search area            |
| **Low Contrast**          | Few features detected         | Lower threshold               |
| **Edge Distortion**       | Edge regions fail             | Weight center detection       |
| **Scene Cut**             | Mass track death at one frame | Redetect after cut            |
| **Insufficient Features** | <8 bundles                    | More sensitive detection      |

### Retry Flow

```
Track Failed
    │
    ▼
FailureDiagnostics.diagnose(analysis)
    │
    ▼
Returns: DiagnosisResult {
    pattern: MOTION_BLUR,
    confidence: 0.8,
    fix_adjustments: {...}
}
    │
    ▼
apply_fix(settings, diagnosis)
    │
    ▼
Retry with adjusted settings
```

---

## Future Improvements

### Planned Features

1. **Real-time Motion Estimation**

   - Analyze actual optical flow before tracking
   - Better search size prediction

2. **Neural Network Model**

   - Replace heuristics with ML
   - Better generalization

3. **Community Model Sync**
   - Opt-in anonymized sharing
   - Download improved defaults

---

## FAQ

**Q: How much data is collected?**

- Only numerical tracking metrics, no image data

**Q: Is my data sent anywhere?**

- No, all data stays on your machine unless you explicitly export

**Q: How many sessions before learning kicks in?**

- 3 sessions per footage class for predictions to activate

**Q: Can I train on someone else's data?**

- Yes, use Import to merge their exported data

---

**Maintained by:** Usama Bin Shahid  
**Contact:** usamasq@gmail.com
