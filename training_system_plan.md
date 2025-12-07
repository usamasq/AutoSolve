# EZTrack Smart Tracking System - Implementation Plan

## Overview

This document outlines a machine learning-based system to continuously improve EZTrack's tracking accuracy by learning from each tracking session.

---

## Phase 1: Data Collection Infrastructure

### 1.1 Track Telemetry System

Every tracking session collects:

| Data Point          | Description                      | Use Case                        |
| ------------------- | -------------------------------- | ------------------------------- |
| **Clip Metadata**   | Resolution, FPS, duration, codec | Categorize footage types        |
| **Track Lifespan**  | Frames each track survived       | Measure tracking stability      |
| **Track Position**  | Screen region (9-zone grid)      | Identify problematic areas      |
| **Track Velocity**  | Movement per frame               | Detect motion patterns          |
| **Jitter Score**    | Variance in velocity             | Measure tracking quality        |
| **Settings Used**   | Pattern/search/correlation       | Correlate settings with success |
| **Success/Failure** | Did track contribute to solve?   | Ground truth labels             |

### 1.2 Storage Architecture

```
%APPDATA%/EZTrack/
├── sessions/
│   ├── 2024-12-07_clip1_session.json
│   └── 2024-12-07_clip2_session.json
├── models/
│   └── settings_predictor.json
└── aggregate_stats.json
```

---

## Phase 2: Feature Engineering

### 2.1 Footage Features (Input)

| Feature             | How to Extract                     | Range   |
| ------------------- | ---------------------------------- | ------- |
| `resolution_class`  | Width bucket (SD/HD/4K)            | 0-2     |
| `fps_class`         | FPS bucket (24/30/60+)             | 0-2     |
| `motion_intensity`  | Avg track velocity first 30 frames | 0.0-1.0 |
| `contrast_estimate` | Feature count at threshold=0.5     | 0-200   |
| `dominant_region`   | Region with most successful tracks | 0-8     |

### 2.2 Settings Features (Output)

| Setting           | Range             | Step |
| ----------------- | ----------------- | ---- |
| `pattern_size`    | 11-31             | 2    |
| `search_size`     | 51-151            | 10   |
| `correlation_min` | 0.4-0.8           | 0.05 |
| `threshold`       | 0.1-0.5           | 0.05 |
| `motion_model`    | Loc/LocRot/Affine | enum |

---

## Phase 3: Learning Algorithms

### 3.1 Rule-Based Learning (Current)

```python
if success_rate < 0.3:
    increase_pattern_size()
    increase_search_size()
    decrease_correlation()
elif success_rate > 0.7:
    # Can be more selective
    increase_threshold()
```

### 3.2 Statistical Learning (Proposed)

For each `(footage_class, region)` combination, track statistics:

```python
class RegionModel:
    def __init__(self):
        self.samples = []

    def add_sample(self, settings, success_rate):
        self.samples.append((settings, success_rate))

    def predict_best_settings(self):
        # Return settings with highest avg success rate
        settings_scores = defaultdict(list)
        for settings, rate in self.samples:
            key = self._settings_to_key(settings)
            settings_scores[key].append(rate)

        best_key = max(settings_scores, key=lambda k: np.mean(settings_scores[k]))
        return self._key_to_settings(best_key)
```

### 3.3 Gradient-Based Optimization (Future)

For more complex patterns, use numerical optimization:

```python
from scipy.optimize import minimize

def objective(settings, history):
    """Predict success rate for given settings."""
    # Use historical data to fit a simple model
    return -predicted_success_rate(settings, history)

result = minimize(objective, initial_settings, method='Nelder-Mead')
optimal_settings = result.x
```

---

## Phase 4: Training Pipeline

### 4.1 Offline Training

```
┌──────────────────┐
│ Session JSONs    │
│ (raw data)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Feature Extract  │
│ (normalize data) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Model Training   │
│ (fit parameters) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Model Export     │
│ (JSON weights)   │
└──────────────────┘
```

### 4.2 Online Learning

During tracking, update model incrementally:

```python
def on_tracking_complete(tracker, analysis):
    # Record what worked
    session_data = {
        'clip_features': extract_clip_features(tracker.clip),
        'settings_used': tracker.current_settings,
        'success_rate': analysis['success_rate'],
        'region_stats': analysis['region_stats'],
    }

    # Save to disk
    save_session(session_data)

    # Update running model
    model.update_online(session_data)
```

---

## Phase 5: Prediction System

### 5.1 Cold Start (No History)

Use heuristic defaults based on footage metadata:

```python
def get_default_settings(clip):
    if clip.size[0] >= 3840:  # 4K
        return {'pattern_size': 25, 'search_size': 100, ...}
    elif clip.fps >= 60:  # High FPS
        return {'pattern_size': 15, 'search_size': 51, ...}
    else:
        return BALANCED_DEFAULTS
```

### 5.2 Warm Start (With History)

Query model for optimal settings:

```python
def get_learned_settings(clip, model):
    features = extract_clip_features(clip)

    # Find similar clips in history
    similar_sessions = model.find_similar(features, k=5)

    # Aggregate their successful settings
    best_settings = model.aggregate_settings(similar_sessions)

    return best_settings
```

---

## Phase 6: Implementation Roadmap

### Week 1: Data Collection

- [ ] Create `SessionRecorder` class
- [ ] Implement `save_session()` and `load_sessions()`
- [ ] Add telemetry hooks to `SmartTracker`

### Week 2: Feature Engineering

- [ ] Create `FeatureExtractor` class
- [ ] Implement footage feature extraction
- [ ] Implement settings normalization

### Week 3: Basic Learning

- [ ] Create `SettingsPredictor` class with rule-based fallback
- [ ] Implement statistical aggregation
- [ ] Add `find_similar()` for clip matching

### Week 4: Integration

- [ ] Hook predictor into tracking pipeline
- [ ] Add A/B testing framework (predicted vs default)
- [ ] Create performance metrics dashboard

### Week 5: Advanced Learning (Optional)

- [ ] Implement gradient-based optimization
- [ ] Add region-specific models
- [ ] Create community data sharing (opt-in)

---

## File Structure

```
autosolve/
├── solver/
│   ├── smart_tracker.py      # Core tracking
│   └── learning/
│       ├── __init__.py
│       ├── session_recorder.py   # Data collection
│       ├── feature_extractor.py  # Feature engineering
│       ├── settings_predictor.py # ML model
│       └── training_utils.py     # Offline training
├── data/
│   ├── sessions/             # Raw session JSONs
│   └── models/               # Trained model weights
└── docs/
    ├── SMART_TRACKING.md     # Technical overview
    └── TRAINING_SYSTEM.md    # This document
```

---

## Success Metrics

| Metric                     | Target  | Measurement                  |
| -------------------------- | ------- | ---------------------------- |
| **First-try success rate** | > 60%   | Solve without retry          |
| **Average iterations**     | < 2     | Retries needed               |
| **Track survival rate**    | > 50%   | Tracks contributing to solve |
| **Average solve error**    | < 1.5px | Reprojection error           |

---

## Data Privacy

- All data stored locally by default
- No automatic upload to servers
- Future: Opt-in anonymous data sharing for community model
