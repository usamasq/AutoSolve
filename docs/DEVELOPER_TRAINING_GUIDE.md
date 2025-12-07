# EZTrack Developer Training Guide

## Overview

This guide explains how to train and tune the pre-shipped tracking defaults for EZTrack. The goal is to provide users with optimal out-of-the-box settings based on your extensive testing.

---

## Training Workflow

### Phase 1: Data Collection

#### 1.1 Gather Diverse Footage

Collect footage from various sources:

| Category        | Examples                                                     |
| --------------- | ------------------------------------------------------------ |
| **Resolution**  | 720p, 1080p, 2K, 4K                                          |
| **Frame Rate**  | 24fps, 25fps, 30fps, 60fps, 120fps                           |
| **Camera Type** | Drone, Handheld, Tripod, Gimbal, Phone                       |
| **Motion Type** | Slow pan, Fast action, Zoom, Tracking shot                   |
| **Environment** | Indoor, Outdoor, Mixed lighting                              |
| **Difficulty**  | Easy (sharp), Medium (slight motion blur), Hard (heavy blur) |

#### 1.2 Organize Test Library

```
training_footage/
├── HD_24fps/
│   ├── easy/
│   │   ├── tripod_indoor_01.mp4
│   │   └── tripod_outdoor_01.mp4
│   ├── medium/
│   │   └── handheld_indoor_01.mp4
│   └── hard/
│       └── action_blur_01.mp4
├── HD_30fps/
│   └── ...
├── 4K_24fps/
│   └── ...
└── 4K_60fps/
    └── ...
```

### Phase 2: Systematic Testing

#### 2.1 Create Testing Script

```python
"""
training_session.py - Automated training data collection
Run from Blender's Python console after loading a clip.
"""

import bpy
import json
from pathlib import Path

# Test matrix
SETTINGS_MATRIX = [
    {'pattern_size': 11, 'search_size': 51, 'correlation': 0.75},
    {'pattern_size': 15, 'search_size': 71, 'correlation': 0.70},
    {'pattern_size': 17, 'search_size': 91, 'correlation': 0.68},
    {'pattern_size': 21, 'search_size': 100, 'correlation': 0.65},
    {'pattern_size': 25, 'search_size': 120, 'correlation': 0.60},
    {'pattern_size': 31, 'search_size': 150, 'correlation': 0.50},
]

def run_test(clip, settings):
    """Run tracking with given settings and return results."""
    from autosolve.solver.smart_tracker import SmartTracker

    tracker = SmartTracker(clip, robust_mode=False)
    tracker.current_settings = settings.copy()
    tracker.configure_settings()
    tracker.clear_tracks()

    # Detect and track
    tracker.detect_features(threshold=0.25)

    for frame in range(clip.frame_start, clip.frame_start + min(100, clip.frame_duration)):
        bpy.context.scene.frame_set(frame)
        tracker.track_frame(backwards=False)

    # Analyze
    analysis = tracker.analyze_and_learn()

    # Try solve
    success = tracker.solve_camera()
    error = tracker.get_solve_error() if success else 999.0

    return {
        'settings': settings,
        'success_rate': analysis['success_rate'],
        'solve_success': success,
        'solve_error': error,
        'track_count': len(tracker.tracking.tracks),
    }

def find_optimal_settings(clip):
    """Test all settings and find optimal."""
    results = []

    for settings in SETTINGS_MATRIX:
        print(f"Testing: {settings}")
        result = run_test(clip, settings)
        results.append(result)
        print(f"  → Success rate: {result['success_rate']:.1%}, Solve: {result['solve_success']}, Error: {result['solve_error']:.2f}")

    # Find best (highest success rate with successful solve)
    successful = [r for r in results if r['solve_success']]
    if successful:
        best = min(successful, key=lambda r: r['solve_error'])
    else:
        best = max(results, key=lambda r: r['success_rate'])

    return best

# Run
clip = bpy.context.edit_movieclip
if clip:
    optimal = find_optimal_settings(clip)
    print(f"\n✓ OPTIMAL SETTINGS: {optimal['settings']}")
    print(f"  Solve error: {optimal['solve_error']:.2f}px")
```

#### 2.2 Record Results

For each footage class, record results in a spreadsheet:

| Clip          | Class    | Pattern | Search | Corr | Success Rate | Solve Error | Notes     |
| ------------- | -------- | ------- | ------ | ---- | ------------ | ----------- | --------- |
| drone_01.mp4  | 4K_30fps | 21      | 100    | 0.65 | 72%          | 0.84        | Good      |
| indoor_01.mp4 | HD_24fps | 15      | 71     | 0.70 | 68%          | 1.12        | OK        |
| action_01.mp4 | HD_60fps | 13      | 51     | 0.72 | 81%          | 0.67        | Excellent |

### Phase 3: Aggregate Results

#### 3.1 Analyze Data

After testing 5+ clips per class, calculate averages:

```python
"""
analyze_results.py - Aggregate training results
"""

import json
from collections import defaultdict

# Load all session data
sessions = []  # Load from export files

# Group by footage class
by_class = defaultdict(list)
for session in sessions:
    footage_class = session.get('footage_class', 'unknown')
    by_class[footage_class].append(session)

# For each class, find best settings
recommendations = {}
for footage_class, class_sessions in by_class.items():
    # Filter to successful sessions
    successful = [s for s in class_sessions if s.get('success')]

    if not successful:
        continue

    # Find settings with lowest average error
    settings_performance = defaultdict(list)
    for session in successful:
        settings_key = json.dumps(session.get('settings', {}), sort_keys=True)
        settings_performance[settings_key].append(session.get('solve_error', 999))

    best_key = min(settings_performance, key=lambda k: sum(settings_performance[k])/len(settings_performance[k]))
    best_settings = json.loads(best_key)
    avg_error = sum(settings_performance[best_key]) / len(settings_performance[best_key])

    recommendations[footage_class] = {
        'settings': best_settings,
        'avg_error': avg_error,
        'sample_count': len(settings_performance[best_key]),
    }

    print(f"{footage_class}: {best_settings} → {avg_error:.2f}px (n={len(settings_performance[best_key])})")
```

### Phase 4: Update Defaults

#### 4.1 Edit PRETRAINED_DEFAULTS

Open `autosolve/solver/smart_tracker.py` and update:

```python
PRETRAINED_DEFAULTS = {
    # Format: 'CLASS': { settings }

    # Your findings from testing:
    'HD_24fps': {
        'pattern_size': 17,      # Based on 8 clips, avg error 1.12px
        'search_size': 91,
        'correlation': 0.68,
        'threshold': 0.28,
        'motion_model': 'LocRot',
    },
    'HD_30fps': {
        'pattern_size': 15,      # Based on 12 clips, avg error 0.94px
        'search_size': 71,
        'correlation': 0.70,
        'threshold': 0.30,
        'motion_model': 'LocRot',
    },
    # ... add all classes you tested
}
```

#### 4.2 Add Comments

Document your findings:

```python
# PRETRAINED_DEFAULTS - Developer-tuned tracking settings
#
# Last updated: 2024-12-07
# Training dataset: 47 clips across 8 footage classes
# Average solve error on training set: 1.02px
# Average first-try success rate: 68%
#
# Methodology:
# - Each class tested with 5+ clips
# - Settings optimized for lowest solve error
# - Validated on held-out test clips
```

---

## Validation Checklist

Before shipping new defaults:

- [ ] Test each class with 3+ clips NOT in training set
- [ ] Verify first-try success rate > 60%
- [ ] Verify average solve error < 2.0px
- [ ] Test on both Windows and macOS
- [ ] Test on Blender 4.2, 4.3, and 5.0

---

## Community Model Integration

### Collecting Anonymous Data

If implementing opt-in anonymous data sharing:

```python
def submit_anonymous_session(session_data):
    """Submit session for community model training."""
    # Remove identifying info
    anonymized = {
        'footage_class': session_data['footage_class'],
        'settings': session_data['settings'],
        'success': session_data['success'],
        'solve_error': session_data.get('solve_error'),
        'success_rate': session_data.get('success_rate'),
        'region_stats': session_data.get('region_stats'),
        # Don't include: clip_name, file_path, timestamps
    }

    # Submit to your server
    # requests.post('https://api.eztrack.io/v1/sessions', json=anonymized)
```

### Processing Community Data

```python
def train_community_model(sessions):
    """Train model from aggregated community data."""
    # Same as Phase 3 but with much larger dataset
    # Update PRETRAINED_DEFAULTS with new findings
    # Ship with next addon version
```

---

## Version History

Track changes to defaults:

| Version | Date       | Changes                                   |
| ------- | ---------- | ----------------------------------------- |
| v1.0.0  | 2024-12-01 | Initial defaults (developer testing only) |
| v1.1.0  | 2024-12-07 | Updated HD defaults, added 4K classes     |
| v1.2.0  | TBD        | Community-trained model integration       |

---

## Tips for Better Training

1. **Test Edge Cases**: Motion blur, low light, uniform textures
2. **Test Different Cameras**: DSLR, mirrorless, phone, drone
3. **Include Difficult Footage**: If it fails, note why and adjust
4. **Document Everything**: Comments in code help future developers
5. **Version Control**: Tag releases with training data versions
