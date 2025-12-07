# EZTrack Smart Tracking System - Technical Documentation

## Overview

EZTrack's Smart Tracking System automates Blender's native motion tracker with professional-grade algorithms that learn and adapt. This document describes the complete technical architecture.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐│
│  │ Quality     │  │ Tripod Mode  │  │ Robust Mode             ││
│  │ Preset      │  │ Checkbox     │  │ (Difficult footage)     ││
│  └─────────────┘  └──────────────┘  └─────────────────────────┘│
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MODAL OPERATOR                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ State Machine: LOAD_LEARNING → CONFIGURE → DETECT →       │ │
│  │   TRACK_FWD → TRACK_BWD → ANALYZE → [RETRY] → FILTER →    │ │
│  │   SOLVE_DRAFT → FILTER_ERROR → SOLVE_FINAL                │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SMART TRACKER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ TrackAnalzer│  │ Settings    │  │ Learning Data           │ │
│  │ (learning)  │  │ Predictor   │  │ (JSON persistence)      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BLENDER TRACKING API                          │
│  bpy.ops.clip.detect_features | track_markers | solve_camera    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. SmartTracker Class

**Location:** `autosolve/solver/smart_tracker.py`

The main orchestrator that manages the entire tracking pipeline.

```python
class SmartTracker:
    # Safeguards - never go below these
    ABSOLUTE_MIN_TRACKS = 12
    SAFE_MIN_TRACKS = 20
    MAX_ITERATIONS = 3

    def __init__(self, clip, robust_mode=False):
        self.clip = clip
        self.analyzer = TrackAnalyzer()
        self.current_settings = {}
```

#### Key Methods

| Method                 | Purpose                                       |
| ---------------------- | --------------------------------------------- |
| `analyze_footage()`    | Determine optimal settings from clip metadata |
| `configure_settings()` | Apply settings to Blender's tracker           |
| `detect_features()`    | Find trackable points in current frame        |
| `track_frame()`        | Track all markers forward/backward one frame  |
| `analyze_and_learn()`  | Analyze results and learn from them           |
| `should_retry()`       | Decide if retry with new settings needed      |
| `prepare_retry()`      | Clear and reconfigure for retry               |

---

### 2. TrackAnalyzer Class

**Location:** `autosolve/solver/smart_tracker.py`

Analyzes tracking patterns to learn what works.

#### Region Grid

The screen is divided into 9 regions for spatial analysis:

```
┌────────────┬────────────┬────────────┐
│  top-left  │ top-center │ top-right  │
├────────────┼────────────┼────────────┤
│  mid-left  │   center   │ mid-right  │
├────────────┼────────────┼────────────┤
│bottom-left │bottom-centr│bottom-right│
└────────────┴────────────┴────────────┘
```

#### Data Collected

```python
@dataclass
class TrackStats:
    name: str           # Track identifier
    lifespan: int       # Frames tracked
    start_frame: int    # When track started
    end_frame: int      # When track ended/failed
    region: str         # Screen region
    avg_velocity: float # Movement per frame
    jitter_score: float # Tracking stability
    success: bool       # Contributed to solve?
```

#### Dead Zones & Sweet Spots

After analysis, regions are classified:

| Classification | Condition          | Action               |
| -------------- | ------------------ | -------------------- |
| **Dead Zone**  | Success rate < 30% | Avoid detection here |
| **Sweet Spot** | Success rate > 70% | Prioritize detection |
| **Neutral**    | 30-70%             | Normal detection     |

---

### 3. Adaptive Settings System

Settings automatically adjust based on success rate:

```python
def get_recommended_settings(success_rate):
    if success_rate < 0.3:  # Very low - be aggressive
        return {
            'pattern_size': 27,   # Large pattern
            'search_size': 130,   # Large search area
            'correlation': 0.50,  # Very forgiving
            'threshold': 0.10,    # Detect many features
        }
    elif success_rate < 0.6:  # Moderate - balanced
        return {
            'pattern_size': 21,
            'search_size': 100,
            'correlation': 0.65,
            'threshold': 0.25,
        }
    else:  # Good - can be selective
        return {
            'pattern_size': 15,
            'search_size': 71,
            'correlation': 0.75,
            'threshold': 0.35,
        }
```

---

## Tracking Pipeline Phases

### Phase 1: Load Learning

```python
# Try to load previous learning data for this clip
tracker.load_learning(clip.name)
```

### Phase 2: Configure

```python
# Analyze footage metadata (resolution, FPS)
analysis = tracker.analyze_footage()

# Apply settings to Blender's tracker
tracker.configure_settings(analysis)

# Clear any existing tracks
tracker.clear_tracks()
```

### Phase 3: Detect Features

```python
# Detect trackable features
num = tracker.detect_features(threshold=0.25)

# Apply per-track settings
for track in tracks:
    track.pattern_size = 15
    track.search_size = 71
    track.correlation_min = 0.70
```

### Phase 4: Track Forward (Relay Race)

```python
for frame in range(start, end):
    tracker.track_frame(backwards=False)

    # Every 30 frames, check density
    if frame % 30 == 0:
        active = tracker.count_active_tracks(frame)
        if active < 25:
            # Replenish with new features
            tracker.detect_features()
```

### Phase 5: Track Backward (Gap Filling)

```python
# Track backwards to fill gaps
for frame in range(end, start, -1):
    tracker.track_frame(backwards=True)
```

### Phase 6: Analyze & Learn

```python
# Analyze what worked
analysis = tracker.analyze_and_learn()

print(f"Success rate: {analysis['success_rate']*100}%")
print(f"Dead zones: {analysis['dead_zones']}")
print(f"Sweet spots: {analysis['sweet_spots']}")

# Decide if retry needed
if tracker.should_retry(analysis):
    tracker.prepare_retry()
    goto Phase 3
```

### Phase 7: Filter & Solve

```python
# Remove short-lived tracks
tracker.filter_short_tracks(min_frames=5)

# Remove velocity outliers
tracker.filter_spikes(limit_multiplier=8.0)

# Draft solve to get error values
tracker.solve_camera()

# Remove high-error tracks
tracker.filter_high_error(max_error=3.0)

# Final solve
tracker.solve_camera()
```

### Phase 8: Save Learning

```python
# Save what we learned for future clips
tracker.save_learning(clip.name)
```

---

## Safeguard System

The system never deletes too many tracks:

```python
def filter_with_safeguards(min_frames):
    current = len(tracks)

    # Count survivors
    survivors = sum(1 for t in tracks if len(t.markers) >= min_frames)

    # Never go below SAFE_MIN_TRACKS
    if survivors < SAFE_MIN_TRACKS:
        print(f"Skipping filter - would leave only {survivors}")
        return

    # Never go below ABSOLUTE_MIN_TRACKS
    max_delete = current - ABSOLUTE_MIN_TRACKS
```

---

## Settings Reference

### Pattern Size

The size of the feature pattern to match:

| Value | Use Case                          |
| ----- | --------------------------------- |
| 11-15 | Sharp, high-contrast footage      |
| 17-21 | Normal footage                    |
| 23-27 | Blurry, low-contrast, motion blur |

### Search Size

How far to search for the pattern in next frame:

| Value   | Use Case              |
| ------- | --------------------- |
| 51-71   | Slow motion, high FPS |
| 81-100  | Normal motion         |
| 110-150 | Fast motion, low FPS  |

### Correlation Threshold

Minimum match quality to accept:

| Value     | Effect                                 |
| --------- | -------------------------------------- |
| 0.75-0.85 | Strict - fewer but accurate tracks     |
| 0.60-0.75 | Balanced                               |
| 0.45-0.60 | Lenient - more tracks, some inaccurate |

### Motion Model

How the pattern is expected to transform:

| Model    | Description            | Use Case          |
| -------- | ---------------------- | ----------------- |
| `Loc`    | Translation only       | Simple movement   |
| `LocRot` | Translation + rotation | Handheld camera   |
| `Affine` | Full 2D transform      | Perspective, zoom |

---

## Learning Data Format

Data is stored as JSON for persistence:

```json
{
  "iteration": 2,
  "dead_zones": ["top-left", "top-right"],
  "sweet_spots": ["center", "bottom-center"],
  "optimal_settings": {
    "pattern_size": 21,
    "search_size": 100,
    "correlation": 0.65
  },
  "region_stats": {
    "center": {
      "name": "center",
      "total_tracks": 45,
      "successful_tracks": 38,
      "avg_lifespan": 87.3,
      "success_rate": 0.844
    }
  }
}
```

---

## Performance Considerations

### Modal Operator Timer

```python
# 0.02s interval = 50 FPS UI updates
self._timer = wm.event_timer_add(0.02, window=context.window)
```

### Context Override

All `bpy.ops.clip` calls need proper context:

```python
def _run_ops(self, op_func, **kwargs):
    override = self._get_context_override()
    with bpy.context.temp_override(**override):
        op_func(**kwargs)
```

---

## Blender API Compatibility

The code handles Blender version differences:

```python
# Settings have version-specific names
if hasattr(settings, 'default_pattern_size'):
    settings.default_pattern_size = value

# clean_tracks action enum changed
try:
    bpy.ops.clip.clean_tracks(action='DELETE_TRACK')
except TypeError:
    bpy.ops.clip.clean_tracks(action='DELETE')
```

---

## Troubleshooting

### "Only X features detected"

- Lower threshold in settings
- Check if footage has enough texture
- Try Robust Mode

### "Solve failed"

- Enable Tripod Mode for nodal pan/tilt
- Check for moving objects in frame
- Try more feature points

### Tracks die immediately

- Increase search_size
- Lower correlation threshold
- Enable Robust Mode
