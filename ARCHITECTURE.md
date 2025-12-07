# AutoSolve Architecture

> **One-click automated camera tracking using Blender's native tracking system**

---

## Overview

AutoSolve is a Blender extension that automates the manual camera tracking workflow by intelligently orchestrating Blender's built-in tracking operators (`bpy.ops.clip.*`). It uses adaptive learning to improve tracking quality over time.

**Key Principle:** No external dependencies - 100% native Blender tracking.

---

## Project Structure

```
autosolve/
├── __init__.py          # Package registration
├── operators.py         # Main operators (Analyze & Solve, training tools)
├── properties.py        # Scene properties and settings
├── ui.py               # N-Panel UI in Movie Clip Editor
└── solver/
    ├── smart_tracker.py      # Adaptive tracking engine
    ├── blender_tracker.py    # Low-level Blender API wrapper
    └── learning/
        ├── __init__.py
        ├── session_recorder.py    # Session telemetry collection
        ├── settings_predictor.py  # Optimal settings prediction
        ├── failure_diagnostics.py # Failure analysis & fixes
        └── pretrained_model.json  # Bundled community defaults
```

---

## Core Components

### 1. Smart Tracker (`solver/smart_tracker.py`)

**Purpose:** Intelligent tracking orchestrator with adaptive learning.

**Key Features:**

- **Phased Detection** - Motion probe → Quality placement → Reinforcement
- **Motion Analysis** - Classifies footage as LOW/MEDIUM/HIGH motion
- **Quality over Quantity** - 1-2 markers per region, not carpet-bombing
- **Adaptive Learning** - Settings adjust based on measured motion characteristics

**Main Class: `SmartTracker`**

```python
class SmartTracker:
    def __init__(self, clip, robust_mode=False, footage_type='AUTO')
    def detect_features_smart()        # Unified smart detection
    def _run_motion_probe()            # Motion analysis (can skip for low motion)
    def _estimate_motion_quick()       # Instant motion classification
    def track_frame()                  # Track one frame
    def track_sequence()               # Batch tracking (optional)
    def cleanup_tracks()               # Unified filtering (spikes, non-rigid)
    def solve_camera()                 # Bundle adjustment
```

**Detection Flow:**

```
Phase 1: MOTION PROBE
├── Place 1 marker in 5 regions
├── Track 20 frames
└── Classify: LOW / MEDIUM / HIGH motion

Phase 2: QUALITY DETECTION
├── Select settings based on motion class
└── Place 1-2 quality markers per region

Phase 3: REINFORCEMENT (if <8 markers)
└── Add markers to reliable center regions
```

---

### 2. Blender Tracker (`solver/blender_tracker.py`)

**Purpose:** Low-level wrapper for Blender's tracking operators.

**Responsibilities:**

- Safe operator execution with context overrides
- Marker selection and filtering
- Error handling for tracking operations

**Key Methods:**

```python
def detect_features(threshold, distance)
def track_markers(backwards, sequence)
def clean_tracks(frames, error)
def filter_tracks_by_error(max_error)
def solve_camera(tripod_mode)
```

---

### 3. Learning System (`solver/learning/`)

**Purpose:** Adaptive learning that improves tracking over time.

**Modules:**

| File                     | Purpose                               |
| ------------------------ | ------------------------------------- |
| `session_recorder.py`    | Collects session telemetry            |
| `settings_predictor.py`  | Predicts optimal settings             |
| `failure_diagnostics.py` | Diagnoses failures & recommends fixes |
| `pretrained_model.json`  | Bundled defaults from community data  |

#### Session Recorder

- Records tracking attempts (settings, success metrics, errors)
- Stores sessions as JSON in `%APPDATA%/AutoSolve/sessions/`
- Anonymizes data (no file paths or identifying info)

#### Settings Predictor

- Classifies footage (e.g., "HD_30fps")
- Loads pretrained model as fallback for cold start
- Applies footage type adjustments (DRONE, INDOOR, etc.)
- Estimates motion from FPS and duration
- Stores user model in `%APPDATA%/AutoSolve/model.json`

#### Failure Diagnostics

- Detects 6 failure patterns:
  - Motion blur, Rapid motion, Low contrast
  - Edge distortion, Scene cut, Insufficient features
- Returns targeted fix recommendations for retry

**Learning Loop:**

```
Clip → Predict Settings → Track → IF FAILS: Diagnose → Fix → Retry
     → Record Session → Update Model
```

---

### 4. Operators (`operators.py`)

**Main Operator: `AUTOSOLVE_OT_run_solve`**

**Modal Pipeline:**

```
LOAD_LEARNING → CONFIGURE → DETECT → TRACK_FORWARD →
TRACK_BACKWARD → ANALYZE → ANALYZE_COVERAGE → FILL_GAPS →
CLEANUP → SOLVE_DRAFT → FILTER_ERROR → SOLVE_FINAL →
REFINE → COMPLETE
```

**Note:** CLEANUP phase consolidates FILTER_SHORT, FILTER_SPIKES, and FILTER_NON_RIGID into a single pass.

**Learning Operators:**

- `AUTOSOLVE_OT_export_training_data` - Export learning data
- `AUTOSOLVE_OT_import_training_data` - Import learning data
- `AUTOSOLVE_OT_reset_training_data` - Reset to defaults
- `AUTOSOLVE_OT_view_training_stats` - View statistics

**Scene Setup:**

- `AUTOSOLVE_OT_setup_scene` - Calls `bpy.ops.clip.setup_tracking_scene`

---

### 5. Properties (`properties.py`)

**Settings Storage:**

```python
class AutoSolveSettings(PropertyGroup):
    # User Settings
    robust_mode: BoolProperty        # Extra iterations for difficult footage
    tripod_mode: BoolProperty        # Rotation-only camera model
    footage_type: EnumProperty       # AUTO, INDOOR, OUTDOOR, DRONE, etc.

    # Runtime State
    is_solving: BoolProperty         # Currently solving
    solve_progress: FloatProperty    # 0.0 to 1.0
    solve_status: StringProperty     # Status message

    # Results
    has_solve: BoolProperty          # Solve succeeded
    solve_error: FloatProperty       # Reprojection error (px)
    point_count: IntProperty         # Number of 3D points
```

---

### 6. UI (`ui.py`)

**Panel: `CLIP_PT_autosolve`**

Location: Movie Clip Editor → Sidebar (N) → AutoSolve tab

**Layout:**

```
┌─────────────────────────────────────┐
│ AutoSolve                           │
├─────────────────────────────────────┤
│ Clip Info: filename.mp4             │
│ Duration: 240 frames                │
│                                     │
│ [    Analyze & Solve    ]          │
│                                     │
│ Options:                            │
│   Footage Type: [Auto ▼]           │
│   Tripod Mode: [ ]                 │
│   Robust Mode: [ ]                 │
│                                     │
│ Results:                            │
│   Error: 0.42 px                   │
│   Points: 87                        │
│                                     │
│ [Setup Scene] [Export Data]        │
└─────────────────────────────────────┘
```

---

## Data Flow

### Complete Tracking Flow

```
┌──────────────┐
│ User clicks  │
│ "Analyze &   │
│ Solve"       │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ AUTOSOLVE_OT_run_solve.execute()        │
├──────────────────────────────────────────┤
│ 1. Create SmartTracker(clip)            │
│ 2. Load local learning data             │
│ 3. Start modal operator                 │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ SmartTracker.analyze_footage()          │
├──────────────────────────────────────────┤
│ • Determine footage class               │
│ • Look up learned/pretrained settings   │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ SmartTracker.detect_strategic_features()│
├──────────────────────────────────────────┤
│ • Divide frame into 9 regions           │
│ • Detect ~3 markers per region          │
│ • Ensure balanced spatial coverage      │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ Track frame-by-frame (modal)            │
├──────────────────────────────────────────┤
│ • Forward: frame_start → frame_end      │
│ • Backward: frame_end → frame_start     │
│ • Live quality validation               │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ SmartTracker.solve_camera()             │
├──────────────────────────────────────────┤
│ • Bundle adjustment                      │
│ • Compute 3D positions                   │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ SmartTracker.analyze_and_learn()        │
├──────────────────────────────────────────┤
│ • Extract training data                  │
│ • SessionRecorder.save()                │
│ • SettingsPredictor.update_model()      │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ Results available in:                    │
├──────────────────────────────────────────┤
│ • clip.tracking.reconstruction           │
│ • UI shows error, point count            │
└──────────────────────────────────────────┘
```

---

## Key Algorithms

### 1. Footage Classification

**Purpose:** Determine optimal settings based on footage characteristics.

```python
def classify_footage(clip):
    width, height = clip.size
    fps = clip.fps

    # Resolution class
    if width >= 3840:
        res_class = "4K"
    elif width >= 1920:
        res_class = "HD"
    else:
        res_class = "SD"

    # FPS class
    fps_class = f"{int(fps)}fps"

    return f"{res_class}_{fps_class}"  # e.g., "HD_30fps"
```

### 2. Strategic Feature Detection

**Purpose:** Ensure balanced spatial coverage for robust solving.

```python
def detect_strategic_features(markers_per_region=3):
    # Divide frame into 9 regions (3x3 grid)
    regions = create_grid(3, 3)

    for region in regions:
        # Detect in this specific region
        detected = detect_in_region(
            region,
            count=markers_per_region,
            threshold=current_settings['threshold']
        )

    # Result: ~27 evenly distributed markers
```

### 3. Adaptive Learning

**Purpose:** Improve settings over time based on success rate.

```python
def update_model(footage_class, session_data):
    # Load existing data for this footage class
    existing = model.get(footage_class, {})

    # Update running averages
    if session_data['success']:
        existing['success_count'] += 1
        existing['avg_settings'] = weighted_average(
            existing['avg_settings'],
            session_data['settings'],
            weight=0.1  # Learning rate
        )

    model[footage_class] = existing
```

---

## Configuration

### Pretrained Defaults

Located in `smart_tracker.py::PRETRAINED_DEFAULTS`:

```python
PRETRAINED_DEFAULTS = {
    'HD_24fps': {
        'pattern_size': 17,
        'search_size': 91,
        'correlation': 0.68,
        'threshold': 0.28,
        'motion_model': 'LocRot',
    },
    'HD_30fps': { ... },
    '4K_60fps': { ... },
}
```

### Local Learning Storage

**Location:** `%APPDATA%/AutoSolve/` (Windows)

**Files:**

- `local_model.json` - Learned optimal settings per footage class
- `sessions/*.json` - Individual tracking session records

---

## Error Handling

### Validation Layers

1. **Pre-tracking:** Check clip loaded, duration > 10 frames
2. **Live tracking:** Monitor active track count every 10 frames
3. **Pre-solve:** Validate track count, lifespan, spatial coverage
4. **Post-solve:** Check reconstruction validity, reprojection error

### Retry Strategy

```python
MAX_ITERATIONS = 2

if should_retry(analysis):
    # Adjust settings
    settings['threshold'] *= 0.9  # More lenient
    settings['pattern_size'] += 2  # Larger template

    # Clear and retry
    clear_tracks()
    detect_features()
    track_sequence()
```

---

## Performance Considerations

### Tracking Speed

- **HD 30fps, 100 frames:** ~30-60 seconds
- **4K 60fps, 100 frames:** ~60-120 seconds

### Optimization Strategies

1. **Frame step:** Skip frames for preview (not implemented)
2. **Region-based:** Only detect in important areas
3. **Adaptive density:** Fewer markers in low-texture regions
4. **Early stop:** Stop if quality drops too low

---

## Testing & Validation

### Manual Testing

1. Load test footage in Movie Clip Editor
2. Click "Analyze & Solve"
3. Verify reconstruction validity
4. Check reprojection error < 2.0px

### Test Cases

- Various resolutions (SD, HD, 4K)
- Various frame rates (24, 30, 60fps)
- Different footage types (indoor, outdoor, drone)
- Difficult cases (motion blur, low texture)

---

## Extension Points

### Adding New Footage Types

1. Add to `FOOTAGE_TYPE_ADJUSTMENTS` in `smart_tracker.py`
2. Define setting overrides
3. Update UI enum in `properties.py`

### Custom Learning Strategies

Extend `SettingsPredictor` in `solver/learning/settings_predictor.py`:

- Add new prediction algorithms
- Implement failure pattern learning
- Export/import community models
- Add new compatibility methods for external integrations

---

## FAQ for Contributors

**Q: Where is the main tracking logic?**  
A: `operators.py::AUTOSOLVE_OT_run_solve.modal()` - Modal operator pipeline

**Q: How do I add a new setting?**  
A: Add to `AutoSolveSettings` in `properties.py`, use in `SmartTracker`

**Q: Where is the learning data stored?**  
A: `%APPDATA%/AutoSolve/` on Windows, equivalent on other platforms

**Q: Why no pycolmap/external solver?**  
A: Blender's native tracker is faster and tightly integrated. External solvers were removed for simplicity.

**Q: How do I debug tracking failures?**  
A: Enable Blender's console, check printed status messages, inspect `session_recorder` output

---

## License

GPL-3.0-or-later

---

**Maintained by:** Usama Bin Shahid  
**Contact:** usamasq@gmail.com
