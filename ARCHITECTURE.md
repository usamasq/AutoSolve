# AutoSolve Architecture

> **One-click automated camera tracking using Blender's native tracking system**

> [!NOTE] > **🧪 Research Beta** - This architecture is actively evolving. The learning system improves with community data contributions.

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
└── tracker/             # Core tracking engine
    ├── smart_tracker.py      # Main tracking orchestrator with learning
    ├── analyzers.py          # TrackAnalyzer & CoverageAnalyzer classes
    ├── validation.py         # ValidationMixin - pre-solve validation
    ├── filtering.py          # FilteringMixin - track cleanup methods
    ├── smoothing.py          # Track smoothing utilities
    ├── constants.py          # Shared constants (REGIONS, TIERED_SETTINGS)
    ├── utils.py              # Utility functions (get_region, etc.)
    └── learning/             # Learning components
        ├── session_recorder.py        # Session telemetry collection
        ├── settings_predictor.py      # Optimal settings prediction
        ├── feature_extractor.py       # Visual feature extraction
        ├── behavior_recorder.py       # User behavior recording
        ├── failure_diagnostics.py     # Failure analysis & fixes
        └── pretrained_model.json      # Bundled community defaults
```

---

## Core Components

### 1. Smart Tracker (`tracker/smart_tracker.py`)

**Purpose:** Intelligent tracking orchestrator with adaptive learning.

**Architecture:** Uses mixin pattern for modularity:

- Inherits from `ValidationMixin` (pre-solve validation methods)
- Inherits from `FilteringMixin` (track cleanup methods)
- Uses `TrackAnalyzer` and `CoverageAnalyzer` for analysis

**Key Features:**

- **Phased Detection** - Motion probe → Quality placement → Reinforcement
- **Probe Caching** - Probe results are cached to disk to speed up re-analysis
- **Motion Analysis** - Classifies footage as LOW/MEDIUM/HIGH motion
- **Quality over Quantity** - 1-2 markers per region, not carpet-bombing
- **Adaptive Learning** - Settings adjust based on measured motion characteristics

**Main Class: `SmartTracker(ValidationMixin, FilteringMixin)`**

```python
class SmartTracker(ValidationMixin, FilteringMixin):
    def __init__(self, clip, robust_mode=False, footage_type='AUTO')
    def detect_features_smart()        # Unified smart detection
    def _run_motion_probe()            # Motion analysis (can skip for low motion)
    def _estimate_motion_quick()       # Instant motion classification
    def track_frame()                  # Track one frame
    def track_sequence()               # Batch tracking (optional)
    def solve_camera()                 # Bundle adjustment
    def analyze_and_learn()            # Post-tracking analysis
    # Inherited from ValidationMixin:
    # - validate_pre_tracking(), validate_pre_solve(), etc.
    # Inherited from FilteringMixin:
    # - cleanup_tracks(), filter_short_tracks(), filter_spikes(), etc.
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

### 2. Supporting Modules

#### `tracker/analyzers.py`

**Purpose:** Analysis classes for tracking patterns and coverage.

| Class              | Purpose                                           |
| ------------------ | ------------------------------------------------- |
| `TrackStats`       | Dataclass - statistics for a single track         |
| `RegionStats`      | Dataclass - statistics for a screen region        |
| `CoverageData`     | Dataclass - coverage data for region-time segment |
| `TrackAnalyzer`    | Analyzes tracking patterns, identifies dead zones |
| `CoverageAnalyzer` | Tracks spatial/temporal marker distribution       |

**Key Constants:**

- `REGIONS` - List of 9 screen regions (imported from `constants.py`)
- `MAX_TRACKS_PER_REGION_PERCENT = 0.30` - Clustering threshold

#### `tracker/validation.py`

**Purpose:** ValidationMixin providing pre-solve validation methods.

```python
class ValidationMixin:
    def validate_pre_tracking()       # Check clip loaded, duration
    def validate_track_quality()      # Per-frame quality validation
    def validate_pre_solve()          # Track count, coverage, lifespan
    def compute_pre_solve_confidence() # Estimate solve success probability
    def sanitize_tracks_before_solve() # Remove problematic tracks
```

#### `tracker/filtering.py`

**Purpose:** FilteringMixin providing track cleanup methods.

```python
class FilteringMixin:
    def cleanup_tracks()              # Unified cleanup pipeline
    def filter_short_tracks()         # Remove short-lived tracks
    def filter_spikes()               # Remove velocity outliers
    def deduplicate_tracks()          # Coverage-aware deduplication
    def filter_non_rigid_motion()     # Remove waves/water/foliage tracks
    def filter_high_error()           # Remove high reprojection error
    def _get_region_for_pos()         # Region lookup helper
```

#### `tracker/constants.py`

**Purpose:** Shared configuration constants.

| Constant                   | Description                                     |
| -------------------------- | ----------------------------------------------- |
| `REGIONS`                  | List of 9 screen regions                        |
| `TIERED_SETTINGS`          | Settings tiers (balanced, moderate, aggressive) |
| `PRETRAINED_DEFAULTS`      | Default settings per footage class              |
| `FOOTAGE_TYPE_ADJUSTMENTS` | Footage-specific overrides                      |

#### `tracker/utils.py`

**Purpose:** Utility functions.

```python
def get_region(x, y) -> str          # Get region name for normalized coords
def get_region_bounds(region) -> tuple  # Get bounds for region
def calculate_jitter(markers) -> float  # Compute track jitter score
```

#### `tracker/smoothing.py`

**Purpose:** Track smoothing to reduce jitter.

```python
def smooth_track_markers(tracking, strength)  # Gaussian weighted-average smoothing
```

**Algorithms:**

- **Track Smoothing:** Gaussian weighted moving average. Window size 3-7 based on strength. Preserves endpoints.

---

### 3. Learning System (`tracker/learning/`)

**Purpose:** Adaptive learning that improves tracking over time.

**Modules:**

| File                     | Purpose                                      |
| ------------------------ | -------------------------------------------- |
| `session_recorder.py`    | Collects session & frame telemetry           |
| `settings_predictor.py`  | Predicts optimal settings                    |
| `failure_diagnostics.py` | Diagnoses failures & recommends fixes        |
| `feature_extractor.py`   | Extracts visual features (thumbnails, stats) |
| `behavior_recorder.py`   | Records user behavior patterns               |
| `pretrained_model.json`  | Bundled defaults from community data         |

#### Session Recorder

- Records tracking attempts (settings, success metrics, errors)
- Captures **frame samples**, **trajectory data**, and **camera intrinsics**
- Stores sessions as JSON in `%APPDATA%/AutoSolve/sessions/`
- Anonymizes data (no file paths or identifying info)

#### Settings Predictor

- Classifies footage (e.g., "HD_30fps")
- Loads pretrained model as fallback for cold start
- Applies footage type adjustments (DRONE, INDOOR, etc.)
- Estimates motion from FPS and duration
- Stores user model in `%APPDATA%/AutoSolve/model.json`

#### Feature Extractor

- Extracts visual signatures from footage
- Generates clip fingerprints (MD5)
- Computes motion histograms and edge density proxies
- Captures thumbnails for dataset visualization

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
LOAD_LEARNING → CONFIGURE → DETECT →
TRACK_FORWARD (adaptive) → TRACK_BACKWARD (adaptive) →
ANALYZE → RETRY_DECISION → CLEANUP →
SOLVE_DRAFT → FILTER_ERROR → SOLVE_FINAL → REFINE → COMPLETE
```

**Adaptive Tracking Features:**

- `monitor_and_replenish()` called every 10 frames during tracking
- Adds markers surgically where survival drops below 50%
- Adapts settings if survival drops below 30%
- Backward pass starts from `frame_end` to ensure all markers are covered

**Learning Operators:**

- `AUTOSOLVE_OT_export_training_data` - Export learning data
- `AUTOSOLVE_OT_import_training_data` - Import learning data
- `AUTOSOLVE_OT_reset_training_data` - Reset to defaults
- `AUTOSOLVE_OT_view_training_stats` - View statistics

**Smoothing Operators:**

- `AUTOSOLVE_OT_smooth_tracks` - Manual track marker smoothing

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

    # Smoothing Settings
    smooth_tracks: BoolProperty      # Enable pre-solve track smoothing
    track_smooth_factor: FloatProperty  # Track smoothing strength (0-1)
    track_smooth_factor: FloatProperty  # Track smoothing strength (0-1)

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
│ SmartTracker.detect_features_smart()   │
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
│ • Forward: optimal_start → frame_end    │
│ • Backward: optimal_start → frame_start │
│ • Fill gaps: bidirectional from gaps    │
│ • Verify: extend tracks to full timeline│
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

### 2. Smart Feature Detection

**Purpose:** Ensure balanced spatial coverage for robust solving.

```python
def detect_features_smart(markers_per_region=2):
    # Divide frame into 9 regions (3x3 grid)
    regions = REGIONS  # From constants.py

    for region in regions:
        # Detect in this specific region
        detected = detect_in_region(
            region,
            count=markers_per_region,
            threshold=current_settings['threshold']
        )

    # Result: ~18 quality markers across frame
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

Located in `tracker/constants.py::PRETRAINED_DEFAULTS`:

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

1. Add to `FOOTAGE_TYPE_ADJUSTMENTS` in `tracker/constants.py`
2. Define setting overrides
3. Update UI enum in `properties.py`

### Custom Learning Strategies

Extend `SettingsPredictor` in `tracker/learning/settings_predictor.py`:

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
