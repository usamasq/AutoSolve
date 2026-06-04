# AutoSolve Architecture

> **One-click automated camera tracking using Blender's native tracking system**

---

## Overview

AutoSolve is a Blender extension that automates the manual camera tracking workflow by intelligently orchestrating Blender's built-in tracking operators (`bpy.ops.clip.*`). It uses deterministic smart defaults, quality-aware filtering, and automatic failure recovery to deliver a solid solve in a single click.

**Key Principle:** No external dependencies - 100% native Blender tracking.

---

## Project Structure

```
autosolve/
├── __init__.py          # Package registration
├── operators.py         # Main operators (Auto-Track & Solve, setup scene, etc.)
├── properties.py        # Scene properties and settings
├── ui.py               # N-Panel UI in Movie Clip Editor
├── clip_state.py       # Multi-clip state manager
└── tracker/             # Core tracking engine
    ├── __init__.py           # Tracker package registration
    ├── smart_tracker.py      # Main tracking orchestrator
    ├── analyzers.py          # TrackAnalyzer & CoverageAnalyzer classes
    ├── validation.py         # ValidationMixin - pre-solve validation
    ├── filtering.py          # FilteringMixin - track cleanup & averaging
    ├── averaging.py          # TrackAverager - cluster averaging for noise reduction
    ├── smoothing.py          # Track smoothing utilities
    ├── constants.py          # Shared constants (REGIONS, TIERED_SETTINGS)
    ├── utils.py              # Utility functions (get_region, etc.)
    ├── failure_diagnostics.py # Failure analysis & fixes
    ├── track_healer.py       # Gap healing with anchor interpolation
    ├── feature_density.py    # Temporal texture analysis
    └── presets/
        └── defaults.json     # Bundled community default presets
```

---

## Core Components

### 1. Smart Tracker (`tracker/smart_tracker.py`)

**Purpose:** Intelligent tracking orchestrator.

**Architecture:** Uses mixin pattern for modularity:
- Inherits from `ValidationMixin` (pre-solve validation methods)
- Inherits from `FilteringMixin` (track cleanup methods)
- Uses `TrackAnalyzer` and `CoverageAnalyzer` for analysis

**Key Features:**
- **Phased Detection** - Motion probe → Quality placement → Reinforcement
- **Probe Caching** - Probe results are cached to speed up re-analysis
- **Motion Analysis** - Classifies footage as LOW/MEDIUM/HIGH motion
- **Quality over Quantity** - Balanced markers per region, not carpet-bombing

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
    def filter_motion_spikes()        # Blender's filter_tracks for drift
    def clean_bad_segments()          # DELETE_SEGMENTS for gap creation
    def average_clustered_tracks()    # Average nearby tracks for noise reduction
    def deduplicate_tracks()          # Coverage-aware deduplication
    def filter_non_rigid_motion()     # Remove waves/water/foliage tracks
    def filter_high_error()           # Remove high reprojection error (2.0px)
    def mark_healing_pending()        # Preserve short tracks for healing
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

#### `tracker/averaging.py`
**Purpose:** Track averaging for noise reduction and segment merging.
```python
class TrackAverager:
    def find_track_clusters(tracking)        # Find nearby track clusters
    def average_cluster(tracking, cluster)   # Average cluster using bpy.ops.clip.average_tracks
    def create_anchor_tracks(tracking)       # Main entry: create averaged anchors

def merge_overlapping_segments(tracking)     # Merge tracks with frame overlap
```

**Strategy: Track More → Refine**
```
TRACK 2× MORE → AVERAGE CLUSTERS → HEAL GAPS → CLEAN → HIGH-QUALITY TRACKS
    100 tracks  →       50        →    55     →   40  →  40 refined
```

#### `tracker/failure_diagnostics.py`
**Purpose:** Failure analysis & targeted fixes during retry iterations.
Detects 6 failure patterns:
- Motion blur
- Rapid motion
- Low contrast
- Edge distortion
- Scene cut
- Insufficient features

Returns targeted fix recommendations for retry.

#### `tracker/track_healer.py`
**Purpose:** Gap healing & segment merging via anchor interpolation.

#### `tracker/feature_density.py`
**Purpose:** Visual density & quality analysis.

---

### 3. Operators (`operators.py`)

**Main Operator:** `AUTOSOLVE_OT_run_solve`

**Modal Pipeline:**
```
CONFIGURE → DETECT → TRACK_FORWARD → TRACK_BACKWARD →
HEAL_TRACKS → ANALYZE → RETRY_DECISION → FILTER_SHORT →
SOLVE_DRAFT → FILTER_ERROR → SOLVE_FINAL → REFINE → COMPLETE
```

**HEAL_TRACKS Phase:**
1. `filter_motion_spikes()` - Detect drifted/dislocated markers
2. `clean_bad_segments()` - Remove only bad portions (5.0px threshold)
3. `heal_tracks()` - Anchor-based gap interpolation
4. `merge_overlapping_segments()` - Average overlapping track segments

**Adaptive Tracking Features:**
- `monitor_and_replenish()` called every 10 frames during tracking
- Adds markers surgically where survival drops below 50%
- Adapts settings if survival drops below 30%
- Backward pass starts from `frame_end` to ensure all markers are covered

---

### 4. Properties (`properties.py`)

**Settings Storage:**
```python
class AutoSolveSettings(PropertyGroup):
    # User Settings
    robust_mode: BoolProperty        # Extra iterations for difficult footage
    tripod_mode: BoolProperty        # Rotation-only camera model
    footage_type: EnumProperty       # AUTO, INDOOR, OUTDOOR, DRONE, etc.

    smooth_tracks: BoolProperty      # Enable pre-solve track smoothing
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

### 5. Clip State Management (`clip_state.py`)

**Purpose:** Manages per-clip runtime state for multi-clip workflows.

**Class: `ClipStateManager` (Singleton)**
- **Isolation:** Ensures UI state is unique to each clip
- **Fingerprinting:** Identifies clips by hash (resolution + fps + duration)

---

### 6. UI (`ui.py`)

**Panels:** Phase-based workflow in Movie Clip Editor → Tools (left sidebar) → AutoSolve tab

The UI uses a **guided, phase-based workflow** that progressively reveals options as the user completes each step.

**Phase Logic (`get_workflow_phase`):**

| Condition                           | Phase         |
| ----------------------------------- | ------------- |
| No valid solve OR currently solving | `TRACK`       |
| Valid solve, no tracking camera     | `SCENE_SETUP` |
| Tracking camera with animation      | `REFINE`      |

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
│ 2. Start modal operator                 │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ SmartTracker.analyze_footage()          │
├──────────────────────────────────────────┤
│ • Determine footage class               │
│ • Look up default presets               │
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
│ Results available in:                    │
├──────────────────────────────────────────┤
│ • clip.tracking.reconstruction           │
│ • UI shows error, point count            │
└──────────────────────────────────────────┘
```

---

## Testing & Validation

### Manual Testing
1. Load test footage in Movie Clip Editor
2. Click "Auto-Track & Solve"
3. Verify reconstruction validity
4. Check reprojection error < 2.0px

---

## License

GPL-3.0-or-later

---

**Maintained by:** Usama Bin Shahid  
**Contact:** usamasq@gmail.com
