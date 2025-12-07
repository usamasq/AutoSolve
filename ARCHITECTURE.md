# AutoSolve Architecture

> One-click camera tracking for Blender, inspired by After Effects' 3D Camera Tracker

---

## Overview

AutoSolve automates Blender's built-in motion tracking system to provide a seamless "load footage → click solve → pick ground → place objects" workflow.

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER WORKFLOW                                 │
├─────────────────────────────────────────────────────────────────┤
│  1. Load footage in Movie Clip Editor                           │
│  2. Click "Analyze & Solve" button                              │
│  3. Wait 30 seconds to 2 minutes                                │
│  4. See colored tracking points appear on footage               │
│  5. Click 3 points on the ground plane                          │
│  6. Scene aligns - ready for 3D object placement                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Blender's Built-in Tracker?

| Feature              | External SfM (COLMAP)  | Blender's Tracker       |
| -------------------- | ---------------------- | ----------------------- |
| **Speed**            | 30+ min for 100 frames | 30 sec - 2 min          |
| **Dependencies**     | pycolmap wheel         | None (built-in)         |
| **GPU Acceleration** | Limited                | Full support            |
| **Integration**      | Custom parsing needed  | Native                  |
| **User Refinement**  | Not possible           | Users can adjust tracks |

---

## System Architecture

```
autosolve/
├── __init__.py              # Extension entry point
├── operators.py             # Main operator: "Analyze & Solve"
├── properties.py            # Settings: quality, tripod mode
├── ui.py                    # N-Panel interface
│
├── solver/
│   └── blender_tracker.py   # ⭐ Core: Automates bpy.ops.clip.*
│
├── visualization/
│   └── point_overlay.py     # GPU-drawn points on footage
│
└── orientation/
    ├── ground_picker.py     # Click-to-select ground points
    └── scene_setup.py       # Align scene, create camera
```

---

## Core Component: AutoTracker

The `AutoTracker` class wraps Blender's tracking API into a single call:

### Pipeline Steps

```
┌────────────────────┐
│ 1. DETECT FEATURES │ → bpy.ops.clip.detect_features()
├────────────────────┤    Places 50-100 markers on first frame
│ • Threshold-based  │    using corner detection algorithm
│ • Min distance     │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ 2. TRACK FORWARD   │ → bpy.ops.clip.track_markers(backwards=False)
├────────────────────┤    Follows each marker frame-by-frame
│ • KLT optical flow │    using correlation matching
│ • Automatic        │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ 3. TRACK BACKWARD  │ → bpy.ops.clip.track_markers(backwards=True)
├────────────────────┤    Fills gaps where forward tracking
│ • Fills gaps       │    couldn't reach
│ • Improves quality │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ 4. CLEAN TRACKS    │ → bpy.ops.clip.clean_tracks()
├────────────────────┤    Removes tracks that:
│ • Short duration   │    - Were tracked < 10 frames
│ • High error       │    - Have high reprojection error
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ 5. SOLVE CAMERA    │ → bpy.ops.clip.solve_camera()
├────────────────────┤    Bundle adjustment to find:
│ • Camera motion    │    - Camera position/rotation per frame
│ • 3D point coords  │    - 3D coordinates of tracked points
└────────────────────┘
```

### Quality Presets

| Preset       | Detection Threshold | Min Distance | Correlation | Use Case       |
| ------------ | ------------------- | ------------ | ----------- | -------------- |
| **Fast**     | 0.3                 | 120px        | 0.75        | Quick preview  |
| **Balanced** | 0.5                 | 80px         | 0.85        | Most footage   |
| **Quality**  | 0.7                 | 50px         | 0.90        | Complex scenes |

---

## Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ MovieClip   │────▶│ AutoTracker  │────▶│ Reconstruction  │
│ (Input)     │     │ (Processing) │     │ (Output)        │
└─────────────┘     └──────────────┘     └─────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ clip.tracking│
                    │ .tracks      │  ← Tracked markers (2D)
                    │ .reconstruction│ ← Solved camera (3D)
                    └──────────────┘
```

### Key Blender Data Structures

```python
# Accessing tracking data
clip = bpy.context.edit_movieclip
tracking = clip.tracking

# Tracked markers
for track in tracking.tracks:
    track.name              # Marker name
    track.markers           # Per-frame positions
    track.has_bundle        # True if 3D position calculated
    track.bundle            # 3D position (x, y, z)
    track.average_error     # Reprojection error

# Reconstruction result
recon = tracking.reconstruction
recon.is_valid              # True if solve succeeded
recon.average_error         # Overall solve error
recon.cameras               # Per-frame camera data
```

---

## Point Overlay System

Draws solved track points on the footage in the Movie Clip Editor:

```
┌─────────────────────────────────────────────────────────────┐
│                  Movie Clip Editor                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                                                      │    │
│  │    🟢 ← Good track (error < 0.3px)                  │    │
│  │         🟡 ← Medium track (0.3 - 0.7px)             │    │
│  │              🔴 ← Poor track (> 0.7px)              │    │
│  │                                                      │    │
│  │  [Video frame with colored tracking points]         │    │
│  │                                                      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

Uses GPU-accelerated drawing via `bpy.types.SpaceClipEditor.draw_handler_add()`:

```python
# Register draw callback
handler = bpy.types.SpaceClipEditor.draw_handler_add(
    draw_function,
    (),
    'WINDOW',
    'POST_PIXEL',  # Draw after the image
)

# Draw points using GPU module
gpu.state.point_size_set(8.0)
shader.bind()
batch.draw(shader)
```

---

## Ground Picker System

Allows users to visually select ground points:

### User Interaction

```
1. User clicks "Pick Ground" button
2. Modal operator starts
3. User clicks on 3+ track points
4. Points highlight as selected
5. User presses Enter
6. Scene transforms to align ground to Z=0
```

### Plane Fitting Algorithm

```python
# Given 3+ points, fit a plane using SVD
import numpy as np

def fit_plane(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # SVD to find normal
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # Last row = plane normal

    return normal, centroid
```

### Scene Alignment

```python
# Align scene so plane becomes Z=0
def align_to_ground(normal, centroid):
    # Rotation to align normal with Z-up
    z_up = Vector((0, 0, 1))
    rotation = normal.rotation_difference(z_up)

    # Apply to all tracked objects
    for obj in tracked_objects:
        obj.rotation_euler.rotate(rotation)
        obj.location -= centroid
```

---

## Operator Flow

```
┌─────────────────────────────────────────────────────────────┐
│  AUTOSOLVE_OT_run_solve (operators.py)                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  execute(context):                                          │
│    │                                                         │
│    ├── Get clip from context.edit_movieclip                │
│    │                                                         │
│    ├── Create AutoTracker(clip)                             │
│    │                                                         │
│    ├── tracker.run(quality, tripod_mode, callback)         │
│    │     │                                                   │
│    │     ├── _detect_features()                             │
│    │     ├── _track_sequence(forwards)                      │
│    │     ├── _track_sequence(backwards)                     │
│    │     ├── _clean_tracks()                                │
│    │     └── _solve_camera()                                │
│    │                                                         │
│    ├── if result.success:                                   │
│    │     ├── Enable point overlay                           │
│    │     └── Update UI (solve_error, point_count)          │
│    │                                                         │
│    └── return {'FINISHED'}                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Settings & Properties

```python
# properties.py
class AutoSolveSettings(PropertyGroup):
    # User-configurable
    quality_preset: EnumProperty(
        items=[('FAST', ...), ('BALANCED', ...), ('QUALITY', ...)],
        default='BALANCED',
    )
    tripod_mode: BoolProperty(default=False)

    # Runtime state
    is_solving: BoolProperty()
    solve_progress: FloatProperty()
    solve_status: StringProperty()

    # Results
    has_solve: BoolProperty()
    solve_error: FloatProperty()
    point_count: IntProperty()
```

---

## UI Layout

```
┌─────────────────────────────────────┐
│ AutoSolve                    [N-Panel]
├─────────────────────────────────────┤
│ ┌─────────────────────────────────┐ │
│ │ 🎬 footage.mp4 | 240 frames    │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │     [  Analyze & Solve  ]      │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ▼ Options (collapsed)              │
│   Quality: [Balanced ▼]            │
│   Tripod Mode: [ ]                 │
│                                     │
│ ▼ Result (after solve)             │
│   Points: 87                       │
│   Error: 0.42 px                   │
│                                     │
│   [Pick Ground]  [Set Scale]       │
│   [Create Scene Objects]           │
└─────────────────────────────────────┘
```

---

## Error Handling

| Error          | Cause                  | User Message                                 |
| -------------- | ---------------------- | -------------------------------------------- |
| No clip loaded | User didn't load video | "No Movie Clip loaded"                       |
| Too few tracks | Detection failed       | "Only X tracks found. Need 8+"               |
| Solve failed   | Not enough parallax    | "Camera solve failed. Try different footage" |
| High error     | Poor tracking          | "Solve error too high (X px)"                |

---

## Future Enhancements

1. **Keyframe Selection** - Detect best frames for detection automatically
2. **Mask Support** - Exclude regions (sky, moving objects)
3. **Multiple Solves** - Compare different settings
4. **Object Tracking** - Track specific objects in addition to camera

---

## Summary

AutoSolve transforms Blender's powerful but manual motion tracking system into a one-click experience by:

1. **Automating** all tracking operations via `bpy.ops.clip.*`
2. **Visualizing** results with GPU-drawn point overlays
3. **Simplifying** ground alignment with click-to-select
4. **Integrating** deeply with Blender's native scene setup
