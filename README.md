# AutoSolve - Automatic Camera Tracking for Blender

> **A personal project by Usama Bin Shahid**  
> _Dedicated to my students, with ❤️ from Pakistan 🇵🇰_

> [!IMPORTANT] > **🧪 Research Beta** - This addon is in active development and features a **learning system** that improves tracking quality over time.
>
> **How it works:**
>
> 1.  **Learns Locally:** It learns from your tracking sessions to improve its own settings on your machine.
> 2.  **Community Driven:** You can **optionally share your data** to help train the community model.
>
> **[Contribute your data](#contribute-training-data)** to help build the best open-source tracking algorithm!

AutoSolve is a Blender addon that **automates the entire camera tracking workflow** - from feature detection to camera solve. It uses **adaptive learning** to improve tracking quality over time by learning from each session.

## What It Does

| Step                     | Manual Workflow                               | AutoSolve                                                  |
| ------------------------ | --------------------------------------------- | ---------------------------------------------------------- |
| **1. Feature Detection** | Place markers manually or use Detect Features | ✅ Smart detection with balanced region coverage           |
| **2. Tracking**          | Track forward/backward, fix lost markers      | ✅ Bidirectional tracking with automatic replenishment     |
| **3. Track Cleanup**     | Delete short/bad tracks manually              | ✅ Automatic filtering of jittery, short, and spike tracks |
| **4. Camera Solve**      | Run solver, hope for low error                | ✅ Iterative refinement with failure diagnosis             |
| **5. Learning**          | Remember what worked                          | ✅ Learns settings that work for your footage types        |

> **Note:** AutoSolve uses Blender's native tracking - no external dependencies required.

---

## Features

| Feature                    | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| **One-Click Tracking**     | Automatic feature detection, tracking, cleanup, and solve |
| **Adaptive Learning**      | Learns optimal settings from your footage over time       |
| **Smart Detection**        | Balanced marker placement across all screen regions       |
| **Bidirectional Tracking** | Starts from mid-clip for better frame coverage            |
| **Quality Prediction**     | Estimates solve quality before running solver             |
| **Failure Diagnosis**      | Detects why tracking failed and applies targeted fixes    |
| **Footage Type Presets**   | Optimized settings for DRONE, INDOOR, HANDHELD, etc.      |

---

## Requirements

- **Blender 4.2.0** or later
- No external dependencies (uses Blender's native tracking only)

---

## Installation

1. Download from the **Blender Extensions Platform**
2. In Blender: `Edit → Preferences → Add-ons`
3. Click **"Install from Disk"** and select the file
4. Enable the extension

---

## Usage

1. Open the **VFX Workspace** in Blender
2. Load footage in the **Movie Clip Editor**
3. Open the **AutoSolve** panel in the sidebar (N)
4. Select a **Footage Type** (or leave on AUTO)
5. Click **Analyze & Solve**
6. Wait for tracking to complete (progress shown in panel)

### Options

| Option           | Purpose                                                                      |
| ---------------- | ---------------------------------------------------------------------------- |
| **Quality**      | Controls speed vs accuracy tradeoff                                          |
|                  | **Fast** - Fewer markers (20), faster tracking, lenient thresholds           |
|                  | **Balanced** - Default settings (35 markers), good for most footage          |
|                  | **Quality** - More markers (50), stricter thresholds, best accuracy          |
| **Footage Type** | Hint for footage characteristics (DRONE, GIMBAL, VFX, etc.)                  |
| **Tripod Mode**  | For nodal pan/tilt shots - uses rotation-only solver, simpler motion model   |
| **Robust Mode**  | For difficult footage - larger search areas, faster monitoring, more markers |

---

## How You Can Help

I'm a solo developer making professional camera tracking accessible to everyone.

**Join the community:** [Discord](https://discord.gg/qUvrXHP9PU)

### 1. Test and Report Issues

- **Found a bug?** Open an [issue on GitHub](https://github.com/yourusername/AutoSolve/issues)
- **Feature request?** Share on [Discord](https://discord.gg/qUvrXHP9PU)

### 2. Contribute Training Data

AutoSolve gets smarter through community data. Your anonymized tracking sessions help improve defaults for everyone.

**How to share your data:**

1. In Blender: `Movie Clip Editor → AutoSolve → Training Data → Export`
2. Share on [Discord](https://discord.gg/qUvrXHP9PU) or email: `usamasq@gmail.com`

**What's collected:**

- ✅ Settings used (pattern size, correlation, etc.)
- ✅ Success/failure metrics per region
- ✅ Solve error and track statistics
- ❌ NO file paths, images, or personal data

### 3. Contribute Code

See the [Contributing Guide](#contributing) below.

---

## Contributing

### Getting Started

```bash
git clone https://github.com/yourusername/AutoSolve.git
cd AutoSolve
```

### Project Structure

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
    ├── constants.py          # Shared constants (REGIONS, TIERED_SETTINGS)
    ├── utils.py              # Utility functions (get_region, etc.)
    └── learning/             # Learning components
        ├── session_recorder.py        # Session telemetry collection
        ├── settings_predictor.py      # Optimal settings prediction
        ├── behavior_recorder.py       # User behavior recording
        ├── failure_diagnostics.py     # Failure analysis & fixes
        └── pretrained_model.json      # Bundled community defaults
```

### Key Files to Understand

| File                     | Purpose                                 |
| ------------------------ | --------------------------------------- |
| `smart_tracker.py`       | Main tracking logic, settings, learning |
| `operators.py`           | Modal pipeline phases                   |
| `failure_diagnostics.py` | Failure pattern detection               |
| `pretrained_model.json`  | Default settings from community data    |

### Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Code structure and data flow
- **[TRAINING_DATA.md](TRAINING_DATA.md)** - Learning system details

---

## Planned Features

### In Progress

- [ ] **UI for footage type selection** - Dropdown in panel
- [ ] **Retry with diagnosis** - Automatic retry using failure analysis

### Future

- [ ] **Setup Tracking Scene** - Auto-create camera and background
- [ ] **Community Model Sync** - Download improved defaults
- [ ] **Real-time Motion Estimation** - Analyze optical flow before tracking
- [ ] **Neural Network Model** - Replace heuristics with ML

---

## Data for Training

Want to help build the best tracking algorithm? Here's how to collect quality training data:

### Optimal Footage for Training

| Type                      | Examples                | Why Useful                   |
| ------------------------- | ----------------------- | ---------------------------- |
| **Varied motion**         | Handheld, gimbal, drone | Tests different search sizes |
| **Different resolutions** | 720p, 1080p, 4K         | Tests scaling behavior       |
| **Challenging scenes**    | Low light, motion blur  | Tests robust mode            |
| **Clean plates**          | Studio, VFX shoots      | Baseline performance         |

### Labeling Your Data

When exporting, add context to your email:

```
Footage: Drone beach flyover
Resolution: 4K
FPS: 24
Result: Success / Fail
Notes: Required 2 retries, edges struggled
```

### Submit Data

### Submit Data

Please refer to **[CONTRIBUTING_DATA.md](CONTRIBUTING_DATA.md)** for:

- ✅ Data privacy details
- ✅ Export instructions
- ✅ Data quality guidelines

---

## License

**GPL-3.0-or-later**

---

## Credits

**Developer:** Usama Bin Shahid  
**Contact:** usamasq@gmail.com

_Your contributions make this better for everyone!_
