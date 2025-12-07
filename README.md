# AutoSolve - One-Click Camera Tracking for Blender

> **A personal project by Usama Bin Shahid**  
> _Dedicated to my students, with ❤️ from Pakistan_

AutoSolve is a Blender extension that automates camera tracking using **adaptive learning algorithms**. It replaces Blender's manual tracking workflow with a single "Analyze & Solve" button that learns and improves from each session.

---

## Features

| Feature                    | Description                                                    |
| -------------------------- | -------------------------------------------------------------- |
| **One-Click Solve**        | Automatic feature detection, tracking, and camera solve        |
| **Adaptive Learning**      | Learns from your footage to improve settings over time         |
| **Exploratory Detection**  | Uses varied settings per region to discover optimal parameters |
| **Bidirectional Tracking** | Starts from mid-clip for better early frame coverage           |
| **Failure Diagnosis**      | Detects why tracking failed and applies targeted fixes         |
| **Footage Type Presets**   | Optimized settings for DRONE, INDOOR, HANDHELD, etc.           |

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

| Option           | Purpose                                                |
| ---------------- | ------------------------------------------------------ |
| **Footage Type** | Hint for footage characteristics (DRONE, INDOOR, etc.) |
| **Tripod Mode**  | For stationary camera with rotational movement only    |
| **Robust Mode**  | More aggressive settings for difficult footage         |

---

## How You Can Help

I'm a solo developer making professional camera tracking accessible to everyone.

### 1. Test and Report Issues

- **Found a bug?** Open an [issue on GitHub](https://github.com/yourusername/AutoSolve/issues)
- **Feature request?** Start a [discussion](https://github.com/yourusername/AutoSolve/discussions)

### 2. Contribute Training Data

AutoSolve improves through community data. Your anonymized tracking sessions help train better defaults for everyone.

**How to export your data:**

1. In Blender: `Movie Clip Editor → AutoSolve → Training Data → Export`
2. This creates a `.json` file with your session metrics
3. **Email to:** `usamasq@gmail.com`

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
AutoSolve/
├── __init__.py              # Addon registration
├── blender_manifest.toml    # Extension manifest
├── autosolve/
│   ├── operators.py         # Main operators (Analyze & Solve)
│   ├── properties.py        # Scene properties
│   ├── ui.py                # N-Panel interface
│   └── solver/
│       ├── smart_tracker.py      # Core tracking engine
│       ├── blender_tracker.py    # Blender API wrapper
│       └── learning/
│           ├── session_recorder.py    # Session telemetry
│           ├── settings_predictor.py  # Settings prediction
│           ├── failure_diagnostics.py # Failure analysis
│           └── pretrained_model.json  # Bundled defaults
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

1. Export via `Training Data → Export`
2. Email to: **usamasq@gmail.com**
3. Include footage description

---

## License

**GPL-3.0-or-later**

---

## Credits

**Developer:** Usama Bin Shahid  
**Contact:** usamasq@gmail.com

_Your contributions make this better for everyone!_
