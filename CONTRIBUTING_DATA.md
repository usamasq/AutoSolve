# Contributing Training Data

> **Help make AutoSolve smarter for everyone!**

> [!IMPORTANT] > **🧪 Research Beta** - AutoSolve is actively learning from community data.
> Every contribution directly improves the model that predicts optimal tracking settings.

---

## Quick Start

1. **Export** your data: `AutoSolve → Training Data → Export Data`
2. **Share** via [Discord](https://discord.gg/kkAmxKsS) or email: [usamasq@gmail.com](mailto:usamasq@gmail.com)
3. That's it! Your data helps improve tracking for the whole community.

---

## What Gets Collected?

Each tracking session records **anonymized numerical data** to help AutoSolve learn what works:

### Per-Session Data

| Category                 | Fields                                               | Purpose                            |
| ------------------------ | ---------------------------------------------------- | ---------------------------------- |
| **Footage Info**         | Resolution, FPS, frame count                         | Learn resolution-specific settings |
| **Settings Used**        | Pattern size, search size, correlation, motion model | Learn what settings work           |
| **Results**              | Solve error, bundle count, success/failure           | Label for training                 |
| **Camera Intrinsics**    | Focal length, sensor size, distortion coefficients   | Improve lens handling              |
| **Motion Analysis**      | Motion class (LOW/MEDIUM/HIGH), parallax score       | Predict settings from motion       |
| **Pre-solve Confidence** | Estimated success probability before solving         | Validate prediction accuracy       |

### Per-Track Data

| Category         | Fields                              | Purpose                        |
| ---------------- | ----------------------------------- | ------------------------------ |
| **Lifecycle**    | Lifespan, start/end frame           | Predict track survival         |
| **Quality**      | Jitter score, reprojection error    | Filter bad tracks early        |
| **Spatial**      | Region (9-grid), trajectory samples | Learn region-specific behavior |
| **Contribution** | Contributed to solve (yes/no)       | Identify useful tracks         |

### Temporal Data (NEW)

| Category          | Fields                                          | Purpose                               |
| ----------------- | ----------------------------------------------- | ------------------------------------- |
| **Frame Samples** | Active tracks, tracks lost, velocity per frame  | Train RNN/LSTM for dropout prediction |
| **Optical Flow**  | Parallax score, direction entropy, dropout rate | Motion-based settings prediction      |

---

## What's NOT Collected

| ❌ NOT Collected    | Why                      |
| ------------------- | ------------------------ |
| File paths          | Privacy                  |
| Image/video content | Not needed               |
| Clip names          | Sanitized to generic IDs |
| User identity       | Anonymous by design      |
| System info         | Not relevant             |

---

## How Sessions Help Training

Each session teaches AutoSolve something:

| Session Type         | What It Teaches                             |
| -------------------- | ------------------------------------------- |
| **Successful solve** | "These settings work for this footage type" |
| **Failed solve**     | "Avoid these settings for similar footage"  |
| **High parallax**    | "Use perspective solve, not tripod"         |
| **Low motion**       | "Smaller search size is fine"               |
| **Edge failures**    | "Certain regions are unreliable"            |

### Example Learning Flow

```
Session 1: HD 30fps drone footage, solve_error = 0.3px, settings worked
Session 2: HD 30fps handheld, solve_error = 2.5px, correlation too low
Session 3: HD 30fps handheld, correlation increased, solve_error = 0.5px

→ AutoSolve learns: "Handheld footage needs higher correlation than drone"
```

---

## Data Quality Guidelines

For best results, contribute data from diverse footage:

### Ideal Contributions

| Category              | Examples                        |
| --------------------- | ------------------------------- |
| **Resolution**        | SD, HD, 4K                      |
| **Frame Rate**        | 24, 30, 60 fps                  |
| **Motion**            | Static, handheld, drone, action |
| **Environment**       | Indoor, outdoor, low-light      |
| **Success & Failure** | Both help learning!             |

### Session Count Impact

| Sessions | Impact             |
| -------- | ------------------ |
| 5-10     | Helpful            |
| 20-50    | Significant        |
| 100+     | Major contribution |

---

## Export Format

Exported data uses **schema_version: 1** with these key fields:

```json
{
  "schema_version": 1,
  "timestamp": "2025-12-09T15:30:00",
  "resolution": [1920, 1080],
  "fps": 30,
  "settings": {
    "pattern_size": 17,
    "search_size": 91,
    "correlation": 0.68,
    "motion_model": "LocRot"
  },
  "success": true,
  "solve_error": 0.42,
  "bundle_count": 45,
  "motion_probe_results": {
    "motion_class": "MEDIUM",
    "texture_class": "GOOD"
  },
  "pre_solve_confidence": {
    "confidence": 0.85,
    "parallax_score": 0.3
  },
  "region_stats": {
    "center": {"total_tracks": 8, "successful_tracks": 7}
  },
  "tracks": [
    {
      "lifespan": 180,
      "region": "center",
      "reprojection_error": 0.35,
      "trajectory": [[0.5, 0.5], [0.51, 0.49], ...]
    }
  ],
  "frame_samples": [
    {"frame": 10, "active_tracks": 32, "tracks_lost": 0}
  ]
}
```

See [TRAINING_DATA.md](TRAINING_DATA.md) for complete schema documentation.

---

## Edit Pattern Capture

When "Record My Edits" is enabled, we also capture how pro users refine tracking:

- Which tracks get deleted and why
- Which tracks are excluded from solving
- Settings adjustments made after initial solve

This helps AutoSolve learn expert-level track cleanup patterns.

---

## Privacy Commitment

- All data stays **local** by default
- Sharing is **opt-in** only
- No tracking of usage habits
- No personal information collected
- Clip names are sanitized

---

## Contact & Community

**Discord:** [Join our community](https://discord.gg/kkAmxKsS)  
**Email:** [usamasq@gmail.com](mailto:usamasq@gmail.com)

Your contributions make AutoSolve better for everyone! 🙏
