# Contributing Training Data

> **Help make AutoSolve smarter for everyone!**

---

## Quick Start

1. **Export** your data: `AutoSolve → Training Data → Export Data`
2. **Share** via [Discord](https://discord.gg/kkAmxKsS) or email: [usamasq@gmail.com](mailto:usamasq@gmail.com)
3. That's it! Your data helps improve tracking for the whole community.

---

## What Data Is Collected?

AutoSolve collects **anonymized numerical metrics only**:

| ✅ Collected          | ❌ NOT Collected       |
| --------------------- | ---------------------- |
| Resolution & FPS      | File paths             |
| Track coordinates     | Image content          |
| Settings used         | User identity          |
| Success/failure rates | Personal info          |
| Camera intrinsics     | Clip names (sanitized) |

---

## Why Contribute?

Your data directly improves:

- **Default Settings** — Better presets for your footage type
- **Failure Detection** — Earlier warning when tracking will fail
- **Region Analysis** — Smarter marker placement

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

### Session Count

More sessions = better defaults!

| Sessions | Impact             |
| -------- | ------------------ |
| 5-10     | Helpful            |
| 20-50    | Significant        |
| 100+     | Major contribution |

---

## Export Format

Exported data uses **format_version: 3** with these key fields:

```json
{
  "format_version": 3,
  "export_type": "autosolve_training_data",
  "sessions": [
    {
      "resolution": [1920, 1080],
      "fps": 30,
      "settings": {...},
      "success": true,
      "solve_error": 0.42,
      "tracks": [...],
      "frame_samples": [...],
      "camera_intrinsics": {...}
    }
  ]
}
```

See [TRAINING_DATA.md](TRAINING_DATA.md) for complete schema documentation.

---

## Edit Pattern Capture

When "Record My Edits" is enabled, we also capture how pro users refine tracking:

- Which tracks get deleted and why
- Which tracks are excluded from solving
- Time spent editing

This helps AutoSolve learn expert-level track cleanup.

---

## Privacy Commitment

- All data stays **local** by default
- Sharing is **opt-in** only
- No tracking of usage
- No personal information collected

---

## Contact & Community

**Discord:** [Join our community](https://discord.gg/kkAmxKsS)  
**Email:** [usamasq@gmail.com](mailto:usamasq@gmail.com)

**GitHub:** Submit data contributions via Issues or PR
