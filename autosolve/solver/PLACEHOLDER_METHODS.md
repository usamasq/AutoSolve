# Incomplete/Placeholder Code Registry

This document lists all incomplete, placeholder, or potentially problematic code patterns across the EZTrack codebase.

---

## ~~Placeholder Methods~~ (IMPLEMENTED)

The following placeholders have been **fully implemented**:

| Method                                  | Status                                             |
| --------------------------------------- | -------------------------------------------------- |
| `SmartTracker.load_learning()`          | ✅ Implemented (placeholder for per-clip learning) |
| `SmartTracker.analyze_footage()`        | ✅ Implemented (logs classification)               |
| `SmartTracker.validate_pre_tracking()`  | ✅ NEW - Full validation                           |
| `SmartTracker.validate_track_quality()` | ✅ NEW - Live validation                           |
| `SmartTracker.validate_pre_solve()`     | ✅ NEW - Pre-solve validation                      |
| `SmartTracker.extract_training_data()`  | ✅ NEW - Training extraction                       |

---

## Bare Exception Handlers

These swallow all exceptions silently. Consider logging or handling specific exception types.

| File                                  | Line | Context                                      |
| ------------------------------------- | ---- | -------------------------------------------- |
| `autosolve/solver/smart_tracker.py`   | 375  | `_get_data_dir`: Fallback to tempdir         |
| `autosolve/solver/smart_tracker.py`   | 708  | `clear_tracks`: Delete track fails           |
| `autosolve/solver/smart_tracker.py`   | 881  | `filter_spikes`: Delete outliers fails       |
| `autosolve/solver/smart_tracker.py`   | 910  | `filter_high_error`: Delete high-error fails |
| `autosolve/solver/blender_tracker.py` | 224  | `_clean_tracks`: Blender API fallback        |
| `autosolve/ui.py`                     | 169  | Training panel: Model load fails             |

---

## Notes

The learning functionality now includes:

- `LocalLearningModel` — Stores per-footage-class learning
- `_load_initial_settings()` — Applies learned settings on init
- `save_session_results()` — Updates model after solve
- `extract_training_data()` — **NEW**: Captures region/velocity patterns
