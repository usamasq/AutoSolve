## 2024-05-23 - Blender API Collection Lookups
**Learning:** Iterating over `track.markers` to find a specific frame is O(N) and extremely slow in Python loops. Blender's API provides `track.markers.find_frame(frame)` which is O(1) (C-side lookup) and significantly faster.
**Action:** Always prefer `find_frame(frame)` over manual iteration when checking for marker existence at a specific frame. Also, using `Set` for membership checks (O(1)) instead of `List` (O(N)) in tight loops yields massive performance gains (94%+).
