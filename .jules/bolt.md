## 2024-05-23 - [Optimized Deduplication with Spatial Hashing]
**Learning:** O(N^2) pairwise comparisons for spatial proximity can be drastically reduced by pre-grouping elements into spatial buckets (regions). In `deduplicate_tracks`, grouping tracks by their 9 regions changed complexity from O(N^2) to O(N + sum(k^2)) where k is tracks per region.
**Action:** Always look for spatial bucketing opportunities when comparing proximity of many objects. Avoid re-calculating region/position data inside nested loops.

## 2024-05-23 - [Optimized Clustering with Spatial Hashing]
**Learning:** For stricter proximity checks (small threshold), a fine-grained grid hash map is superior to coarse regions. Implemented O(N) grid-based neighbor search for `_cluster_by_proximity` and `merge_overlapping_segments`, replacing O(N^2) loops.
**Action:** When `threshold` is small relative to domain size, use grid hashing `(int(x/thresh), int(y/thresh))` to find neighbors efficiently. Remember to check adjacent cells to catch boundary cases.
## 2024-05-24 - [Python Loop Optimization: Pre-normalization]
**Learning:** In tight loops like Gaussian smoothing, performing division and summation inside the loop (even if values are constant for the window) adds up. Pre-calculating normalized weights and using `zip` to avoid indexing provided ~8% speedup.
**Action:** When applying weighted averages with a fixed kernel, always pre-normalize the kernel weights to sum to 1.0 outside the loop to avoid repeated division.

## 2025-05-27 - [Euclidean Distance Optimization]
**Learning:** Computing `sqrt()` in inner loops for distance comparisons is expensive. Comparing squared distances against a squared threshold eliminates the square root operation and yields ~13% speedup in clustering loops.
**Action:** When comparing distances against a fixed threshold, pre-calculate `threshold_sq = threshold * threshold` and compare `dx*dx + dy*dy < threshold_sq`.

## 2026-01-16 - [Sorting Overhead in Track Statistics]
**Learning:** Calling `sorted(track.markers, key=lambda m: m.frame)` to compute min/max frames or average position introduces O(M log M) overhead per track. For statistical aggregations, a single linear pass O(M) is ~60% faster.
**Action:** Use manual min/max/sum accumulation in a single loop over `track.markers` instead of `sorted()` when order is not strictly required for the calculation.

## 2026-01-16 - [Robust Vector Mocking]
**Learning:** When mocking `mathutils.Vector` for Blender add-on testing, the mock must support both attribute access (`.x`, `.y`) and sequence protocol (`__getitem__`, `__len__`, iteration). Codebases often mix `vector.x` and `Vector(vector)` (which iterates), so a dual-compatible mock is essential for verifying logic without changing source code.
**Action:** Use a `MockVector` class that implements both `__getattr__`/properties and `__getitem__`/`__iter__` when creating standalone verification scripts.

## 2026-01-17 - [Pre-calculated Properties in Nested Loops]
**Learning:** In Python, repeated property access or list comprehensions (e.g., `len([m for m in track.markers])`) inside nested loops accumulate significant overhead. Pre-calculating these values into a dictionary (O(1) lookup) during the initial pass improved deduplication performance by ~27%.
**Action:** Always pre-calculate invariant properties (like list lengths or derived stats) into a dictionary before entering nested comparison loops (O(N^2) or O(Nk)).

## 2026-05-27 - [Optimizing Marker Counting]
**Learning:** Using `len([m for m in track.markers if not m.mute]) >= limit` creates a full list allocation just to check a count, which is expensive for tracks with many markers.
**Action:** Use a helper function with a loop and early exit (short-circuiting) to count active markers. This avoids list allocation and can return as soon as the limit is reached, providing up to 10x speedup for boolean checks.

## 2026-05-28 - [O(1) Blender C-API Lookups]
**Learning:** Iterating through `track.markers` (O(M)) to check for specific frames is significantly slower than using Blender's C-API optimized `track.markers.find_frame(frame)` (O(1)). In `filter_high_error`, replacing manual loops with `find_frame` yielded a ~92% speedup.
**Action:** Always prefer `collection.find_frame(frame)` or similar C-API lookup methods over manual Python loops when checking for existence or retrieving items by key.
