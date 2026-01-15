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
