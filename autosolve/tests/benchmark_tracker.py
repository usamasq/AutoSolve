# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Performance benchmarks for Smart Tracker.

These benchmarks require Blender to run. Execute from within Blender's
Python console or via the operator.

Usage:
    1. Open Blender
    2. Load a test clip in the Movie Clip Editor
    3. Run this script from Text Editor
"""

import time
from typing import Dict, List, Tuple


def benchmark_decorator(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  ⏱️  {func.__name__}: {elapsed:.3f}s")
        return result, elapsed
    return wrapper


class TrackerBenchmark:
    """Benchmark suite for smart tracker performance."""
    
    def __init__(self, clip):
        """Initialize with a Blender MovieClip."""
        self.clip = clip
        self.results: Dict[str, List[float]] = {}
    
    def _record_result(self, name: str, elapsed: float):
        """Record a benchmark result."""
        if name not in self.results:
            self.results[name] = []
        self.results[name].append(elapsed)
    
    @benchmark_decorator
    def benchmark_detect_in_region(self, tracker, iterations: int = 9):
        """Benchmark per-region detection (old method)."""
        import bpy
        from autosolve.solver.smart_tracker import CoverageAnalyzer
        
        total = 0
        for region in CoverageAnalyzer.REGIONS:
            detected = tracker.detect_in_region(region, count=3)
            total += detected
        return total
    
    @benchmark_decorator  
    def benchmark_detect_all_regions(self, tracker):
        """Benchmark single-pass detection (optimized method)."""
        results = tracker.detect_all_regions(markers_per_region=3)
        return sum(results.values())
    
    @benchmark_decorator
    def benchmark_cleanup_tracks(self, tracker):
        """Benchmark track cleanup filters."""
        removed = tracker.cleanup_tracks()
        return removed
    
    @benchmark_decorator
    def benchmark_motion_probe(self, tracker):
        """Benchmark motion probe execution."""
        # Clear cached probe to force re-run
        tracker.cached_motion_probe = None
        result = tracker._run_motion_probe()
        return result
    
    def run_detection_comparison(self) -> Dict:
        """
        Compare old vs new detection methods.
        
        Returns comparative timing data.
        """
        print("\n" + "="*60)
        print("DETECTION METHOD COMPARISON")
        print("="*60)
        
        from autosolve.solver.smart_tracker import SmartTracker
        
        # Test detect_all_regions (new optimized method)
        tracker = SmartTracker(self.clip, robust_mode=False, footage_type='AUTO')
        tracker.clear_tracks()
        
        results_new, time_new = self.benchmark_detect_all_regions(tracker)
        count_new = results_new
        
        # Test detect_in_region (old method) - only if we want comparison
        tracker2 = SmartTracker(self.clip, robust_mode=False, footage_type='AUTO')
        tracker2.clear_tracks()
        
        results_old, time_old = self.benchmark_detect_in_region(tracker2)
        count_old = results_old
        
        # Results
        speedup = (time_old / time_new) if time_new > 0 else 0
        
        print(f"\n📊 Results:")
        print(f"   Old method (per-region): {time_old:.3f}s, {count_old} markers")
        print(f"   New method (single-pass): {time_new:.3f}s, {count_new} markers")
        print(f"   Speedup: {speedup:.1f}x faster")
        
        return {
            'old_time': time_old,
            'new_time': time_new,
            'speedup': speedup,
            'old_count': count_old,
            'new_count': count_new,
        }
    
    def run_full_benchmark(self) -> Dict:
        """Run full benchmark suite."""
        print("\n" + "="*60)
        print("SMART TRACKER PERFORMANCE BENCHMARK")
        print("="*60)
        print(f"Clip: {self.clip.name}")
        print(f"Size: {self.clip.size[0]}x{self.clip.size[1]}")
        print(f"Frames: {self.clip.frame_duration}")
        print("="*60)
        
        from autosolve.solver.smart_tracker import SmartTracker
        
        tracker = SmartTracker(self.clip, robust_mode=False, footage_type='AUTO')
        
        results = {}
        
        # Motion probe
        print("\n🔍 Motion Probe:")
        _, probe_time = self.benchmark_motion_probe(tracker)
        results['probe_time'] = probe_time
        
        # Detection
        print("\n🎯 Feature Detection:")
        _, detect_time = self.benchmark_detect_all_regions(tracker)
        results['detect_time'] = detect_time
        
        # Cleanup
        print("\n🧹 Track Cleanup:")
        _, cleanup_time = self.benchmark_cleanup_tracks(tracker)
        results['cleanup_time'] = cleanup_time
        
        # Summary
        total = probe_time + detect_time + cleanup_time
        print("\n" + "="*60)
        print(f"📊 TOTAL TIME: {total:.3f}s")
        print("="*60)
        
        results['total_time'] = total
        return results


def run_benchmark():
    """Run benchmark from Blender."""
    import bpy
    
    clip = bpy.context.edit_movieclip
    if not clip:
        print("❌ No movie clip loaded!")
        return
    
    benchmark = TrackerBenchmark(clip)
    benchmark.run_full_benchmark()


if __name__ == '__main__':
    run_benchmark()
