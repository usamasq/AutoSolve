
import time
import random
import sys
import statistics
from unittest.mock import MagicMock

# Mock mathutils
class Vector:
    def __init__(self, co):
        self.x = co[0]
        self.y = co[1]

    def __sub__(self, other):
        return Vector((self.x - other.x, self.y - other.y))

    @property
    def length(self):
        return (self.x**2 + self.y**2)**0.5

# Mock bpy
class MockMarker:
    def __init__(self, frame, co, mute=False):
        self.frame = frame
        self.co = MagicMock()
        self.co.x = co[0]
        self.co.y = co[1]
        self.mute = mute

class MockTrack:
    def __init__(self, name, markers):
        self.name = name
        self.markers = markers
        self.select = False

class MockTracking:
    def __init__(self, tracks):
        self.tracks = tracks

class MockContext:
    scene = MagicMock()
    scene.frame_current = 1

sys.modules['bpy'] = MagicMock()
sys.modules['bpy'].context = MockContext()
sys.modules['mathutils'] = MagicMock()
sys.modules['mathutils'].Vector = Vector

# Import the function to test
# We'll copy the function here to avoid complex imports with missing deps
class FilteringMixin:
    SAFE_MIN_TRACKS = 10
    ABSOLUTE_MIN_TRACKS = 5

    def __init__(self, tracking):
        self.tracking = tracking
        self.recorder = None

    def _run_ops(self, op, **kwargs):
        pass

    def select_all_tracks(self):
        pass

    # Original function
    def filter_spikes_original(self, limit_multiplier: float = 8.0):
        """Filter velocity outliers."""
        current = len(self.tracking.tracks)

        if current < self.SAFE_MIN_TRACKS:
            return

        track_speeds = {}
        total_speed = 0.0
        count = 0

        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue

            markers.sort(key=lambda x: x.frame)
            displacement = (Vector((markers[-1].co.x, markers[-1].co.y)) - Vector((markers[0].co.x, markers[0].co.y))).length
            duration = abs(markers[-1].frame - markers[0].frame)

            if duration > 0:
                speed = displacement / duration
                track_speeds[track.name] = speed
                total_speed += speed
                count += 1

        if count == 0:
            return

        avg = max(total_speed / count, 0.001)
        limit = avg * limit_multiplier

        to_delete = [n for n, s in track_speeds.items() if s > limit]
        max_del = min(len(to_delete), current - self.ABSOLUTE_MIN_TRACKS)

        if max_del <= 0:
            return

        sorted_tracks = sorted(track_speeds.items(), key=lambda x: x[1], reverse=True)
        to_delete = set(n for n, _ in sorted_tracks[:max_del] if track_speeds[n] > limit)

        # for track in self.tracking.tracks:
        #     track.select = track.name in to_delete

        # ... skipping ops calls for benchmark ...

    # Optimized function
    def filter_spikes_optimized(self, limit_multiplier: float = 8.0):
        """Filter velocity outliers."""
        current = len(self.tracking.tracks)

        if current < self.SAFE_MIN_TRACKS:
            return

        track_speeds = {}
        total_speed = 0.0
        count = 0

        for track in self.tracking.tracks:
            # First pass: find min/max frame markers linearly
            min_frame = float('inf')
            max_frame = float('-inf')
            min_marker = None
            max_marker = None

            active_count = 0

            for m in track.markers:
                if m.mute:
                    continue
                active_count += 1
                if m.frame < min_frame:
                    min_frame = m.frame
                    min_marker = m
                if m.frame > max_frame:
                    max_frame = m.frame
                    max_marker = m

            if active_count < 2 or not min_marker or not max_marker:
                continue

            # No sorting needed!

            displacement = (Vector((max_marker.co.x, max_marker.co.y)) - Vector((min_marker.co.x, min_marker.co.y))).length
            duration = abs(max_frame - min_frame)

            if duration > 0:
                speed = displacement / duration
                track_speeds[track.name] = speed
                total_speed += speed
                count += 1

        if count == 0:
            return

        avg = max(total_speed / count, 0.001)
        limit = avg * limit_multiplier

        to_delete = [n for n, s in track_speeds.items() if s > limit]
        max_del = min(len(to_delete), current - self.ABSOLUTE_MIN_TRACKS)

        if max_del <= 0:
            return

        sorted_tracks = sorted(track_speeds.items(), key=lambda x: x[1], reverse=True)
        to_delete = set(n for n, _ in sorted_tracks[:max_del] if track_speeds[n] > limit)


def generate_test_data(num_tracks=1000, markers_per_track=200):
    tracks = []
    for i in range(num_tracks):
        markers = []
        # Randomize frame order to simulate unsorted data
        frames = list(range(1, markers_per_track + 1))
        random.shuffle(frames)

        for f in frames:
            markers.append(MockMarker(f, (random.random(), random.random())))

        tracks.append(MockTrack(f"Track_{i}", markers))
    return MockTracking(tracks)

def run_benchmark():
    num_tracks = 1000
    markers_per_track = 150

    print(f"Generating data: {num_tracks} tracks, {markers_per_track} markers/track...")
    tracking = generate_test_data(num_tracks, markers_per_track)
    mixer = FilteringMixin(tracking)

    print("Running Original...")
    start = time.time()
    mixer.filter_spikes_original()
    end = time.time()
    orig_time = end - start
    print(f"Original Time: {orig_time:.4f}s")

    print("Running Optimized...")
    start = time.time()
    mixer.filter_spikes_optimized()
    end = time.time()
    opt_time = end - start
    print(f"Optimized Time: {opt_time:.4f}s")

    improvement = (orig_time - opt_time) / orig_time * 100
    print(f"Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    run_benchmark()
