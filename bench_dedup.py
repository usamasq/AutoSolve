
import time
import random
import sys
import math
from collections import defaultdict
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

class MockMarkers:
    def __init__(self, markers_list):
        self._markers = markers_list
        # Build index for O(1) lookup
        self._map = {m.frame: m for m in markers_list}

    def find_frame(self, frame):
        return self._map.get(frame)

    def __iter__(self):
        return iter(self._markers)

    def __len__(self):
        return len(self._markers)

class MockTrack:
    def __init__(self, name, markers_list):
        self.name = name
        self.markers = MockMarkers(markers_list)
        self.select = False

class TracksCollection:
    def __init__(self, tracks):
        self._tracks = tracks
        self._map = {t.name: t for t in tracks}
    def __iter__(self):
        return iter(self._tracks)
    def __len__(self):
        return len(self._tracks)
    def get(self, name):
        return self._map.get(name)

class MockTracking:
    def __init__(self, tracks):
        self.tracks = TracksCollection(tracks)

class MockContext:
    scene = MagicMock()
    scene.frame_current = 1

sys.modules['bpy'] = MagicMock()
sys.modules['bpy'].context = MockContext()
sys.modules['mathutils'] = MagicMock()
sys.modules['mathutils'].Vector = Vector

class MockClip:
    size = [1920, 1080]

class FilteringMixin:
    SAFE_MIN_TRACKS = 10

    def __init__(self, tracking):
        self.tracking = tracking
        self.clip = MockClip()

    def _run_ops(self, op, **kwargs):
        pass

    def select_all_tracks(self):
        pass

    def scene_to_clip_frame(self, frame):
        return frame

    def _get_region_for_pos(self, x: float, y: float) -> str:
        col = 0 if x < 0.33 else (1 if x < 0.66 else 2)
        row = 2 if y < 0.33 else (1 if y < 0.66 else 0)
        regions = [
            ['top-left', 'top-center', 'top-right'],
            ['mid-left', 'center', 'mid-right'],
            ['bottom-left', 'bottom-center', 'bottom-right']
        ]
        return regions[row][col]

    def deduplicate_tracks_original(self, min_distance_px: int = 30):
        # Simulate logic without coverage checks
        # Assume we are in a saturated state for the benchmark

        current = len(self.tracking.tracks)

        # Manually force all regions to be "saturated" for the test
        saturated_regions = {'center', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'top-center', 'bottom-center', 'mid-left', 'mid-right'}

        width = self.clip.size[0]
        # Match implementation in filtering.py which uses hardcoded 15px relative width
        min_dist_norm = 15 / width

        tracks_by_region = {}
        current_frame = 1 # mocked
        clip_frame = 1

        for track in self.tracking.tracks:
            marker = track.markers.find_frame(clip_frame)
            if marker:
                pos = (marker.co.x, marker.co.y)
                region = self._get_region_for_pos(pos[0], pos[1])

                if region not in tracks_by_region:
                    tracks_by_region[region] = []
                tracks_by_region[region].append((track.name, pos))

        to_delete = set()

        for region in saturated_regions:
            if region not in tracks_by_region:
                continue

            region_tracks = tracks_by_region[region]

            grid = defaultdict(list)
            cell_size = min_dist_norm

            for name, pos in region_tracks:
                cx = math.floor(pos[0] / cell_size)
                cy = math.floor(pos[1] / cell_size)
                grid[(cx, cy)].append((name, pos))

            def check_and_mark(n1, p1, n2, p2):
                if n1 in to_delete or n2 in to_delete:
                    return

                d = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
                if d < min_dist_norm:
                    t1 = self.tracking.tracks.get(n1)
                    t2 = self.tracking.tracks.get(n2)
                    if t1 and t2:
                        l1 = len(t1.markers)
                        l2 = len(t2.markers)
                        to_delete.add(n1 if l1 < l2 else n2)

            for (cx, cy), entries in sorted(grid.items()):
                for i in range(len(entries)):
                    name1, pos1 = entries[i]
                    for j in range(i + 1, len(entries)):
                        name2, pos2 = entries[j]
                        check_and_mark(name1, pos1, name2, pos2)

                neighbor_offsets = [(1, 0), (0, 1), (1, 1), (-1, 1)]
                for dx, dy in neighbor_offsets:
                    neighbor_key = (cx + dx, cy + dy)
                    if neighbor_key in grid:
                        neighbor_entries = grid[neighbor_key]
                        for name1, pos1 in entries:
                            for name2, pos2 in neighbor_entries:
                                check_and_mark(name1, pos1, name2, pos2)

        return len(to_delete)

    def deduplicate_tracks_optimized(self, min_distance_px: int = 30):
        current = len(self.tracking.tracks)

        saturated_regions = {'center', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'top-center', 'bottom-center', 'mid-left', 'mid-right'}

        width = self.clip.size[0]
        min_dist_norm = 15 / width

        # PRE-CALCULATE SQUARED THRESHOLD
        min_dist_norm_sq = min_dist_norm * min_dist_norm

        tracks_by_region = {}
        current_frame = 1
        clip_frame = 1

        for track in self.tracking.tracks:
            marker = track.markers.find_frame(clip_frame)
            if marker:
                pos = (marker.co.x, marker.co.y)
                region = self._get_region_for_pos(pos[0], pos[1])

                if region not in tracks_by_region:
                    tracks_by_region[region] = []
                tracks_by_region[region].append((track.name, pos))

        to_delete = set()

        for region in saturated_regions:
            if region not in tracks_by_region:
                continue

            region_tracks = tracks_by_region[region]

            grid = defaultdict(list)
            cell_size = min_dist_norm

            for name, pos in region_tracks:
                cx = math.floor(pos[0] / cell_size)
                cy = math.floor(pos[1] / cell_size)
                grid[(cx, cy)].append((name, pos))

            def check_and_mark(n1, p1, n2, p2):
                if n1 in to_delete or n2 in to_delete:
                    return

                # OPTIMIZED: Squared distance check
                d_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                if d_sq < min_dist_norm_sq:
                    t1 = self.tracking.tracks.get(n1)
                    t2 = self.tracking.tracks.get(n2)
                    if t1 and t2:
                        l1 = len(t1.markers)
                        l2 = len(t2.markers)
                        to_delete.add(n1 if l1 < l2 else n2)

            for (cx, cy), entries in sorted(grid.items()):
                for i in range(len(entries)):
                    name1, pos1 = entries[i]
                    for j in range(i + 1, len(entries)):
                        name2, pos2 = entries[j]
                        check_and_mark(name1, pos1, name2, pos2)

                neighbor_offsets = [(1, 0), (0, 1), (1, 1), (-1, 1)]
                for dx, dy in neighbor_offsets:
                    neighbor_key = (cx + dx, cy + dy)
                    if neighbor_key in grid:
                        neighbor_entries = grid[neighbor_key]
                        for name1, pos1 in entries:
                            for name2, pos2 in neighbor_entries:
                                check_and_mark(name1, pos1, name2, pos2)

        return len(to_delete)

def generate_test_data(num_tracks=2000):
    tracks = []
    # Generate tracks concentrated in center to trigger max collisions
    for i in range(num_tracks):
        markers = []
        # Frame 1 is what matters for the test
        # Random positions near center (0.5, 0.5)
        # Using a small range to ensure high density
        x = 0.5 + random.uniform(-0.1, 0.1)
        y = 0.5 + random.uniform(-0.1, 0.1)

        markers.append(MockMarker(1, (x, y)))
        # Add some dummy markers for length check
        for f in range(2, 20):
             markers.append(MockMarker(f, (x, y)))

        tracks.append(MockTrack(f"Track_{i}", markers))
    return MockTracking(tracks)

def run_benchmark():
    num_tracks = 3000
    print(f"Generating data: {num_tracks} tracks...")
    tracking = generate_test_data(num_tracks)
    mixer = FilteringMixin(tracking)

    print("Running Original...")
    start = time.time()
    res_orig = mixer.deduplicate_tracks_original()
    end = time.time()
    orig_time = end - start
    print(f"Original Time: {orig_time:.4f}s (Removed {res_orig})")

    # Reset (though we didn't actually delete tracks from tracking list in mock)

    print("Running Optimized...")
    start = time.time()
    res_opt = mixer.deduplicate_tracks_optimized()
    end = time.time()
    opt_time = end - start
    print(f"Optimized Time: {opt_time:.4f}s (Removed {res_opt})")

    assert res_orig == res_opt, f"Results mismatch! {res_orig} vs {res_opt}"

    improvement = (orig_time - opt_time) / orig_time * 100
    print(f"Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    run_benchmark()
