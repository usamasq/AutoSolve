# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
TrackAverager - Creates high-quality anchor tracks by averaging nearby clusters.

Uses Blender's bpy.ops.clip.average_tracks operator to combine nearby tracks
into single high-quality averaged tracks with reduced noise.

Integration points:
1. FilteringMixin.average_clustered_tracks() - Called during cleanup_tracks pipeline
2. TrackHealer.merge_overlapping_segments() - Called during healing phase
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

try:
    import bpy
except ImportError:
    bpy = None


class TrackAverager:
    """
    Creates high-quality anchor tracks by averaging nearby track clusters.
    
    Algorithm:
    1. Build spatial clusters of tracks within proximity threshold
    2. For each cluster (2+ tracks), average them using Blender's operator
    3. Remove source tracks (optional) to avoid duplication
    
    Usage:
        averager = TrackAverager()
        created = averager.create_anchor_tracks(tracking)
        print(f"Created {created} averaged anchors")
    """
    
    # Default threshold: 1.5% of frame width (~29px on 1080p)
    DEFAULT_PROXIMITY = 0.015
    
    # Minimum cluster size to average
    MIN_CLUSTER_SIZE = 2
    
    def __init__(self, proximity_threshold: float = None):
        """
        Initialize TrackAverager.
        
        Args:
            proximity_threshold: Distance threshold in normalized coords (0-1).
                                 Default is 0.015 (1.5% of frame).
        """
        self.proximity = proximity_threshold or self.DEFAULT_PROXIMITY
    
    def find_track_clusters(self, tracking, frame: int = None) -> List[List[str]]:
        """
        Find groups of tracks that are spatially close together.
        
        Uses simple distance-based clustering: tracks within proximity
        threshold are grouped together.
        
        Args:
            tracking: bpy.types.MovieTracking object
            frame: Frame to analyze positions at (default: current frame)
            
        Returns:
            List of clusters, each cluster is a list of track names
        """
        if not bpy:
            return []
        
        if frame is None:
            frame = bpy.context.scene.frame_current
        
        # Collect track positions at specified frame
        track_positions: Dict[str, Tuple[float, float]] = {}
        
        for track in tracking.tracks:
            # Get marker at this frame
            marker = track.markers.find_frame(frame)
            if not marker:
                # Try to find any active marker
                markers = [m for m in track.markers if not m.mute]
                if markers:
                    marker = markers[len(markers) // 2]  # Use middle marker
            
            if marker and not marker.mute:
                track_positions[track.name] = (marker.co.x, marker.co.y)
        
        if len(track_positions) < 2:
            return []
        
        # Build clusters using union-find approach
        clusters = self._cluster_by_proximity(track_positions)
        
        # Filter to clusters with 2+ tracks
        valid_clusters = [c for c in clusters if len(c) >= self.MIN_CLUSTER_SIZE]
        
        return valid_clusters
    
    def _cluster_by_proximity(self, positions: Dict[str, Tuple[float, float]]) -> List[List[str]]:
        """
        Cluster tracks by spatial proximity using union-find with spatial hashing.
        
        Args:
            positions: Dict of track_name -> (x, y) normalized position
            
        Returns:
            List of clusters (lists of track names)
        """
        # Union-Find structure
        parent = {name: name for name in positions}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Grid hashing for O(N) neighbor search
        grid = defaultdict(list)
        cell_size = self.proximity

        for name, pos in positions.items():
            cx = int(pos[0] / cell_size)
            cy = int(pos[1] / cell_size)
            grid[(cx, cy)].append((name, pos))

        # Check neighbors
        for (cx, cy), entries in grid.items():
            # Check within cell
            for i in range(len(entries)):
                name1, pos1 = entries[i]
                # Check against other items in same cell
                for j in range(i + 1, len(entries)):
                    name2, pos2 = entries[j]

                    dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) ** 0.5
                    if dist < self.proximity:
                        union(name1, name2)

            # Check neighbor cells (only forward directions to avoid duplicates)
            # Right, Down, Down-Right, Down-Left
            neighbor_offsets = [(1, 0), (0, 1), (1, 1), (-1, 1)]
            for dx, dy in neighbor_offsets:
                neighbor_key = (cx + dx, cy + dy)
                if neighbor_key in grid:
                    neighbor_entries = grid[neighbor_key]
                    for name1, pos1 in entries:
                        for name2, pos2 in neighbor_entries:
                            dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) ** 0.5
                            if dist < self.proximity:
                                union(name1, name2)
        
        # Collect clusters
        clusters_dict = defaultdict(list)
        for name in names:
            root = find(name)
            clusters_dict[root].append(name)
        
        return list(clusters_dict.values())
    
    def average_cluster(self, tracking, cluster: List[str], 
                       keep_originals: bool = False) -> Optional[str]:
        """
        Average tracks in a cluster using Blender's average_tracks operator.
        
        Args:
            tracking: bpy.types.MovieTracking object
            cluster: List of track names to average
            keep_originals: If True, preserve original tracks (default: False)
            
        Returns:
            Name of the averaged track, or None if failed
        """
        if not bpy or len(cluster) < 2:
            return None
        
        # Clear selection
        for track in tracking.tracks:
            track.select = False
        
        # Select cluster tracks, set first as active
        active_track = None
        selected_count = 0
        
        for name in cluster:
            track = tracking.tracks.get(name)
            if track:
                track.select = True
                selected_count += 1
                if active_track is None:
                    active_track = track
        
        if selected_count < 2 or active_track is None:
            return None
        
        # Set active track
        tracking.tracks.active = active_track
        original_name = active_track.name
        
        try:
            # Call Blender's average_tracks operator
            # This averages selected tracks into the active track
            result = bpy.ops.clip.average_tracks(keep_original=keep_originals)
            
            if result == {'FINISHED'}:
                print(f"AutoSolve: Averaged {selected_count} tracks → {original_name}")
                return original_name
            else:
                return None
                
        except Exception as e:
            print(f"AutoSolve: average_tracks failed: {e}")
            return None
    
    def create_anchor_tracks(self, tracking, 
                            keep_originals: bool = False) -> int:
        """
        Main entry point: Find clusters and create averaged anchor tracks.
        
        This is the method to call from FilteringMixin.average_clustered_tracks().
        
        Args:
            tracking: bpy.types.MovieTracking object
            keep_originals: If True, preserve original tracks
            
        Returns:
            Number of averaged anchor tracks created
        """
        clusters = self.find_track_clusters(tracking)
        
        if not clusters:
            return 0
        
        created = 0
        for cluster in clusters:
            result = self.average_cluster(tracking, cluster, keep_originals)
            if result:
                created += 1
        
        if created > 0:
            print(f"AutoSolve: Created {created} averaged anchor tracks "
                  f"from {sum(len(c) for c in clusters)} source tracks")
        
        return created


def merge_overlapping_segments(tracking, min_overlap: int = 5) -> int:
    """
    Find and average track segments with significant frame overlap.
    
    When two tracks have overlapping frame ranges (>= min_overlap frames)
    AND are spatially close, they likely represent the same feature
    tracked twice. Averaging them produces a more accurate combined track.
    
    This is called during the healing phase.
    
    Args:
        tracking: bpy.types.MovieTracking object
        min_overlap: Minimum overlapping frames required (default: 5)
        
    Returns:
        Number of track pairs merged
    """
    if not bpy:
        return 0
    
    # Collect track frame ranges and positions
    track_info: Dict[str, Dict] = {}
    
    for track in tracking.tracks:
        markers = [m for m in track.markers if not m.mute]
        if len(markers) < 3:
            continue
        
        markers_sorted = sorted(markers, key=lambda m: m.frame)
        start_frame = markers_sorted[0].frame
        end_frame = markers_sorted[-1].frame
        
        # Get average position
        avg_x = sum(m.co.x for m in markers_sorted) / len(markers_sorted)
        avg_y = sum(m.co.y for m in markers_sorted) / len(markers_sorted)
        
        track_info[track.name] = {
            'start': start_frame,
            'end': end_frame,
            'lifespan': end_frame - start_frame,
            'position': (avg_x, avg_y),
            'marker_count': len(markers_sorted),
        }
    
    if len(track_info) < 2:
        return 0
    
    # Find overlapping pairs that are also spatially close
    SPATIAL_THRESHOLD = 0.02  # 2% of frame
    merge_candidates: List[Tuple[str, str]] = []
    
    # Use grid hashing to avoid O(N^2)
    grid = defaultdict(list)
    cell_size = SPATIAL_THRESHOLD

    for name, info in track_info.items():
        pos = info['position']
        cx = int(pos[0] / cell_size)
        cy = int(pos[1] / cell_size)
        grid[(cx, cy)].append((name, info))

    # Check neighbors in grid
    for (cx, cy), entries in grid.items():
        # Check within same cell
        for i in range(len(entries)):
            name1, info1 = entries[i]
            for j in range(i + 1, len(entries)):
                name2, info2 = entries[j]

                # Check temporal overlap
                overlap_start = max(info1['start'], info2['start'])
                overlap_end = min(info1['end'], info2['end'])
                overlap = overlap_end - overlap_start

                if overlap < min_overlap:
                    continue

                # Check spatial proximity
                pos1, pos2 = info1['position'], info2['position']
                dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) ** 0.5

                if dist < SPATIAL_THRESHOLD:
                    merge_candidates.append((name1, name2))

        # Check neighbor cells (forward directions only)
        neighbor_offsets = [(1, 0), (0, 1), (1, 1), (-1, 1)]
        for dx, dy in neighbor_offsets:
            neighbor_key = (cx + dx, cy + dy)
            if neighbor_key in grid:
                neighbor_entries = grid[neighbor_key]
                for name1, info1 in entries:
                    for name2, info2 in neighbor_entries:

                        # Check temporal overlap
                        overlap_start = max(info1['start'], info2['start'])
                        overlap_end = min(info1['end'], info2['end'])
                        overlap = overlap_end - overlap_start

                        if overlap < min_overlap:
                            continue

                        # Check spatial proximity
                        pos1, pos2 = info1['position'], info2['position']
                        dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) ** 0.5

                        if dist < SPATIAL_THRESHOLD:
                            merge_candidates.append((name1, name2))
    
    if not merge_candidates:
        return 0
    
    # Merge candidates using averaging
    averager = TrackAverager(proximity_threshold=SPATIAL_THRESHOLD)
    merged = 0
    
    # Process pairs (avoid double-merging by tracking which tracks are consumed)
    consumed: Set[str] = set()
    
    for name1, name2 in merge_candidates:
        if name1 in consumed or name2 in consumed:
            continue
        
        result = averager.average_cluster(tracking, [name1, name2], keep_originals=False)
        if result:
            merged += 1
            # One track survives (the active one), others are averaged into it
            consumed.add(name2)  # Mark second track as consumed
    
    if merged > 0:
        print(f"AutoSolve: Merged {merged} overlapping track segment pairs")
    
    return merged
