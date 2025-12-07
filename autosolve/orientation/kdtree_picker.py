# SPDX-FileCopyrightText: 2024-2025 Your Name
# SPDX-License-Identifier: GPL-3.0-or-later

"""
KDTree-based point cloud picker.

Since bpy.types.PointCloud doesn't support ray_cast (no faces to hit),
we use a 3D spatial search approach with mathutils.kdtree.

The algorithm:
1. Build a KDTree from all 3D points
2. On mouse move, deproject to a 3D ray
3. Search for points within a cylindrical radius of the ray
4. Return the nearest cluster for selection/highlighting
"""

import numpy as np
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree
from typing import List, Tuple, Optional


class PointCloudPicker:
    """
    Spatial picker for point clouds using KDTree.
    
    Usage:
        picker = PointCloudPicker(points_array)
        indices = picker.find_near_ray(ray_origin, ray_direction)
    """
    
    def __init__(self, points: np.ndarray):
        """
        Initialize the picker with point positions.
        
        Args:
            points: Nx3 numpy array of point positions.
        """
        self._points = points
        self._tree = KDTree(len(points))
        
        for i, pt in enumerate(points):
            self._tree.insert(Vector(pt), i)
        
        self._tree.balance()
        
        # Cache bounding box for early exit optimization
        if len(points) > 0:
            self._bbox_min = points.min(axis=0)
            self._bbox_max = points.max(axis=0)
        else:
            self._bbox_min = np.zeros(3)
            self._bbox_max = np.zeros(3)
    
    def find_near_ray(
        self,
        ray_origin: Vector,
        ray_direction: Vector,
        radius: float = 0.1,
        max_distance: float = 100.0,
        num_samples: int = 50,
    ) -> List[int]:
        """
        Find all points within a cylindrical volume along a ray.
        
        This simulates "clicking into" the point cloud by finding
        points that the mouse ray passes near.
        
        Args:
            ray_origin: Start point of the ray (camera/cursor position).
            ray_direction: Normalized direction of the ray.
            radius: Radius of the search cylinder.
            max_distance: Maximum distance along ray to search.
            num_samples: Number of sample points along the ray.
        
        Returns:
            List of point indices within the cylinder, sorted by
            distance to ray origin.
        
        Visual:
            ray_origin ●━━━━━━━━━━━━━━━━━━━━━▶ ray_direction
                       ╔═══════════════════════╗
                       ║  search cylinder      ║  ← radius
                       ╚═══════════════════════╝
                            · · · · ·  ← candidate points
        """
        candidates = set()
        
        # Sample points along the ray
        step = max_distance / num_samples
        
        for i in range(num_samples):
            t = i * step
            sample_point = ray_origin + ray_direction * t
            
            # Query KDTree for points within radius
            nearby = self._tree.find_range(sample_point, radius)
            
            for point, index, dist in nearby:
                candidates.add(index)
        
        # Sort by distance to ray origin
        if candidates:
            candidates_list = list(candidates)
            distances = [
                (ray_origin - Vector(self._points[i])).length
                for i in candidates_list
            ]
            sorted_indices = sorted(
                range(len(candidates_list)),
                key=lambda x: distances[x]
            )
            return [candidates_list[i] for i in sorted_indices]
        
        return []
    
    def find_nearest(self, position: Vector) -> Optional[Tuple[Vector, int, float]]:
        """
        Find the single nearest point to a position.
        
        Args:
            position: 3D position to search from.
        
        Returns:
            Tuple of (point_position, index, distance), or None if empty.
        """
        result = self._tree.find(position)
        return result if result[1] is not None else None
    
    def find_in_radius(self, position: Vector, radius: float) -> List[Tuple[Vector, int, float]]:
        """
        Find all points within a radius of a position.
        
        Args:
            position: Center of search sphere.
            radius: Search radius.
        
        Returns:
            List of (point_position, index, distance) tuples.
        """
        return self._tree.find_range(position, radius)
    
    def get_points(self, indices: List[int]) -> np.ndarray:
        """
        Get point positions for a list of indices.
        
        Args:
            indices: List of point indices.
        
        Returns:
            Nx3 array of point positions.
        """
        return self._points[indices]


def deproject_mouse_to_ray(
    context,
    mouse_x: int,
    mouse_y: int
) -> Tuple[Vector, Vector]:
    """
    Convert 2D mouse position to a 3D ray in world space.
    
    Args:
        context: Blender context.
        mouse_x: Mouse X coordinate (pixels from left).
        mouse_y: Mouse Y coordinate (pixels from bottom).
    
    Returns:
        Tuple of (ray_origin, ray_direction) in world space.
    """
    from bpy_extras.view3d_utils import region_2d_to_origin_3d, region_2d_to_vector_3d
    
    region = context.region
    rv3d = context.region_data
    
    coord = (mouse_x, mouse_y)
    
    ray_origin = region_2d_to_origin_3d(region, rv3d, coord)
    ray_direction = region_2d_to_vector_3d(region, rv3d, coord)
    
    return ray_origin, ray_direction
