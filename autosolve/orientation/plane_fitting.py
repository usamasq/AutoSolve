# SPDX-FileCopyrightText: 2024-2025 Your Name
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Plane fitting algorithms.

Provides SVD and RANSAC-based plane fitting for ground plane detection.
Used by the Ground Wand tool to determine the floor orientation.
"""

import numpy as np
from mathutils import Vector, Matrix
from typing import Tuple, Optional


def fit_plane_svd(points: np.ndarray) -> Tuple[Vector, Vector]:
    """
    Fit a plane to 3D points using Singular Value Decomposition.
    
    The plane is defined by its centroid and normal vector.
    
    Args:
        points: Nx3 numpy array of point positions.
    
    Returns:
        Tuple of (centroid, normal) as mathutils Vectors.
        The normal is oriented towards the camera (positive Z side).
    
    Algorithm:
        1. Compute centroid of points
        2. Center the points
        3. Compute SVD
        4. The normal is the singular vector with smallest singular value
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")
    
    # Compute centroid
    centroid = points.mean(axis=0)
    
    # Center the points
    centered = points - centroid
    
    # SVD
    u, s, vh = np.linalg.svd(centered)
    
    # The normal is the last row of vh (smallest singular value)
    normal = vh[-1]
    
    # Ensure normal points "up" (positive Z)
    if normal[2] < 0:
        normal = -normal
    
    return Vector(centroid), Vector(normal)


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float = 0.05,
    iterations: int = 100,
    min_inliers_ratio: float = 0.5,
) -> Tuple[Vector, Vector, np.ndarray]:
    """
    Robust plane fitting with RANSAC for noisy point clouds.
    
    More robust than SVD when there are outliers (e.g., objects
    sitting on the floor that shouldn't be included in the plane).
    
    Args:
        points: Nx3 numpy array of point positions.
        threshold: Maximum distance from plane to be considered inlier.
        iterations: Number of RANSAC iterations.
        min_inliers_ratio: Minimum ratio of inliers for valid plane.
    
    Returns:
        Tuple of (centroid, normal, inlier_mask).
        inlier_mask is a boolean array indicating which points are inliers.
    
    Algorithm:
        1. Randomly sample 3 points
        2. Fit plane through those 3 points
        3. Count inliers (points within threshold of plane)
        4. Repeat, keep best result
        5. Refit using all inliers
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")
    
    best_inliers = None
    best_inlier_count = 0
    
    n_points = len(points)
    
    for _ in range(iterations):
        # Random sample of 3 points
        idx = np.random.choice(n_points, 3, replace=False)
        sample = points[idx]
        
        # Fit plane through 3 points
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        
        # Skip degenerate case
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        
        normal = normal / norm_len
        d = -np.dot(normal, sample[0])
        
        # Calculate distances to plane
        distances = np.abs(np.dot(points, normal) + d)
        
        # Count inliers
        inliers = distances < threshold
        inlier_count = np.sum(inliers)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers
    
    if best_inliers is None or best_inlier_count < min_inliers_ratio * n_points:
        # Fall back to SVD if RANSAC fails
        centroid, normal = fit_plane_svd(points)
        return centroid, normal, np.ones(n_points, dtype=bool)
    
    # Refit using all inliers
    inlier_points = points[best_inliers]
    centroid, normal = fit_plane_svd(inlier_points)
    
    return centroid, normal, best_inliers


def compute_ground_alignment_matrix(
    centroid: Vector,
    normal: Vector,
) -> Matrix:
    """
    Compute a 4x4 transformation matrix that aligns a plane to the world XY plane.
    
    After applying this matrix:
    - The plane's centroid is at world origin (0, 0, 0)
    - The plane's normal aligns with world Z-up (0, 0, 1)
    
    Args:
        centroid: The plane's centroid.
        normal: The plane's normal vector (should be normalized).
    
    Returns:
        4x4 transformation Matrix to apply to the scene.
    
    Usage:
        After computing this matrix, apply it to:
        - The camera
        - The point cloud parent Empty
        - Any other scene objects
    """
    # Target normal (world Z-up)
    target_normal = Vector((0.0, 0.0, 1.0))
    
    # Compute rotation to align normal to Z-up
    normal = normal.normalized()
    
    # Handle edge case: normal is already aligned
    dot = normal.dot(target_normal)
    
    if dot > 0.9999:
        # Already aligned, just translate
        rotation_matrix = Matrix.Identity(3)
    elif dot < -0.9999:
        # Opposite direction, rotate 180° around X
        rotation_matrix = Matrix.Rotation(np.pi, 3, 'X').to_3x3()
    else:
        # General case: use cross product for rotation axis
        axis = normal.cross(target_normal)
        axis.normalize()
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        rotation_matrix = Matrix.Rotation(angle, 3, axis).to_3x3()
    
    # Build 4x4 matrix
    matrix = rotation_matrix.to_4x4()
    
    # Rotate the centroid and use as translation
    rotated_centroid = rotation_matrix @ centroid
    matrix.translation = -rotated_centroid
    
    return matrix
