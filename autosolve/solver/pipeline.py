# SPDX-FileCopyrightText: 2024-2025 Your Name
# SPDX-License-Identifier: GPL-3.0-or-later

"""
SfM Pipeline module.

Wraps pycolmap to provide a simple interface for Structure-from-Motion
reconstruction. This module runs the heavy computation and should be
called from a background thread.

⚠️ THREAD SAFETY: This module should run in a BACKGROUND THREAD.
   It does NOT access any bpy.* APIs.
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, List
import numpy as np

# Add user site-packages to path for Windows Store Blender installations
# where pip installs to user directory but Blender doesn't check there
_user_site = Path(os.environ.get('APPDATA', '')) / 'Python' / f'Python{sys.version_info.major}{sys.version_info.minor}' / 'site-packages'
if _user_site.exists() and str(_user_site) not in sys.path:
    sys.path.insert(0, str(_user_site))


@dataclass
class CameraFrame:
    """Camera pose for a single frame."""
    frame: int
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3D translation vector
    focal_length: float
    principal_point: tuple[float, float]
    
    # Distortion parameters (only if use_distortion=True)
    k1: float = 0.0
    k2: float = 0.0


@dataclass
class ReconstructionResult:
    """Result of the SfM reconstruction."""
    
    # Camera poses per frame
    cameras: List[CameraFrame] = field(default_factory=list)
    
    # 3D point cloud (Nx3)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    
    # Point colors (Nx3, RGB 0-255)
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.uint8))
    
    # Statistics
    reprojection_error: float = 0.0
    num_registered_images: int = 0
    
    # Scale factor applied during normalization
    scale_factor: float = 1.0


class SolveError(Exception):
    """Base exception for solve errors."""
    pass


class NotEnoughFeaturesError(SolveError):
    """Raised when not enough features are detected."""
    pass


class NotEnoughMotionError(SolveError):
    """Raised when there's not enough camera motion to solve."""
    pass


class MatchingFailedError(SolveError):
    """Raised when feature matching fails."""
    pass


def run_reconstruction(
    image_dir: Path,
    output_dir: Path,
    tripod_mode: bool = False,
    use_distortion: bool = True,
    quality_preset: str = 'BALANCED',
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> ReconstructionResult:
    """
    Run the full SfM reconstruction pipeline.
    
    Args:
        image_dir: Directory containing extracted frames (JPEG).
        output_dir: Directory to write reconstruction output.
        tripod_mode: If True, use rotation-only model (for nodal pan).
        use_distortion: If True, estimate lens distortion (k1, k2).
        quality_preset: 'FAST' (2k), 'BALANCED' (4k), or 'QUALITY' (8k) features.
        progress_callback: Optional callback(status_message, progress_0_to_1).
    
    Returns:
        ReconstructionResult with camera poses and point cloud.
    
    Raises:
        SolveError: If reconstruction fails.
        ImportError: If pycolmap is not installed.
    
    Note:
        ⚠️ This function should be called from a BACKGROUND THREAD.
        It performs heavy computation and will block for several minutes.
    """
    try:
        import pycolmap
    except ImportError as e:
        raise ImportError(
            "pycolmap is not installed. Please install it with: pip install pycolmap"
        ) from e
    
    # Ensure directories exist
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    database_path = output_dir / "database.db"
    
    # Remove old database if exists
    if database_path.exists():
        database_path.unlink()
    
    # ═══════════════════════════════════════════════════════════
    # STEP 1: Feature Extraction
    # ═══════════════════════════════════════════════════════════
    if progress_callback:
        progress_callback("Extracting features", 0.1)
    
    # Choose camera model based on distortion setting
    camera_model = "OPENCV" if use_distortion else "SIMPLE_PINHOLE"
    
    # Try to run feature extraction with appropriate API
    extraction_success = False
    extraction_error = None
    
    # Attempt 1: pycolmap 3.13+ with FeatureExtractionOptions
    if hasattr(pycolmap, 'FeatureExtractionOptions'):
        try:
            ext_options = pycolmap.FeatureExtractionOptions()
            # Map quality preset to feature count
            feature_counts = {'FAST': 2048, 'BALANCED': 4096, 'QUALITY': 8192}
            max_features = feature_counts.get(quality_preset, 4096)
            
            # Configure SIFT options if available as a sub-property
            if hasattr(ext_options, 'sift'):
                ext_options.sift.max_num_features = max_features
                ext_options.sift.upright = True
            
            pycolmap.extract_features(
                database_path=str(database_path),
                image_path=str(image_dir),
                camera_model=camera_model,
                extraction_options=ext_options,
            )
            extraction_success = True
        except Exception as e:
            extraction_error = e
    
    # Attempt 2: pycolmap 3.13+ with defaults only (skip custom options)
    if not extraction_success:
        try:
            pycolmap.extract_features(
                database_path=str(database_path),
                image_path=str(image_dir),
                camera_model=camera_model,
            )
            extraction_success = True
        except Exception as e:
            extraction_error = e
    
    # Attempt 3: Older pycolmap with sift_options argument
    if not extraction_success and hasattr(pycolmap, 'SiftExtractionOptions'):
        try:
            sift_opts = pycolmap.SiftExtractionOptions()
            sift_opts.max_num_features = 8192
            sift_opts.upright = True
            
            pycolmap.extract_features(
                database_path=str(database_path),
                image_path=str(image_dir),
                camera_model=camera_model,
                sift_options=sift_opts,
            )
            extraction_success = True
        except Exception as e:
            extraction_error = e
    
    if not extraction_success:
        raise NotEnoughFeaturesError(f"Feature extraction failed: {extraction_error}") from extraction_error
    
    # ═══════════════════════════════════════════════════════════
    # STEP 2: Feature Matching
    # ═══════════════════════════════════════════════════════════
    if progress_callback:
        progress_callback("Matching features", 0.3)
    
    # Note: pycolmap 3.13 SiftMatchingOptions doesn't have guided_matching
    # Use exhaustive matching - slower but more reliable for finding initial pairs
    # Sequential matching was causing "no initial pair" failures
    try:
        pycolmap.match_exhaustive(database_path=str(database_path))
    except Exception as e:
        raise MatchingFailedError(f"Feature matching failed: {e}") from e
    
    # ═══════════════════════════════════════════════════════════
    # STEP 3: Incremental Mapping (SfM)
    # ═══════════════════════════════════════════════════════════
    if progress_callback:
        progress_callback("Solving camera motion", 0.5)
    
    try:
        # Configure pipeline options (pycolmap 3.13+)
        # Note: In older pycolmap it was IncrementalMapperOptions
        if hasattr(pycolmap, 'IncrementalPipelineOptions'):
            pipeline_options = pycolmap.IncrementalPipelineOptions()
            
            # Make the mapper more lenient for video/drone footage
            if hasattr(pipeline_options, 'mapper'):
                # Lower thresholds for finding initial pair
                if hasattr(pipeline_options.mapper, 'init_num_trials'):
                    pipeline_options.mapper.init_num_trials = 500  # More attempts
                if hasattr(pipeline_options.mapper, 'init_min_num_inliers'):
                    pipeline_options.mapper.init_min_num_inliers = 50  # Lower requirement
                if hasattr(pipeline_options.mapper, 'abs_pose_min_num_inliers'):
                    pipeline_options.mapper.abs_pose_min_num_inliers = 15  # Lower for registration
                if hasattr(pipeline_options.mapper, 'min_num_matches'):
                    pipeline_options.mapper.min_num_matches = 10  # Lower threshold
        else:
            # Fallback for older pycolmap
            pipeline_options = pycolmap.IncrementalMapperOptions()
        
        # Run incremental mapping
        maps = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_dir),
            output_path=str(output_dir),
            options=pipeline_options,
        )
        
    except Exception as e:
        raise SolveError(f"Incremental mapping failed: {e}") from e
    
    # Handle return type (dict in pycolmap 3.13+, list in older versions)
    if isinstance(maps, dict):
        if not maps:
            raise NotEnoughMotionError(
                "Could not reconstruct camera motion. "
                "Ensure the footage has enough parallax and distinct features."
            )
        # Get the first/largest reconstruction from the dict
        reconstruction = next(iter(maps.values()))
    else:
        if not maps:
            raise NotEnoughMotionError(
                "Could not reconstruct camera motion. "
                "Ensure the footage has enough parallax and distinct features."
            )
        reconstruction = maps[0]
    
    # ═══════════════════════════════════════════════════════════
    # STEP 4: Post-processing
    # ═══════════════════════════════════════════════════════════
    if progress_callback:
        progress_callback("Processing results...", 0.8)
    
    # Normalize scale to fit within 10m bounding cube
    scale_factor = _normalize_reconstruction(reconstruction)
    
    # Convert to our result format
    result = _parse_reconstruction(reconstruction, scale_factor)
    
    if progress_callback:
        progress_callback("Complete", 1.0)
    
    return result


def _normalize_reconstruction(reconstruction) -> float:
    """
    Normalize the reconstruction to fit within a ~10m bounding cube.
    
    This prevents viewport clipping issues and ensures reasonable scale.
    
    Args:
        reconstruction: pycolmap.Reconstruction object.
    
    Returns:
        The scale factor that was applied.
    """
    import pycolmap
    
    if len(reconstruction.points3D) == 0:
        return 1.0
    
    # Get all point positions
    points = np.array([p.xyz for p in reconstruction.points3D.values()])
    
    # Calculate bounding box
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    max_dimension = max(bbox_size)
    
    if max_dimension < 1e-6:
        return 1.0
    
    # Target: fit within 10 meter cube
    TARGET_SIZE = 10.0
    scale_factor = TARGET_SIZE / max_dimension
    
    # Apply scale using pycolmap's transform
    # Sim3d(scale, rotation, translation)
    transform = pycolmap.Sim3d(scale_factor, np.eye(3), np.zeros(3))
    reconstruction.transform(transform)
    
    return scale_factor


def _parse_reconstruction(reconstruction, scale_factor: float) -> ReconstructionResult:
    """
    Convert pycolmap Reconstruction to our ReconstructionResult format.
    
    Also handles coordinate system conversion from COLMAP to Blender:
    - COLMAP: Y-down, Z-forward
    - Blender: Z-up, Y-forward
    """
    result = ReconstructionResult()
    result.scale_factor = scale_factor
    
    # Coordinate transform: COLMAP → Blender
    # Swap Y and Z, negate the new Z
    COLMAP_TO_BLENDER = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0],
    ])
    
    # Parse cameras
    for image_id, image in reconstruction.images.items():
        camera = reconstruction.cameras[image.camera_id]
        
        # Get rotation and translation (handle pycolmap 3.13 API)
        # pycolmap 3.13 uses cam_from_world (Rigid3d), older used rotmat()/tvec
        if hasattr(image, 'cam_from_world'):
            # pycolmap 3.13+ API
            pose = image.cam_from_world
            R_w2c = pose.rotation.matrix()
            t_w2c = pose.translation
        elif hasattr(image, 'rotmat'):
            # Older pycolmap API
            R_w2c = image.rotmat()
            t_w2c = image.tvec
        else:
            # Fallback: try qvec/tvec
            from scipy.spatial.transform import Rotation
            R_w2c = Rotation.from_quat([image.qvec[1], image.qvec[2], image.qvec[3], image.qvec[0]]).as_matrix()
            t_w2c = image.tvec
        
        # Inverse to get Camera-to-World (Position and Orientation)
        R_c2w = R_w2c.T
        center_c2w = -R_c2w @ t_w2c
        
        # 1. Transform Position from COLMAP World to Blender World
        #    COLMAP World: Y-down, Z-forward
        #    Blender World: Z-up, Y-forward
        #    Transformation: Swap Y/Z, Negate new Z (effectively X, -Z, Y)
        translation = COLMAP_TO_BLENDER @ center_c2w
        
        # 2. Transform Orientation
        #    We need the rotation of the Blender Camera object.
        #    Blender Camera local axes: X-Right, Y-Up, Z-Back (looking down -Z)
        #    COLMAP Camera local axes:  X-Right, Y-Down, Z-Forward (looking down +Z)
        #    
        #    Axis adjustment matrix (COLMAP cam -> Blender cam):
        #    X -> X
        #    Y -> -Y (Down -> Up)
        #    Z -> -Z (Fwd -> Back)
        CAM_AXES_ADJUST = np.diag([1, -1, -1])
        
        #    Total rotation: World_Convert @ R_c2w @ Axes_Adjust
        rotation = COLMAP_TO_BLENDER @ R_c2w @ CAM_AXES_ADJUST
        
        # Get intrinsics
        focal = camera.focal_length
        cx, cy = camera.principal_point_x, camera.principal_point_y
        
        # Get distortion if available
        k1, k2 = 0.0, 0.0
        if hasattr(camera, 'params') and len(camera.params) >= 6:
            # OPENCV model: fx, fy, cx, cy, k1, k2, p1, p2
            k1 = camera.params[4] if len(camera.params) > 4 else 0.0
            k2 = camera.params[5] if len(camera.params) > 5 else 0.0
        
        # Extract frame number from filename
        # Assumes format: frame_000001.jpg
        try:
            frame_num = int(Path(image.name).stem.split('_')[-1])
        except (ValueError, IndexError):
            frame_num = image_id
        
        result.cameras.append(CameraFrame(
            frame=frame_num,
            rotation=rotation,
            translation=translation,
            focal_length=focal,
            principal_point=(cx, cy),
            k1=k1,
            k2=k2,
        ))
    
    # Sort cameras by frame number
    result.cameras.sort(key=lambda c: c.frame)
    
    # Parse 3D points
    if len(reconstruction.points3D) > 0:
        points = []
        colors = []
        
        for point3d in reconstruction.points3D.values():
            # Transform position to Blender coordinates
            pos = COLMAP_TO_BLENDER @ point3d.xyz
            points.append(pos)
            colors.append(point3d.color)
        
        result.points = np.array(points)
        result.colors = np.array(colors, dtype=np.uint8)
    
    # Statistics
    result.reprojection_error = reconstruction.compute_mean_reprojection_error()
    result.num_registered_images = len(reconstruction.images)
    
    return result
