# SPDX-FileCopyrightText: 2024-2025 Your Name
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Compositor setup for lens undistortion.

Creates compositor nodes to undistort the footage when lens distortion
parameters (k1, k2) are present in the solve result.

This is critical for accurate compositing - without undistortion,
3D objects will "drift" at the edges of the frame.
"""

import bpy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..solver.pipeline import ReconstructionResult


def setup_undistort_compositor(
    context: bpy.types.Context,
    clip: bpy.types.MovieClip,
    k1: float,
    k2: float,
) -> None:
    """
    Set up compositor nodes for lens undistortion.
    
    Creates a node tree:
        [Movie Clip] → [Movie Distort (Undistort)] → [Composite]
    
    This makes the footage match the linear 3D camera projection.
    
    Args:
        context: Blender context.
        clip: The Movie Clip with tracking data.
        k1: First radial distortion coefficient.
        k2: Second radial distortion coefficient.
    
    Note:
        This requires the clip to have camera tracking data with
        lens settings configured. If not present, we'll set them up.
    """
    scene = context.scene
    
    # Enable compositing
    scene.use_nodes = True
    
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    
    # Check if already set up
    for node in nodes:
        if node.type == 'MOVIEDISTORTION' and node.clip == clip:
            return  # Already configured
    
    # Clear default nodes (keep Composite output)
    composite_node = None
    for node in list(nodes):
        if node.type == 'COMPOSITE':
            composite_node = node
        elif node.type in ('R_LAYERS', 'IMAGE'):
            # Keep render layers if present
            pass
        else:
            pass  # Keep existing nodes
    
    if composite_node is None:
        composite_node = nodes.new('CompositorNodeComposite')
        composite_node.location = (600, 300)
    
    # Create Movie Clip node
    clip_node = nodes.new('CompositorNodeMovieClip')
    clip_node.clip = clip
    clip_node.location = (0, 300)
    clip_node.label = f"Source: {clip.name}"
    
    # Create Movie Distortion node
    distort_node = nodes.new('CompositorNodeMovieDistortion')
    distort_node.clip = clip
    distort_node.distortion_type = 'UNDISTORT'
    distort_node.location = (300, 300)
    distort_node.label = "Lens Undistort"
    
    # Link nodes
    links.new(clip_node.outputs['Image'], distort_node.inputs['Image'])
    links.new(distort_node.outputs['Image'], composite_node.inputs['Image'])
    
    # Set up the clip's tracking lens data
    _configure_clip_lens(clip, k1, k2)


def _configure_clip_lens(
    clip: bpy.types.MovieClip,
    k1: float,
    k2: float,
) -> None:
    """
    Configure the Movie Clip's lens distortion settings.
    
    These settings are used by the Movie Distortion node.
    
    Args:
        clip: The Movie Clip to configure.
        k1: First radial distortion coefficient.
        k2: Second radial distortion coefficient.
    """
    tracking = clip.tracking
    camera = tracking.camera
    
    # Set distortion model
    camera.distortion_model = 'POLYNOMIAL'
    
    # Set coefficients
    camera.k1 = k1
    camera.k2 = k2
    camera.k3 = 0.0  # Not provided by pycolmap OPENCV model
    
    # Ensure principal point is centered (can be refined later)
    camera.principal_point = (0.5, 0.5)


def has_significant_distortion(k1: float, k2: float, threshold: float = 0.001) -> bool:
    """
    Check if the distortion is significant enough to warrant correction.
    
    Small distortion values can be ignored without visible artifacts.
    
    Args:
        k1: First radial distortion coefficient.
        k2: Second radial distortion coefficient.
        threshold: Minimum value to consider significant.
    
    Returns:
        True if distortion should be corrected.
    """
    return abs(k1) > threshold or abs(k2) > threshold


def setup_distortion_from_result(
    context: bpy.types.Context,
    clip: bpy.types.MovieClip,
    result: 'ReconstructionResult',
) -> bool:
    """
    Set up distortion correction from a solve result.
    
    This is the main entry point for distortion handling.
    Only sets up nodes if distortion is significant.
    
    Args:
        context: Blender context.
        clip: The Movie Clip.
        result: The reconstruction result.
    
    Returns:
        True if distortion was set up, False if skipped.
    """
    # Get distortion from first camera (all frames share the same lens)
    if not result.cameras:
        return False
    
    cam = result.cameras[0]
    k1, k2 = cam.k1, cam.k2
    
    if not has_significant_distortion(k1, k2):
        return False
    
    setup_undistort_compositor(context, clip, k1, k2)
    return True
