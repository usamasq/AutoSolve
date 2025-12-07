# SPDX-FileCopyrightText: 2024-2025 Your Name
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Camera setup module.

Handles:
- Setting up Movie Clip as camera background image
- Configuring render settings to match clip (resolution, FPS, frame range)
- Creating camera animation from solve result
"""

import bpy
from mathutils import Matrix, Quaternion
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..solver.pipeline import ReconstructionResult, CameraFrame


def setup_camera_background(
    cam_obj: bpy.types.Object,
    clip: bpy.types.MovieClip,
) -> None:
    """
    Set up the Movie Clip as a camera background image.
    
    This gives users the classic "matchmove" viewport:
    - Video plays behind the 3D scene
    - Easy to verify object placement
    
    Args:
        cam_obj: The camera object to add background to.
        clip: The Movie Clip to use as background.
    """
    cam_data = cam_obj.data
    
    # Enable background images
    cam_data.show_background_images = True
    
    # Check if already has this clip as background
    for bg in cam_data.background_images:
        if bg.source == 'MOVIE_CLIP' and bg.clip == clip:
            return  # Already set up
    
    # Add the movie clip as background
    bg = cam_data.background_images.new()
    bg.source = 'MOVIE_CLIP'
    bg.clip = clip
    bg.alpha = 1.0
    bg.display_depth = 'BACK'
    bg.frame_method = 'FIT'


def configure_render_settings(
    context: bpy.types.Context,
    clip: bpy.types.MovieClip,
    cam_obj: bpy.types.Object,
) -> None:
    """
    Match render settings to the source footage.
    
    Configures:
    - Resolution
    - Frame rate
    - Frame range
    - Active camera
    
    Args:
        context: Blender context.
        clip: The source Movie Clip.
        cam_obj: The solved camera to set as active.
    """
    scene = context.scene
    
    # Resolution from clip
    if clip.size[0] > 0 and clip.size[1] > 0:
        scene.render.resolution_x = clip.size[0]
        scene.render.resolution_y = clip.size[1]
        scene.render.resolution_percentage = 100
    
    # Frame range
    scene.frame_start = clip.frame_start
    scene.frame_end = clip.frame_start + clip.frame_duration - 1
    
    # FPS
    fps = clip.fps
    if fps > 0:
        if fps == int(fps):
            scene.render.fps = int(fps)
            scene.render.fps_base = 1.0
        else:
            # Handle fractional FPS (e.g., 23.976, 29.97)
            # Common fractional rates
            if abs(fps - 23.976) < 0.01:
                scene.render.fps = 24000
                scene.render.fps_base = 1001
            elif abs(fps - 29.97) < 0.01:
                scene.render.fps = 30000
                scene.render.fps_base = 1001
            elif abs(fps - 59.94) < 0.01:
                scene.render.fps = 60000
                scene.render.fps_base = 1001
            else:
                # Generic fractional handling
                scene.render.fps = round(fps * 1000)
                scene.render.fps_base = 1000
    
    # Set active camera
    scene.camera = cam_obj


def create_camera_animation(
    cam_obj: bpy.types.Object,
    result: 'ReconstructionResult',
    clip: bpy.types.MovieClip,
) -> None:
    """
    Apply solved camera animation to the camera object.
    
    Creates keyframes for:
    - Location (translation)
    - Rotation
    - Focal length
    
    Args:
        cam_obj: The camera object to animate.
        result: The reconstruction result with camera poses.
        clip: The source clip (for frame offset).
    """
    cam_data = cam_obj.data
    
    # Set up camera data
    # Use clip dimensions for sensor calculation
    if clip.size[0] > 0:
        cam_data.sensor_fit = 'HORIZONTAL'
        cam_data.sensor_width = 36.0  # Standard full-frame width
    
    # Clear existing animation
    if cam_obj.animation_data:
        cam_obj.animation_data_clear()
    
    # Ensure action exists
    if not cam_obj.animation_data:
        cam_obj.animation_data_create()
    
    action = bpy.data.actions.new(name=f"{cam_obj.name}_Action")
    cam_obj.animation_data.action = action
    
    # Create keyframes for each frame
    for cam_frame in result.cameras:
        frame = cam_frame.frame
        
        # Convert rotation matrix to quaternion
        rotation_matrix = Matrix(cam_frame.rotation.tolist()).to_3x3()
        quaternion = rotation_matrix.to_quaternion()
        
        # Set location
        cam_obj.location = cam_frame.translation.tolist()
        cam_obj.keyframe_insert(data_path="location", frame=frame)
        
        # Set rotation
        cam_obj.rotation_mode = 'QUATERNION'
        cam_obj.rotation_quaternion = quaternion
        cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        
        # Set focal length
        # Convert from pixels to mm using sensor size
        if clip.size[0] > 0:
            focal_mm = (cam_frame.focal_length / clip.size[0]) * cam_data.sensor_width
            cam_data.lens = focal_mm
            cam_data.keyframe_insert(data_path="lens", frame=frame)


def create_camera_from_result(
    context: bpy.types.Context,
    result: 'ReconstructionResult',
    clip: bpy.types.MovieClip,
    parent: bpy.types.Object = None,
) -> bpy.types.Object:
    """
    Create a fully configured camera from solve result.
    
    This is the main entry point that:
    1. Creates camera object
    2. Sets up background image
    3. Configures render settings
    4. Creates animation
    
    Args:
        context: Blender context.
        result: The reconstruction result.
        clip: The source Movie Clip.
        parent: Optional parent object for hierarchy.
    
    Returns:
        The created camera object.
    """
    # Create camera data
    cam_name = f"Camera_{clip.name}"
    cam_data = bpy.data.cameras.new(cam_name)
    
    # Create camera object
    cam_obj = bpy.data.objects.new(cam_name, cam_data)
    context.collection.objects.link(cam_obj)
    
    # Set parent if provided
    if parent:
        cam_obj.parent = parent
    
    # Setup
    setup_camera_background(cam_obj, clip)
    configure_render_settings(context, clip, cam_obj)
    create_camera_animation(cam_obj, result, clip)
    
    return cam_obj
