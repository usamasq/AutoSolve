# SPDX-FileCopyrightText: 2024-2025 Your Name
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Frame extractor module.

Extracts frames from a Blender Movie Clip to disk as JPEG images
for processing by pycolmap.

Uses the Video Sequence Editor (VSE) for maximum compatibility
across all Blender versions and video formats.

⚠️ THREAD SAFETY: This module MUST run on the main thread
   because it accesses bpy.data and renders.
"""

import bpy
import os
from pathlib import Path
from typing import Optional, Callable


def extract_frames(
    clip: bpy.types.MovieClip,
    output_dir: Path,
    step: int = 1,
    quality: int = 95,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Extract frames from a Movie Clip to JPEG files.
    
    Uses the VSE (Video Sequence Editor) which is the most reliable
    method for rendering video frames across all Blender versions.
    
    Args:
        clip: The Movie Clip to extract frames from.
        output_dir: Directory to save extracted frames.
        step: Extract every Nth frame (1 = all frames).
        quality: JPEG quality (1-100).
        progress_callback: Optional callback(current, total) for progress.
    
    Returns:
        Number of frames extracted.
    
    Raises:
        ValueError: If clip has no frames.
        OSError: If output directory cannot be created.
    """
    if not clip or clip.frame_duration < 1:
        raise ValueError("Movie Clip has no frames or is invalid")
    
    # Get absolute path to the video file
    video_path = bpy.path.abspath(clip.filepath)
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store original context
    original_scene = bpy.context.scene
    
    # Create temporary scene for VSE rendering
    temp_scene_name = f"_autosolve_vse_extract"
    
    # Remove if exists (from failed previous run)
    if temp_scene_name in bpy.data.scenes:
        bpy.data.scenes.remove(bpy.data.scenes[temp_scene_name])
    
    temp_scene = bpy.data.scenes.new(temp_scene_name)
    
    extracted_count = 0
    
    try:
        # Switch to temp scene
        bpy.context.window.scene = temp_scene
        
        # Configure render settings to match clip size
        temp_scene.render.resolution_x = clip.size[0]
        temp_scene.render.resolution_y = clip.size[1]
        temp_scene.render.resolution_percentage = 100
        
        # Output settings
        temp_scene.render.image_settings.file_format = 'JPEG'
        temp_scene.render.image_settings.quality = quality
        
        # Frame range
        temp_scene.frame_start = 1
        temp_scene.frame_end = clip.frame_duration
        
        # ════════════════════════════════════════════════════════════
        # SET UP VIDEO SEQUENCE EDITOR (VSE)
        # This is the MOST RELIABLE way to render video frames
        # ════════════════════════════════════════════════════════════
        
        # Ensure VSE exists
        if not temp_scene.sequence_editor:
            temp_scene.sequence_editor_create()
        
        vse = temp_scene.sequence_editor
        
        # Add the video as a movie strip
        # Handle Blender 5.0 API change: sequences -> strips
        if hasattr(vse, 'strips'):
            # Blender 5.0+
            movie_strip = vse.strips.new_movie(
                name="source_video",
                filepath=video_path,
                channel=1,
                frame_start=1,
            )
        else:
            # Blender 4.x
            movie_strip = vse.sequences.new_movie(
                name="source_video",
                filepath=video_path,
                channel=1,
                frame_start=1,
            )
        
        # Use VSE rendering mode
        temp_scene.render.use_sequencer = True
        
        print(f"AutoSolve: Extracting frames to {output_dir}")
        print(f"AutoSolve: Video source: {video_path}")
        print(f"AutoSolve: Frame range: 1-{clip.frame_duration}, step={step}")
        
        # Calculate frames to extract
        frames_to_render = list(range(1, clip.frame_duration + 1, step))
        total = len(frames_to_render)
        
        # Render each frame
        for i, frame in enumerate(frames_to_render):
            # Progress callback with frame info
            if progress_callback:
                progress_callback(i + 1, total)
            
            # Set the frame
            temp_scene.frame_set(frame)
            
            # Output path for this frame (use clip's frame numbering for consistency)
            output_frame_num = clip.frame_start + (frame - 1)
            frame_path = output_dir / f"frame_{output_frame_num:06d}.jpg"
            temp_scene.render.filepath = str(frame_path)
            
            # Render the frame from VSE
            bpy.ops.render.render(write_still=True)
            
            # Verify the file was created
            if frame_path.exists() and frame_path.stat().st_size > 0:
                extracted_count += 1
            else:
                print(f"AutoSolve: WARNING - Frame {frame} not saved correctly")
                    
    finally:
        # Cleanup: Restore original scene
        bpy.context.window.scene = original_scene
        
        # Remove temporary scene (and its VSE data)
        if temp_scene_name in bpy.data.scenes:
            bpy.data.scenes.remove(bpy.data.scenes[temp_scene_name])
    
    # Final progress
    if progress_callback:
        progress_callback(total, total)
    
    print(f"AutoSolve: Extracted {extracted_count}/{total} frames successfully")
    
    return extracted_count


def get_clip_as_image_sequence_path(clip: bpy.types.MovieClip) -> Optional[Path]:
    """
    If the clip is already an image sequence, return its directory.
    This allows us to skip extraction entirely.
    
    Args:
        clip: The Movie Clip.
    
    Returns:
        Path to the image sequence directory, or None if not a sequence.
    """
    if clip.source != 'SEQUENCE':
        return None
    
    clip_path = bpy.path.abspath(clip.filepath)
    return Path(clip_path).parent
