# SPDX-FileCopyrightText: 2024-2025 Your Name
# SPDX-License-Identifier: GPL-3.0-or-later

"""
GPU-accelerated point cloud renderer.

Uses Blender's gpu module to efficiently render large point clouds
(100k+ points) at 60 FPS in the 3D Viewport.

This is the "temporary" visualization that shows during orientation.
The final scene objects are created by the finalize operator.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
from typing import Optional
from mathutils import Matrix


class PointCloudRenderer:
    """
    Renders point clouds using GPU batches.
    
    Usage:
        renderer = PointCloudRenderer()
        renderer.update_points(positions, colors)
        renderer.enable()
        # ... user interacts with viewport ...
        renderer.disable()
    """
    
    def __init__(self):
        self._batch: Optional[gpu.types.GPUBatch] = None
        self._shader: Optional[gpu.types.GPUShader] = None
        self._draw_handler = None
        self._enabled = False
        
        # Point data
        self._positions: Optional[np.ndarray] = None
        self._colors: Optional[np.ndarray] = None
        
        # Display settings
        self.point_size = 3.0
        self.alpha = 1.0
        
        # Highlight for hover (used by Ground Wand)
        self._highlight_indices: set = set()
        self._highlight_color = (1.0, 1.0, 0.0, 1.0)  # Yellow
    
    def update_points(self, positions: np.ndarray, colors: Optional[np.ndarray] = None):
        """
        Update the point cloud data.
        
        Args:
            positions: Nx3 array of point positions.
            colors: Nx3 array of RGB colors (0-255). If None, uses white.
        """
        self._positions = positions.astype(np.float32)
        
        if colors is not None:
            # Normalize to 0-1 range
            self._colors = (colors.astype(np.float32) / 255.0)
        else:
            # Default white
            self._colors = np.ones((len(positions), 3), dtype=np.float32)
        
        # Rebuild batch
        self._rebuild_batch()
    
    def set_highlight(self, indices: set):
        """
        Set which points should be highlighted (e.g., for Ground Wand hover).
        
        Args:
            indices: Set of point indices to highlight.
        """
        self._highlight_indices = indices
        self._rebuild_batch()
    
    def clear_highlight(self):
        """Remove all highlights."""
        self._highlight_indices = set()
        self._rebuild_batch()
    
    def _rebuild_batch(self):
        """Rebuild the GPU batch with current data."""
        if self._positions is None or len(self._positions) == 0:
            self._batch = None
            return
        
        # Get or create shader
        if self._shader is None:
            self._shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        
        # We'll use a simple approach: draw all points with vertex colors
        # For better performance, use a custom shader (see shader.py)
        
        # Apply highlight colors
        final_colors = self._colors.copy()
        for idx in self._highlight_indices:
            if 0 <= idx < len(final_colors):
                final_colors[idx] = np.array(self._highlight_color[:3])
        
        # Create batch
        # Note: 3D_UNIFORM_COLOR doesn't support per-vertex colors
        # We need to use 3D_FLAT_COLOR or a custom shader
        try:
            # Try Blender 4.0+ 3D shader first
            try:
                self._shader = gpu.shader.from_builtin('3D_FLAT_COLOR')
            except ValueError:
                self._shader = gpu.shader.from_builtin('FLAT_COLOR')
            
            self._batch = batch_for_shader(
                self._shader,
                'POINTS',
                {
                    "pos": self._positions.tolist(),
                    "color": np.hstack([
                        final_colors,
                        np.full((len(final_colors), 1), self.alpha)
                    ]).tolist(),
                },
            )
        except Exception:
            # Fallback to uniform color if FLAT_COLOR not available
            try:
                self._shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
            except ValueError:
                self._shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            self._batch = batch_for_shader(
                self._shader,
                'POINTS',
                {"pos": self._positions.tolist()},
            )
    
    def enable(self):
        """Enable rendering in all 3D Viewports."""
        if self._enabled:
            return
        
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback,
            (),
            'WINDOW',
            'POST_VIEW'
        )
        self._enabled = True
        
        # Force viewport redraw
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    
    def disable(self):
        """Disable rendering."""
        if not self._enabled:
            return
        
        if self._draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(
                self._draw_handler,
                'WINDOW'
            )
            self._draw_handler = None
        
        self._enabled = False
        
        # Force viewport redraw to clear
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    
    def _draw_callback(self):
        """Called each viewport redraw."""
        if self._batch is None or self._shader is None:
            return
        
        # Set point size
        gpu.state.point_size_set(self.point_size)
        
        # Enable blending for alpha
        gpu.state.blend_set('ALPHA')
        
        # Draw
        self._shader.bind()
        
        # For UNIFORM_COLOR shader, set the color
        if hasattr(self._shader, 'uniform_float'):
            try:
                self._shader.uniform_float("color", (1.0, 1.0, 1.0, self.alpha))
            except ValueError:
                pass  # FLAT_COLOR shader doesn't have this uniform
        
        self._batch.draw(self._shader)
        
        # Reset state
        gpu.state.blend_set('NONE')
    
    def cleanup(self):
        """Clean up resources. Call when addon is disabled."""
        self.disable()
        self._batch = None
        self._shader = None
        self._positions = None
        self._colors = None


# Global renderer instance (lazy initialization)
_renderer: Optional[PointCloudRenderer] = None


def get_renderer() -> PointCloudRenderer:
    """Get the global point cloud renderer instance."""
    global _renderer
    if _renderer is None:
        _renderer = PointCloudRenderer()
    return _renderer


def cleanup_renderer():
    """Clean up the global renderer."""
    global _renderer
    if _renderer is not None:
        _renderer.cleanup()
        _renderer = None
