# SPDX-FileCopyrightText: 2024-2025 Your Name
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve - One-click camera tracking for Blender.

This extension provides automated Structure-from-Motion (SfM) camera tracking
using pycolmap. It replaces Blender's manual tracking workflow with a single
"Analyze & Solve" button.
"""

bl_info = {
    "name": "AutoSolve",
    "author": "Your Name",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "Clip Editor > Sidebar > AutoSolve",
    "description": "One-click camera tracking powered by Structure-from-Motion",
    "category": "Motion Tracking",
}


def register():
    """Register all extension modules."""
    from . import autosolve
    autosolve.register()


def unregister():
    """Unregister all extension modules."""
    from . import autosolve
    autosolve.unregister()


if __name__ == "__main__":
    register()
