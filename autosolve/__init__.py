# SPDX-FileCopyrightText: 2024-2025 Your Name
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve main package.

This module coordinates registration of all submodules:
- properties: Scene properties and settings
- operators: Blender operators (Solve, Ground Wand, Scale)
- ui: Panel definitions
- preferences: Addon preferences
"""

from . import properties
from . import operators
from . import ui

# All classes that need registration
_modules = (
    properties,
    operators,
    ui,
)


def register():
    """Register all classes from submodules."""
    for module in _modules:
        module.register()


def unregister():
    """Unregister all classes from submodules in reverse order."""
    for module in reversed(_modules):
        module.unregister()
