# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve main package.

Modules:
- properties: Scene properties and settings
- operators: Main tracking operator and learning tools
- ui: N-Panel interface
- tracker: Smart tracking with adaptive learning
"""

try:
    import bpy
    from . import properties
    from . import operators
    from . import ui
    HAS_BPY = True
except ImportError:
    HAS_BPY = False

if HAS_BPY:
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
else:
    def register():
        pass

    def unregister():
        pass
