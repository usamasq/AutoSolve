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

        def _deferred_python_detect():
            try:
                from .worker.client import detect_system_python, check_dependencies
                path = detect_system_python()
                if path:
                    for scene in bpy.data.scenes:
                        settings = getattr(scene, 'autosolve', None)
                        if settings and not settings.external_python_path:
                            settings.external_python_path = path
                            if check_dependencies(path):
                                settings.installer_state = 'SUCCESS'
                                settings.installer_progress = "AI Packages are already installed."
                            else:
                                settings.installer_state = 'IDLE'
                                settings.installer_progress = "Packages missing. Click Install below."
            except Exception:
                pass
            return None

        if hasattr(bpy.app, "timers"):
            bpy.app.timers.register(_deferred_python_detect, first_interval=2.0)


    def unregister():
        """Unregister all classes from submodules in reverse order."""
        for module in reversed(_modules):
            module.unregister()
else:
    def register():
        pass

    def unregister():
        pass
