# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Learning subsystem for EZTrack Smart Tracking.

This module provides data collection, storage, and prediction
capabilities for improving tracking accuracy over time.
"""

from .session_recorder import SessionRecorder
from .settings_predictor import SettingsPredictor

__all__ = ['SessionRecorder', 'SettingsPredictor']
