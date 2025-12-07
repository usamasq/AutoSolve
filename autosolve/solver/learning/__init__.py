# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Learning subsystem for AutoSolve Smart Tracking.

This module provides data collection, storage, prediction,
and failure diagnosis for improving tracking accuracy over time.
"""

from .session_recorder import SessionRecorder
from .settings_predictor import SettingsPredictor
from .failure_diagnostics import FailureDiagnostics, FailurePattern, DiagnosisResult

__all__ = ['SessionRecorder', 'SettingsPredictor', 'FailureDiagnostics', 'FailurePattern', 'DiagnosisResult']

