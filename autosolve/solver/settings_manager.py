# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Centralized Settings Management for Smart Tracker.

This module provides a single source of truth for all tracker settings,
consolidating logic from:
- PRETRAINED_DEFAULTS (resolution/fps based)
- FOOTAGE_TYPE_ADJUSTMENTS (drone, indoor, etc.)
- TIERED_SETTINGS (iteration-based refinement)
- Learned settings from SettingsPredictor
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, List
import json
from pathlib import Path


@dataclass
class TrackerSettings:
    """Immutable tracker settings configuration."""
    pattern_size: int = 15
    search_size: int = 71
    correlation: float = 0.70
    threshold: float = 0.30
    motion_model: str = 'LocRot'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Blender API."""
        return asdict(self)
    
    def with_overrides(self, **kwargs) -> 'TrackerSettings':
        """Create new settings with specific overrides."""
        data = asdict(self)
        data.update(kwargs)
        return TrackerSettings(**data)


# ═══════════════════════════════════════════════════════════════════════════
# PRE-DEFINED SETTINGS TABLES
# ═══════════════════════════════════════════════════════════════════════════

# Base settings by footage resolution and frame rate
RESOLUTION_FPS_DEFAULTS = {
    'HD_24fps': TrackerSettings(pattern_size=17, search_size=91, correlation=0.68, threshold=0.28),
    'HD_30fps': TrackerSettings(pattern_size=15, search_size=71, correlation=0.70, threshold=0.30),
    'HD_60fps': TrackerSettings(pattern_size=13, search_size=61, correlation=0.72, threshold=0.32),
    '4K_24fps': TrackerSettings(pattern_size=21, search_size=111, correlation=0.65, threshold=0.25),
    '4K_30fps': TrackerSettings(pattern_size=19, search_size=101, correlation=0.67, threshold=0.27),
    '4K_60fps': TrackerSettings(pattern_size=17, search_size=91, correlation=0.70, threshold=0.30),
    '8K_24fps': TrackerSettings(pattern_size=25, search_size=141, correlation=0.62, threshold=0.22),
    '8K_30fps': TrackerSettings(pattern_size=23, search_size=121, correlation=0.65, threshold=0.25),
}

# Adjustments by footage type (applied on top of resolution defaults)
FOOTAGE_TYPE_ADJUSTMENTS = {
    'AUTO': {},  # No adjustments
    'INDOOR': {
        'correlation': 0.72,
        'threshold': 0.30,
    },
    'DRONE': {
        'search_size_mult': 1.3,
        'pattern_size_mult': 1.2,
        'correlation': 0.60,
        'threshold': 0.20,
        'motion_model': 'Affine',
    },
    'ACTION': {
        'search_size_mult': 1.4,
        'correlation': 0.55,
        'motion_model': 'Affine',
    },
    'OUTDOOR': {
        'search_size_mult': 1.1,
        'correlation': 0.65,
    },
    'HANDHELD': {
        'search_size_mult': 1.2,
        'correlation': 0.62,
        'motion_model': 'LocRotScale',
    },
    'GIMBAL': {
        'correlation': 0.72,
        'threshold': 0.32,
    },
    'VFX': {
        'pattern_size_mult': 1.3,
        'search_size_mult': 0.9,
        'correlation': 0.75,
        'threshold': 0.35,
    },
}

# Motion class overrides (for dynamic adaptation)
MOTION_CLASS_SETTINGS = {
    'LOW': TrackerSettings(pattern_size=15, search_size=71, correlation=0.72, threshold=0.30, motion_model='Loc'),
    'MEDIUM': TrackerSettings(pattern_size=19, search_size=101, correlation=0.65, threshold=0.25, motion_model='LocRot'),
    'HIGH': TrackerSettings(pattern_size=25, search_size=141, correlation=0.55, threshold=0.20, motion_model='Affine'),
}

# Tiered settings for iterative refinement
TIER_SETTINGS = {
    'ultra_aggressive': TrackerSettings(pattern_size=11, search_size=151, correlation=0.45, threshold=0.15, motion_model='Affine'),
    'aggressive': TrackerSettings(pattern_size=15, search_size=121, correlation=0.55, threshold=0.20, motion_model='Affine'),
    'moderate': TrackerSettings(pattern_size=17, search_size=91, correlation=0.62, threshold=0.25, motion_model='LocRot'),
    'balanced': TrackerSettings(pattern_size=19, search_size=81, correlation=0.68, threshold=0.28, motion_model='LocRot'),
    'selective': TrackerSettings(pattern_size=21, search_size=71, correlation=0.75, threshold=0.32, motion_model='LocRot'),
}


class SettingsManager:
    """
    Single source of truth for tracker settings.
    
    Merges settings from multiple sources in priority order:
    1. Learned settings (highest priority)
    2. Motion class adjustments
    3. Footage type adjustments
    4. Resolution/FPS base settings (lowest)
    """
    
    def __init__(self, footage_class: str, footage_type: str = 'AUTO'):
        """
        Initialize settings manager.
        
        Args:
            footage_class: Resolution/FPS class (e.g., 'HD_30fps', '4K_24fps')
            footage_type: Footage type (e.g., 'DRONE', 'INDOOR', 'AUTO')
        """
        self.footage_class = footage_class
        self.footage_type = footage_type
        self._base_settings = self._load_base_settings()
        self._current_tier = 'balanced'
        self._learned_overrides: Dict = {}
        
    def _load_base_settings(self) -> TrackerSettings:
        """Load base settings from resolution/fps defaults."""
        if self.footage_class in RESOLUTION_FPS_DEFAULTS:
            return RESOLUTION_FPS_DEFAULTS[self.footage_class]
        # Default fallback
        return TrackerSettings()
    
    def _apply_footage_adjustments(self, settings: TrackerSettings) -> TrackerSettings:
        """Apply footage type adjustments to settings."""
        if self.footage_type not in FOOTAGE_TYPE_ADJUSTMENTS:
            return settings
        
        adjustments = FOOTAGE_TYPE_ADJUSTMENTS[self.footage_type]
        if not adjustments:
            return settings
        
        # Convert to dict for manipulation
        data = settings.to_dict()
        
        # Apply multipliers
        if 'search_size_mult' in adjustments:
            data['search_size'] = int(data['search_size'] * adjustments['search_size_mult'])
        if 'pattern_size_mult' in adjustments:
            data['pattern_size'] = int(data['pattern_size'] * adjustments['pattern_size_mult'])
        
        # Apply direct overrides
        for key in ['correlation', 'threshold', 'motion_model']:
            if key in adjustments:
                data[key] = adjustments[key]
        
        return TrackerSettings(**data)
    
    def get_settings(self, motion_class: Optional[str] = None) -> TrackerSettings:
        """
        Get merged settings with all layers applied.
        
        Args:
            motion_class: Optional motion class override ('LOW', 'MEDIUM', 'HIGH')
            
        Returns:
            TrackerSettings with all adjustments applied
        """
        # Start with base settings
        settings = self._base_settings
        
        # Apply footage type adjustments
        settings = self._apply_footage_adjustments(settings)
        
        # Apply motion class if specified
        if motion_class and motion_class in MOTION_CLASS_SETTINGS:
            motion_settings = MOTION_CLASS_SETTINGS[motion_class]
            # Motion class provides complete override
            settings = motion_settings
        
        # Apply any learned overrides
        if self._learned_overrides:
            settings = settings.with_overrides(**self._learned_overrides)
        
        return settings
    
    def get_tier_settings(self, tier: str = None) -> TrackerSettings:
        """
        Get settings for a specific refinement tier.
        
        Args:
            tier: Tier name (ultra_aggressive, aggressive, moderate, balanced, selective)
            
        Returns:
            TrackerSettings for the specified tier
        """
        tier = tier or self._current_tier
        if tier not in TIER_SETTINGS:
            tier = 'balanced'
        return TIER_SETTINGS[tier]
    
    def set_tier(self, tier: str):
        """Set current refinement tier."""
        if tier in TIER_SETTINGS:
            self._current_tier = tier
    
    def update_learned(self, overrides: Dict):
        """Update learned settings overrides."""
        self._learned_overrides.update(overrides)
    
    def get_tier_for_success_rate(self, success_rate: float) -> str:
        """
        Determine appropriate tier based on tracking success rate.
        
        Args:
            success_rate: Track survival rate (0.0 to 1.0)
            
        Returns:
            Tier name to use
        """
        if success_rate < 0.15:
            return 'ultra_aggressive'
        elif success_rate < 0.25:
            return 'aggressive'
        elif success_rate < 0.40:
            return 'moderate'
        elif success_rate < 0.70:
            return 'balanced'
        else:
            return 'selective'
    
    def get_motion_settings_dict(self, motion_class: str) -> Dict:
        """
        Get settings as dictionary for backwards compatibility.
        
        Args:
            motion_class: Motion classification ('LOW', 'MEDIUM', 'HIGH')
            
        Returns:
            Dictionary with tracker settings
        """
        settings = self.get_settings(motion_class)
        return settings.to_dict()
