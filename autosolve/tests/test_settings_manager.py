# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unit tests for Smart Tracker components.

These tests can run without Blender by mocking bpy dependencies.
Run with: python -m pytest autosolve/tests/test_smart_tracker.py -v
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock bpy before importing our modules
sys.modules['bpy'] = Mock()
sys.modules['mathutils'] = Mock()

from autosolve.solver.settings_manager import (
    TrackerSettings,
    SettingsManager,
    RESOLUTION_FPS_DEFAULTS,
    FOOTAGE_TYPE_ADJUSTMENTS,
    MOTION_CLASS_SETTINGS,
    TIER_SETTINGS,
)


class TestTrackerSettings(unittest.TestCase):
    """Tests for TrackerSettings dataclass."""
    
    def test_default_values(self):
        """Test default settings values."""
        settings = TrackerSettings()
        self.assertEqual(settings.pattern_size, 15)
        self.assertEqual(settings.search_size, 71)
        self.assertEqual(settings.correlation, 0.70)
        self.assertEqual(settings.threshold, 0.30)
        self.assertEqual(settings.motion_model, 'LocRot')
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = TrackerSettings(pattern_size=21, search_size=101)
        d = settings.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d['pattern_size'], 21)
        self.assertEqual(d['search_size'], 101)
    
    def test_with_overrides(self):
        """Test creating settings with overrides."""
        base = TrackerSettings()
        overridden = base.with_overrides(correlation=0.55, motion_model='Affine')
        
        # Original unchanged
        self.assertEqual(base.correlation, 0.70)
        self.assertEqual(base.motion_model, 'LocRot')
        
        # New has overrides
        self.assertEqual(overridden.correlation, 0.55)
        self.assertEqual(overridden.motion_model, 'Affine')
        
        # Non-overridden values preserved
        self.assertEqual(overridden.pattern_size, 15)


class TestSettingsManager(unittest.TestCase):
    """Tests for SettingsManager class."""
    
    def test_initialization(self):
        """Test settings manager initialization."""
        manager = SettingsManager('HD_30fps', 'INDOOR')
        self.assertEqual(manager.footage_class, 'HD_30fps')
        self.assertEqual(manager.footage_type, 'INDOOR')
    
    def test_base_settings_loading(self):
        """Test base settings are loaded correctly."""
        manager = SettingsManager('4K_24fps', 'AUTO')
        settings = manager.get_settings()
        
        # Should match the 4K_24fps defaults
        expected = RESOLUTION_FPS_DEFAULTS['4K_24fps']
        self.assertEqual(settings.pattern_size, expected.pattern_size)
        self.assertEqual(settings.search_size, expected.search_size)
    
    def test_unknown_footage_class_fallback(self):
        """Test fallback for unknown footage class."""
        manager = SettingsManager('UNKNOWN_class', 'AUTO')
        settings = manager.get_settings()
        
        # Should fall back to defaults
        self.assertEqual(settings.pattern_size, 15)
    
    def test_motion_class_override(self):
        """Test motion class completely overrides base."""
        manager = SettingsManager('HD_30fps', 'INDOOR')
        
        # Without motion class
        base = manager.get_settings()
        
        # With HIGH motion class
        high = manager.get_settings(motion_class='HIGH')
        
        expected_high = MOTION_CLASS_SETTINGS['HIGH']
        self.assertEqual(high.search_size, expected_high.search_size)
        self.assertEqual(high.motion_model, 'Affine')
    
    def test_tier_settings(self):
        """Test tier-based settings retrieval."""
        manager = SettingsManager('HD_30fps', 'AUTO')
        
        aggressive = manager.get_tier_settings('aggressive')
        selective = manager.get_tier_settings('selective')
        
        # Aggressive should have lower correlation than selective
        self.assertLess(aggressive.correlation, selective.correlation)
        # Aggressive should have larger search
        self.assertGreater(aggressive.search_size, selective.search_size)
    
    def test_tier_for_success_rate(self):
        """Test tier selection based on success rate."""
        manager = SettingsManager('HD_30fps', 'AUTO')
        
        # Very low success rate
        self.assertEqual(manager.get_tier_for_success_rate(0.10), 'ultra_aggressive')
        
        # Low success rate
        self.assertEqual(manager.get_tier_for_success_rate(0.20), 'aggressive')
        
        # Medium success rate
        self.assertEqual(manager.get_tier_for_success_rate(0.35), 'moderate')
        
        # Good success rate
        self.assertEqual(manager.get_tier_for_success_rate(0.60), 'balanced')
        
        # Excellent success rate
        self.assertEqual(manager.get_tier_for_success_rate(0.85), 'selective')
    
    def test_learned_overrides(self):
        """Test learned settings override."""
        manager = SettingsManager('HD_30fps', 'AUTO')
        
        # Add learned overrides
        manager.update_learned({'correlation': 0.80, 'threshold': 0.40})
        
        settings = manager.get_settings()
        self.assertEqual(settings.correlation, 0.80)
        self.assertEqual(settings.threshold, 0.40)


class TestSettingsPresets(unittest.TestCase):
    """Tests for settings presets and adjustments."""
    
    def test_resolution_presets_exist(self):
        """Test all expected resolution presets exist."""
        expected = ['HD_24fps', 'HD_30fps', 'HD_60fps', 
                   '4K_24fps', '4K_30fps', '4K_60fps']
        for preset in expected:
            self.assertIn(preset, RESOLUTION_FPS_DEFAULTS)
    
    def test_footage_type_adjustments_exist(self):
        """Test all expected footage type adjustments exist."""
        expected = ['AUTO', 'INDOOR', 'DRONE', 'ACTION', 'OUTDOOR']
        for preset in expected:
            self.assertIn(preset, FOOTAGE_TYPE_ADJUSTMENTS)
    
    def test_motion_class_settings_exist(self):
        """Test motion class settings exist."""
        for motion_class in ['LOW', 'MEDIUM', 'HIGH']:
            self.assertIn(motion_class, MOTION_CLASS_SETTINGS)
            
            settings = MOTION_CLASS_SETTINGS[motion_class]
            self.assertIsInstance(settings, TrackerSettings)
    
    def test_tier_settings_order(self):
        """Test tier settings have sensible ordering."""
        # ultra_aggressive should have lowest correlation
        ultra = TIER_SETTINGS['ultra_aggressive']
        selective = TIER_SETTINGS['selective']
        
        self.assertLess(ultra.correlation, selective.correlation)
        self.assertGreater(ultra.search_size, selective.search_size)


if __name__ == '__main__':
    unittest.main()
