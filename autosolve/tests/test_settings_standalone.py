# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Standalone unit tests for SettingsManager.

Run with: python autosolve/tests/test_settings_standalone.py
"""

import unittest
import sys
import os

# Add parent directory to path for direct import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from the solver module (no package __init__)
from solver.settings_manager import (
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
        print("✅ test_default_values passed")
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = TrackerSettings(pattern_size=21, search_size=101)
        d = settings.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d['pattern_size'], 21)
        self.assertEqual(d['search_size'], 101)
        print("✅ test_to_dict passed")
    
    def test_with_overrides(self):
        """Test creating settings with overrides."""
        base = TrackerSettings()
        overridden = base.with_overrides(correlation=0.55, motion_model='Affine')
        
        # Original unchanged
        self.assertEqual(base.correlation, 0.70)
        
        # New has overrides
        self.assertEqual(overridden.correlation, 0.55)
        self.assertEqual(overridden.motion_model, 'Affine')
        print("✅ test_with_overrides passed")


class TestSettingsManager(unittest.TestCase):
    """Tests for SettingsManager class."""
    
    def test_initialization(self):
        """Test settings manager initialization."""
        manager = SettingsManager('HD_30fps', 'INDOOR')
        self.assertEqual(manager.footage_class, 'HD_30fps')
        self.assertEqual(manager.footage_type, 'INDOOR')
        print("✅ test_initialization passed")
    
    def test_base_settings_loading(self):
        """Test base settings are loaded correctly."""
        manager = SettingsManager('4K_24fps', 'AUTO')
        settings = manager.get_settings()
        
        expected = RESOLUTION_FPS_DEFAULTS['4K_24fps']
        self.assertEqual(settings.pattern_size, expected.pattern_size)
        print("✅ test_base_settings_loading passed")
    
    def test_motion_class_override(self):
        """Test motion class overrides base."""
        manager = SettingsManager('HD_30fps', 'INDOOR')
        high = manager.get_settings(motion_class='HIGH')
        
        expected_high = MOTION_CLASS_SETTINGS['HIGH']
        self.assertEqual(high.motion_model, 'Affine')
        print("✅ test_motion_class_override passed")
    
    def test_tier_settings(self):
        """Test tier-based settings retrieval."""
        manager = SettingsManager('HD_30fps', 'AUTO')
        
        aggressive = manager.get_tier_settings('aggressive')
        selective = manager.get_tier_settings('selective')
        
        self.assertLess(aggressive.correlation, selective.correlation)
        print("✅ test_tier_settings passed")
    
    def test_tier_for_success_rate(self):
        """Test tier selection based on success rate."""
        manager = SettingsManager('HD_30fps', 'AUTO')
        
        self.assertEqual(manager.get_tier_for_success_rate(0.10), 'ultra_aggressive')
        self.assertEqual(manager.get_tier_for_success_rate(0.85), 'selective')
        print("✅ test_tier_for_success_rate passed")
    
    def test_learned_overrides(self):
        """Test learned settings override."""
        manager = SettingsManager('HD_30fps', 'AUTO')
        manager.update_learned({'correlation': 0.80, 'threshold': 0.40})
        
        settings = manager.get_settings()
        self.assertEqual(settings.correlation, 0.80)
        print("✅ test_learned_overrides passed")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("SETTINGS MANAGER UNIT TESTS")
    print("="*60 + "\n")
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTrackerSettings))
    suite.addTests(loader.loadTestsFromTestCase(TestSettingsManager))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
    print("="*60)
