# SPDX-FileCopyrightText: 2026 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Lightweight compatibility test for refactored SmartTracker mixins.
Mocks Blender's bpy module to run under standard python.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 1. Mock Blender dependencies
mock_bpy = MagicMock()
sys.modules['bpy'] = mock_bpy

mock_mathutils = MagicMock()
sys.modules['mathutils'] = mock_mathutils

IMPORT_SUCCESS = False
IMPORT_ERROR = None
# 2. Import SmartTracker
try:
    from autosolve.tracker.smart_tracker import SmartTracker
    IMPORT_SUCCESS = True
except Exception as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = e


class TestSmartTrackerRefactor(unittest.TestCase):
    def setUp(self):
        self.assertTrue(IMPORT_SUCCESS, f"Failed to import SmartTracker: {IMPORT_ERROR}")
        
        # Setup mock clip
        self.mock_clip = MagicMock()
        self.mock_clip.frame_start = 1
        self.mock_clip.frame_duration = 250
        self.mock_clip.fps = 24.0
        self.mock_clip.size = (1920, 1080)
        self.mock_clip.filepath = "C:/test_clip.mp4"
        self.mock_clip.name = "test_clip"
        
        # Mock tracking and settings
        self.mock_clip.tracking = MagicMock()
        self.mock_clip.tracking.settings = MagicMock()
        
    def test_initialization(self):
        """Test that SmartTracker initializes and resolves configurations correctly."""
        tracker = SmartTracker(self.mock_clip)
        self.assertEqual(tracker.clip, self.mock_clip)
        self.assertEqual(tracker.robust_mode, False)
        self.assertEqual(tracker.footage_type, 'AUTO')
        self.assertEqual(tracker.resolution_class, 'HD_24fps')
        self.assertEqual(tracker.footage_class, 'HD_24fps_AUTO')
        
    def test_mixin_inheritance(self):
        """Verify that methods from different mixins are correctly bound to SmartTracker."""
        tracker = SmartTracker(self.mock_clip)
        
        # Test methods from ProbeCacheMixin
        self.assertTrue(hasattr(tracker, '_get_probe_cache_path'))
        self.assertTrue(hasattr(tracker, '_run_motion_probe'))
        
        # Test methods from DetectionMixin
        self.assertTrue(hasattr(tracker, 'detect_features_smart'))
        self.assertTrue(hasattr(tracker, '_get_feature_placement'))
        
        # Test methods from StrategicMixin
        self.assertTrue(hasattr(tracker, 'get_optimal_start_frame'))
        self.assertTrue(hasattr(tracker, 'verify_full_timeline_coverage'))
        
        # Test methods from LearningMixin
        self.assertTrue(hasattr(tracker, 'adapt_settings_mid_session'))
        self.assertTrue(hasattr(tracker, 'is_in_temporal_dead_zone'))
        
        # Test methods from CleanupMixin
        self.assertTrue(hasattr(tracker, 'heal_tracks'))
        self.assertTrue(hasattr(tracker, 'preserve_good_tracks'))

    def test_frame_conversions(self):
        """Test simple core frame coordinate conversion methods."""
        tracker = SmartTracker(self.mock_clip)
        # clip start is 1, so scene frame 10 should be clip frame 10
        self.assertEqual(tracker.scene_to_clip_frame(10), 10)
        self.assertEqual(tracker.clip_to_scene_frame(10), 10)
        
        # Change clip start to 101
        self.mock_clip.frame_start = 101
        tracker = SmartTracker(self.mock_clip)
        self.assertEqual(tracker.scene_to_clip_frame(110), 10)
        self.assertEqual(tracker.clip_to_scene_frame(10), 110)

    def test_motion_estimation_quick(self):
        """Test quick motion classification fallback values."""
        tracker = SmartTracker(self.mock_clip)
        self.assertEqual(tracker._estimate_motion_quick(), 'HIGH')  # default for 24fps/250f range


if __name__ == "__main__":
    unittest.main()
