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

    def test_frame_offset_bugfixes(self):
        """Test that frame index conversion bugfixes are correct when frame_start != 1."""
        self.mock_clip.frame_start = 101
        self.mock_clip.frame_duration = 100
        
        # 1. Test CoverageAnalyzer
        from autosolve.tracker.analyzers import CoverageAnalyzer
        analyzer = CoverageAnalyzer(clip_frame_start=101, clip_frame_end=200, segment_size=50)
        
        mock_track = MagicMock()
        mock_marker = MagicMock()
        mock_marker.frame = 10  # clip-relative frame 10 (which is scene frame 110)
        mock_marker.mute = False
        mock_marker.co.x = 0.5
        mock_marker.co.y = 0.5

        mock_marker2 = MagicMock()
        mock_marker2.frame = 90  # clip-relative frame 90 (which is scene frame 190)
        mock_marker2.mute = False
        mock_marker2.co.x = 0.5
        mock_marker2.co.y = 0.5

        mock_track.markers = [mock_marker, mock_marker2]
        
        mock_tracking = MagicMock()
        mock_tracking.tracks = [mock_track]
        
        analyzer.analyze_tracking(mock_tracking, min_lifespan=1)
        # Check that segment (101, 151) has track count 1
        segment = (101, 151)
        self.assertEqual(analyzer.coverage['center'][segment].track_count, 1)

        # 2. Test SmartTracker.verify_full_timeline_coverage
        tracker = SmartTracker(self.mock_clip)
        tracker.tracking.tracks = [mock_track]
        
        coverage_res = tracker.verify_full_timeline_coverage()
        # Since track spans scene frames 110 to 190, earliest_track_start is 110 (instead of 10)
        self.assertEqual(coverage_res['earliest_track_start'], 110)
        self.assertEqual(coverage_res['latest_track_end'], 190)

        # 3. Test SmartTracker._mute_tracks_in_bad_frames
        # bad_frames list contains scene-relative frame numbers
        # If scene frame 110 is bad, then clip marker on frame 10 should be muted
        mock_marker.mute = False
        tracker._mute_tracks_in_bad_frames([110])
        self.assertTrue(mock_marker.mute)

        # 4. Test SmartTracker.select_optimal_keyframes
        # Mock camera object
        camera = MagicMock()
        tracker.clip.tracking.camera = camera
        # Mock frame tracks for select_optimal_keyframes
        # We need at least 8 tracks at each keyframe to be considered valid
        mock_tracks = []
        for i in range(8):
            t = MagicMock()
            m_a = MagicMock()
            m_a.frame = 10
            m_a.mute = False
            m_a.co.x = 0.1 * i
            m_a.co.y = 0.1 * i
            m_b = MagicMock()
            m_b.frame = 90
            m_b.mute = False
            m_b.co.x = 0.1 * i + 0.05
            m_b.co.y = 0.1 * i + 0.05
            t.markers = [m_a, m_b]
            t.name = f"track_{i}"
            mock_tracks.append(t)
        
        tracker.tracking.tracks = mock_tracks
        tracker.select_optimal_keyframes()
        # Camera keyframes should be assigned scene-relative values: 110 and 190
        self.assertEqual(camera.keyframe_a, 110)
        self.assertEqual(camera.keyframe_b, 190)

        # 5. Test TrackAverager with clip
        from autosolve.tracker.averaging import TrackAverager
        averager = TrackAverager(proximity_threshold=0.05, clip=self.mock_clip)
        
        # find_track_clusters calls find_frame(frame)
        # Mock tracks to return position at clip frame 10
        # If scene frame 110 is the current scene frame
        mock_bpy.context.scene.frame_current = 110
        
        # Mock tracks with markers at clip frame 10
        avg_tracks = []
        for i in range(2):
            t = MagicMock()
            m = MagicMock()
            m.frame = 10
            m.mute = False
            m.co.x = 0.5
            m.co.y = 0.5
            # Mock find_frame to return marker when queried with clip_frame (10)
            t.markers.find_frame = MagicMock(side_effect=lambda f: m if f == 10 else None)
            t.name = f"avg_{i}"
            avg_tracks.append(t)
            
        mock_tracking.tracks = avg_tracks
        clusters = averager.find_track_clusters(mock_tracking)
        self.assertEqual(len(clusters), 1)
        self.assertIn("avg_0", clusters[0])
        self.assertIn("avg_1", clusters[0])

        # 6. Test SmartTracker._identify_weak_regions_at_frame
        # If scene frame 110 is queried, it should convert to clip frame 10
        tracker.tracking.tracks = avg_tracks
        weak = tracker._identify_weak_regions_at_frame(110)
        # The tracks are at center (0.5, 0.5), so center should NOT be weak, other regions should be weak
        self.assertNotIn('center', weak)
        self.assertIn('top-left', weak)


if __name__ == "__main__":
    unittest.main()
