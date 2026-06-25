# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
StrategicMixin for SmartTracker.

Manages optimal start frame calculation, coverage gap filling, strategic tracking
iterations, and timeline coverage verification (detecting start/end gaps).
"""

import bpy
from typing import Dict, List, Optional


class StrategicMixin:
    """Mixin for SmartTracker strategic timeline tracking operations."""

    def get_optimal_start_frame(self) -> int:
        """
        Get the optimal frame to start detection/tracking from.
        """
        frame_start = self.clip.frame_start
        frame_end = frame_start + self.clip.frame_duration - 1
        
        if self.clip.frame_duration < 60:
            return frame_start
        
        middle_frame = frame_start + (self.clip.frame_duration // 2)
        
        print(f"AutoSolve: Optimal start frame: {middle_frame} "
              f"(range: {frame_start}-{frame_end})")
        
        return middle_frame
    
    def fill_coverage_gaps(self) -> Dict:
        """
        Fill gaps in coverage by adding markers to weak zones.
        """
        result = {
            'markers_added': 0,
            'detection_frames': [],
        }
        
        self.coverage_analyzer.analyze_tracking(self.tracking)
        summary = self.coverage_analyzer.get_coverage_summary()
        
        if summary['is_balanced']:
            print(f"AutoSolve: Coverage is balanced ({summary['regions_with_tracks']}/9 regions)")
            return result
        
        weak_zones = self.coverage_analyzer.get_weak_zones()
        if not weak_zones:
            print("AutoSolve: No weak zones identified")
            return result
        
        processed_regions = set()
        
        for region, segment in weak_zones[:5]:
            if region in processed_regions:
                continue
            
            target_frame = segment[0]
            bpy.context.scene.frame_set(target_frame)
            
            added = self.detect_in_region(region, count=2)
            result['markers_added'] += added
            if added > 0:
                result['detection_frames'].append(target_frame)
            processed_regions.add(region)
            
            print(f"AutoSolve: Added {added} markers to {region} at frame {target_frame}")
        
        return result
    
    def get_coverage_analysis(self) -> Dict:
        """
        Analyze current coverage and return summary.
        """
        self.coverage_analyzer.analyze_tracking(self.tracking)
        return self.coverage_analyzer.get_coverage_summary()
    
    def is_coverage_balanced(self) -> bool:
        """Check if current tracking has balanced coverage."""
        summary = self.get_coverage_analysis()
        return summary['is_balanced']
    
    def strategic_track_iteration(self) -> Dict:
        """
        Perform one iteration of strategic tracking.
        """
        self.strategic_iteration += 1
        print(f"AutoSolve: Strategic iteration {self.strategic_iteration}")
        
        summary = self.get_coverage_analysis()
        
        result = {
            'iteration': self.strategic_iteration,
            'coverage_before': summary.copy(),
            'markers_added': 0,
            'detection_frames': [],
            'coverage_after': None,
        }
        
        if summary['is_balanced']:
            print("AutoSolve: Coverage is balanced, no more iterations needed")
            return result
        
        gap_result = self.fill_coverage_gaps()
        result['markers_added'] = gap_result['markers_added']
        result['detection_frames'] = gap_result['detection_frames']
        
        result['coverage_after'] = self.get_coverage_analysis()
        
        return result
    
    def verify_full_timeline_coverage(self) -> Dict:
        """
        Verify that all surviving tracks cover the full timeline.
        """
        frame_start = self.clip.frame_start
        frame_end = frame_start + self.clip.frame_duration - 1
        
        result = {
            'needs_backward_extension': [],
            'needs_forward_extension': [],
            'earliest_track_start': frame_end,
            'latest_track_end': frame_start,
            'total_tracks': 0,
            'fully_covered_tracks': 0,
            'recommended_action': 'none',
        }
        
        MARGIN = 5
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            result['total_tracks'] += 1
            
            markers_sorted = sorted(markers, key=lambda m: m.frame)
            track_start_clip = markers_sorted[0].frame
            track_end_clip = markers_sorted[-1].frame
            track_start = self.clip_to_scene_frame(track_start_clip)
            track_end = self.clip_to_scene_frame(track_end_clip)
            
            result['earliest_track_start'] = min(result['earliest_track_start'], track_start)
            result['latest_track_end'] = max(result['latest_track_end'], track_end)
            
            needs_backward = track_start > frame_start + MARGIN
            needs_forward = track_end < frame_end - MARGIN
            
            if needs_backward:
                result['needs_backward_extension'].append({
                    'name': track.name,
                    'current_start': track_start,
                    'target_start': frame_start,
                })
            
            if needs_forward:
                result['needs_forward_extension'].append({
                    'name': track.name,
                    'current_end': track_end,
                    'target_end': frame_end,
                })
            
            if not needs_backward and not needs_forward:
                result['fully_covered_tracks'] += 1
        
        if result['needs_backward_extension'] and result['needs_forward_extension']:
            result['recommended_action'] = 'extend_both'
        elif result['needs_backward_extension']:
            result['recommended_action'] = 'extend_backward'
        elif result['needs_forward_extension']:
            result['recommended_action'] = 'extend_forward'
        else:
            result['recommended_action'] = 'none'
        
        coverage_pct = result['fully_covered_tracks'] / max(result['total_tracks'], 1) * 100
        print(f"AutoSolve: Timeline coverage: {result['fully_covered_tracks']}/{result['total_tracks']} tracks "
              f"({coverage_pct:.0f}%) cover full range")
        
        if result['needs_backward_extension']:
            print(f"AutoSolve: {len(result['needs_backward_extension'])} tracks need backward extension")
        if result['needs_forward_extension']:
            print(f"AutoSolve: {len(result['needs_forward_extension'])} tracks need forward extension")
        
        return result
    
    def should_continue_strategic(self) -> bool:
        """
        Determine if more strategic iterations are needed.
        """
        if self.strategic_iteration >= self.MAX_STRATEGIC_ITERATIONS:
            print(f"AutoSolve: Max strategic iterations reached ({self.MAX_STRATEGIC_ITERATIONS})")
            return False
        
        summary = self.get_coverage_analysis()
        return not summary['is_balanced']

    def preserve_existing_tracks(self) -> int:
        """
        Preserve user's existing tracked markers.
        """
        preserved = 0
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 5:
                continue
            
            markers_sorted = sorted(markers, key=lambda m: m.frame)
            lifespan = markers_sorted[-1].frame - markers_sorted[0].frame
            
            if lifespan >= 20:
                if hasattr(track, 'lock'):
                    track.lock = True
                preserved += 1
        
        if preserved > 0:
            print(f"AutoSolve: Preserved {preserved} existing well-tracked markers")
        
        return preserved
    
    def enhance_priority_regions(self) -> int:
        """
        Add more markers to user-defined priority regions.
        """
        priority = self.get_user_priority_regions()
        added = 0
        
        for region in priority['high']:
            count = self.detect_in_region(region, count=4)
            added += count
            print(f"AutoSolve: Priority region {region}: +{count} markers")
        
        for region in priority['existing']:
            self.coverage_analyzer.analyze_tracking(self.tracking)
            for seg, data in self.coverage_analyzer.coverage.get(region, {}).items():
                if data.successful_tracks < 3:
                    count = self.detect_in_region(region, count=2)
                    added += count
                    break
        
        return added
