# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Validation mixin for SmartTracker.

Contains methods for validating tracking data before and during tracking,
as well as pre-solve validation and confidence estimation.
"""

import math
from typing import Dict, List, Tuple

import bpy


class ValidationMixin:
    """Mixin providing validation methods for SmartTracker."""
    
    def validate_pre_tracking(self) -> Tuple[bool, List[str]]:
        """
        Validate conditions before tracking begins.
        
        Checks:
        - Clip has sufficient frames
        - Markers have valid initial positions
        - No NaN/Inf values in existing markers
        
        Returns:
            Tuple of (is_valid, list of issues found)
        """
        issues = []
        
        # Check clip duration
        if self.clip.frame_duration < 10:
            issues.append(f"Clip too short: {self.clip.frame_duration} frames (need 10+)")
        
        # Check existing markers for NaN/Inf
        nan_tracks = []
        for track in self.tracking.tracks:
            for marker in track.markers:
                if marker.mute:
                    continue
                # Check for invalid coordinates
                if math.isnan(marker.co.x) or math.isnan(marker.co.y):
                    nan_tracks.append(track.name)
                    break
                if math.isinf(marker.co.x) or math.isinf(marker.co.y):
                    nan_tracks.append(track.name)
                    break
                # Check bounds (normalized 0-1)
                if not (0 <= marker.co.x <= 1) or not (0 <= marker.co.y <= 1):
                    nan_tracks.append(track.name)
                    break
        
        if nan_tracks:
            issues.append(f"Invalid marker data in {len(nan_tracks)} tracks: {', '.join(nan_tracks[:5])}")
        
        is_valid = len(issues) == 0
        if is_valid:
            print("AutoSolve: Pre-tracking validation passed")
        else:
            print(f"AutoSolve: Pre-tracking validation failed: {'; '.join(issues)}")
        
        return is_valid, issues
    
    def validate_track_quality(self, frame: int) -> Dict:
        """
        Validate track quality at a specific frame during tracking.
        
        Checks for:
        - Out-of-bounds markers
        - Velocity spikes (sudden jumps)
        - Tracks that should be muted
        
        Returns:
            Dict with validation results and tracks to mute
        """
        result = {
            'frame': frame,
            'active_tracks': 0,
            'out_of_bounds': [],
            'velocity_spikes': [],
            'tracks_to_mute': [],
        }
        
        for track in self.tracking.tracks:
            marker = track.markers.find_frame(frame)
            if not marker or marker.mute:
                continue
            
            result['active_tracks'] += 1
            
            # Check bounds
            if not (0 <= marker.co.x <= 1) or not (0 <= marker.co.y <= 1):
                result['out_of_bounds'].append(track.name)
                result['tracks_to_mute'].append(track.name)
                continue
            
            # Check for NaN
            if math.isnan(marker.co.x) or math.isnan(marker.co.y):
                result['tracks_to_mute'].append(track.name)
                continue
            
            # Check velocity spike (compare to previous frame)
            prev_marker = track.markers.find_frame(frame - 1)
            if prev_marker and not prev_marker.mute:
                dx = abs(marker.co.x - prev_marker.co.x)
                dy = abs(marker.co.y - prev_marker.co.y)
                displacement = (dx**2 + dy**2) ** 0.5
                
                # If displacement > 10% of frame in one step, likely a spike
                if displacement > 0.1:
                    result['velocity_spikes'].append(track.name)
                    result['tracks_to_mute'].append(track.name)
        
        # Mute problematic tracks
        for track in self.tracking.tracks:
            if track.name in result['tracks_to_mute']:
                marker = track.markers.find_frame(frame)
                if marker:
                    marker.mute = True
        
        if result['tracks_to_mute']:
            print(f"AutoSolve: Frame {frame} - Muted {len(result['tracks_to_mute'])} bad tracks")
        
        return result
    
    def validate_pre_solve(self) -> Tuple[bool, List[str]]:
        """
        Validate track data before camera solve.
        
        Ensures:
        - Sufficient valid tracks
        - No NaN/Inf in marker data
        - Adequate track coverage
        - Minimum lifespan requirements
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Count valid tracks
        valid_tracks = 0
        total_markers = 0
        nan_count = 0
        short_tracks = 0
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            # Check for NaN/Inf
            has_nan = False
            for m in markers:
                if math.isnan(m.co.x) or math.isnan(m.co.y):
                    has_nan = True
                    nan_count += 1
                    break
                if math.isinf(m.co.x) or math.isinf(m.co.y):
                    has_nan = True
                    nan_count += 1
                    break
            
            if has_nan:
                continue
            
            # Check lifespan
            markers_sorted = sorted(markers, key=lambda m: m.frame)
            lifespan = markers_sorted[-1].frame - markers_sorted[0].frame
            if lifespan < 5:
                short_tracks += 1
                continue
            
            valid_tracks += 1
            total_markers += len(markers)
        
        # Validation checks
        if valid_tracks < self.ABSOLUTE_MIN_TRACKS:
            issues.append(f"Too few valid tracks: {valid_tracks} (need {self.ABSOLUTE_MIN_TRACKS}+)")
        
        if nan_count > 0:
            issues.append(f"Found {nan_count} tracks with NaN/Inf values")
        
        if short_tracks > valid_tracks:
            issues.append(f"Too many short-lived tracks: {short_tracks}")
        
        # Check frame coverage
        frame_coverage = {}
        for track in self.tracking.tracks:
            for marker in track.markers:
                if not marker.mute:
                    frame_coverage[marker.frame] = frame_coverage.get(marker.frame, 0) + 1
        
        if frame_coverage:
            avg_tracks_per_frame = sum(frame_coverage.values()) / len(frame_coverage)
            if avg_tracks_per_frame < 8:
                issues.append(f"Low average track coverage: {avg_tracks_per_frame:.1f} tracks/frame")
        
        is_valid = len(issues) == 0
        if is_valid:
            print(f"AutoSolve: Pre-solve validation passed ({valid_tracks} valid tracks)")
        else:
            print(f"AutoSolve: Pre-solve validation failed: {'; '.join(issues)}")
        
        return is_valid, issues
    
    def compute_pre_solve_confidence(self) -> Dict:
        """
        Estimate solve quality before running the solver.
        
        Uses track features to predict likelihood of successful solve.
        Records result for ML training correlation with actual solve error.
        
        Returns:
            Dict with confidence, parallax_score, distribution_score, warnings
        """
        confidence = 1.0
        warnings = []
        
        # 1. Check track count
        track_count = len(self.tracking.tracks)
        if track_count < 15:
            confidence *= 0.6
            warnings.append(f"Low track count ({track_count})")
        elif track_count < 25:
            confidence *= 0.85
        
        # 2. Compute parallax score from motion vectors
        motion_vectors = []
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            markers.sort(key=lambda x: x.frame)
            dx = markers[-1].co.x - markers[0].co.x
            dy = markers[-1].co.y - markers[0].co.y
            motion_vectors.append((dx, dy))
        
        parallax_score = 0.0
        if len(motion_vectors) >= 3:
            # Compute variance in motion directions
            avg_dx = sum(v[0] for v in motion_vectors) / len(motion_vectors)
            avg_dy = sum(v[1] for v in motion_vectors) / len(motion_vectors)
            
            variance = sum(
                ((v[0] - avg_dx)**2 + (v[1] - avg_dy)**2) 
                for v in motion_vectors
            ) / len(motion_vectors)
            
            avg_magnitude = (avg_dx**2 + avg_dy**2)**0.5
            if avg_magnitude > 0.0001:
                parallax_score = min(1.0, (variance**0.5) / avg_magnitude)
        
        if parallax_score < 0.05:
            confidence *= 0.5
            warnings.append("Very low parallax - consider tripod mode")
        elif parallax_score < 0.15:
            confidence *= 0.75
            warnings.append("Low parallax - perspective solve may struggle")
        
        # 3. Check track distribution (coverage of 9 regions)
        regions_covered = set()
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if markers:
                avg_x = sum(m.co.x for m in markers) / len(markers)
                avg_y = sum(m.co.y for m in markers) / len(markers)
                col = 0 if avg_x < 0.33 else (1 if avg_x < 0.66 else 2)
                row = 2 if avg_y < 0.33 else (1 if avg_y < 0.66 else 0)
                regions_covered.add((row, col))
        
        distribution_score = len(regions_covered) / 9.0
        
        if distribution_score < 0.5:
            confidence *= 0.7
            warnings.append(f"Poor track distribution ({len(regions_covered)}/9 regions)")
        elif distribution_score < 0.7:
            confidence *= 0.9
        
        # 4. Check track lifespan consistency
        lifespans = []
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) >= 2:
                markers.sort(key=lambda x: x.frame)
                lifespans.append(markers[-1].frame - markers[0].frame)
        
        if lifespans:
            avg_lifespan = sum(lifespans) / len(lifespans)
            if avg_lifespan < 30:
                confidence *= 0.8
                warnings.append(f"Short average track lifespan ({avg_lifespan:.0f} frames)")
        
        result = {
            'confidence': round(confidence, 3),
            'parallax_score': round(parallax_score, 3),
            'track_distribution_score': round(distribution_score, 3),
            'track_count': track_count,
            'warnings': warnings,
        }
        
        # Log and record for ML training
        print(f"AutoSolve: Pre-solve confidence: {confidence:.0%} "
              f"(parallax: {parallax_score:.2f}, distribution: {distribution_score:.2f})")
        if warnings:
            print(f"  Warnings: {', '.join(warnings)}")
        
        # Record to session for correlation with actual solve error
        if hasattr(self, 'recorder') and self.recorder and self.recorder.current_session:
            self.recorder.record_pre_solve_confidence(result)
        
        return result
    
    def sanitize_tracks_before_solve(self) -> int:
        """
        Actively clean up problematic tracks before camera solve.
        
        This method REMOVES tracks that would cause Ceres solver errors:
        - Tracks with NaN/Inf values
        - Tracks with out-of-bounds markers
        - Tracks with too few markers
        - Tracks with impossible velocity spikes
        
        Returns:
            Number of tracks removed
        """
        tracks_to_remove = []
        
        for track in self.tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            
            # Remove tracks with too few markers
            if len(markers) < 3:
                tracks_to_remove.append(track.name)
                continue
            
            # Check for bad data
            is_bad = False
            prev_pos = None
            
            for marker in markers:
                x, y = marker.co.x, marker.co.y
                
                # Check NaN/Inf
                if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                    is_bad = True
                    break
                
                # Check out of bounds (with margin)
                if x < -0.1 or x > 1.1 or y < -0.1 or y > 1.1:
                    is_bad = True
                    break
                
                # Check velocity spike
                if prev_pos is not None:
                    dx = abs(x - prev_pos[0])
                    dy = abs(y - prev_pos[1])
                    if dx > 0.2 or dy > 0.2:  # 20% of frame in one step
                        is_bad = True
                        break
                
                prev_pos = (x, y)
            
            if is_bad:
                tracks_to_remove.append(track.name)
        
        # Remove bad tracks
        if tracks_to_remove:
            for track in self.tracking.tracks:
                track.select = track.name in tracks_to_remove
            
            try:
                self._run_ops(bpy.ops.clip.delete_track)
                print(f"AutoSolve: Sanitized {len(tracks_to_remove)} bad tracks before solve")
            except:
                pass
        
        return len(tracks_to_remove)
