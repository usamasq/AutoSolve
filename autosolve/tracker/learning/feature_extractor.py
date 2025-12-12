# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
FeatureExtractor - Visual feature extraction for ML training.

Extracts content-aware features from video clips for learning:
1. Feature density per region (using Blender's detect_features)
2. Motion classification (LOW/MEDIUM/HIGH)
3. Edge density proxy (from track survival rates)
"""

import hashlib
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

try:
    import bpy
except ImportError:
    bpy = None  # For testing outside Blender


@dataclass
class VisualFeatures:
    """Container for extracted visual features."""
    
    # Clip fingerprint (for per-clip learning)
    clip_fingerprint: str = ""
    
    # Motion classification
    motion_class: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    motion_magnitude: float = 0.0
    motion_variance: float = 0.0
    
    # Feature density per region (count of detected features)
    # Uses Blender's detect_features operator for actual feature detection
    feature_density: Dict[str, int] = field(default_factory=dict)
    feature_density_total: int = 0
    feature_density_mean: float = 0.0
    
    # Multi-frame feature density (for detecting temporal changes)
    # Samples at 25%, 50%, 75% of clip duration
    feature_density_timeline: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Example: {"frame_25pct": {"top-left": 45, ...}, "frame_50pct": {...}, ...}
    
    # Edge density per region (0-1, higher = more texture) - proxy from track survival
    edge_density: Dict[str, float] = field(default_factory=dict)
    edge_density_mean: float = 0.0
    
    # Contrast statistics per region
    contrast_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Temporal motion profile (motion magnitude at evenly-spaced frames)
    temporal_motion_profile: List[float] = field(default_factory=list)
    
    # Optical flow histogram (binned by direction, 8 bins)
    flow_direction_histogram: List[float] = field(default_factory=lambda: [0.0] * 8)
    flow_magnitude_histogram: List[float] = field(default_factory=lambda: [0.0] * 5)
    
    # Failure analysis
    failure_frames: List[int] = field(default_factory=list)
    failure_positions: List[List[float]] = field(default_factory=list)  # [[x, y], ...]


class FeatureExtractor:
    """
    Extracts visual features from video clips for ML training.
    
    Uses only CPU operations (no external ML libraries required).
    """
    
    # Region definitions (same as REGIONS in utils.py)
    REGIONS = [
        'top-left', 'top-center', 'top-right',
        'mid-left', 'center', 'mid-right',
        'bottom-left', 'bottom-center', 'bottom-right'
    ]
    
    # Motion classification thresholds (normalized velocity)
    MOTION_LOW_THRESHOLD = 0.005  # Below this = LOW motion
    MOTION_HIGH_THRESHOLD = 0.02  # Above this = HIGH motion
    
    def __init__(self, clip: Optional['bpy.types.MovieClip'] = None):
        self.clip = clip
        self.features = VisualFeatures()
    
    def extract_all(self, clip: Optional['bpy.types.MovieClip'] = None, 
                    tracking_data: Optional[Dict] = None,
                    defer_thumbnails: bool = False) -> VisualFeatures:
        """
        Extract all visual features from clip.
        
        Args:
            clip: Blender MovieClip (optional, uses self.clip if not provided)
            tracking_data: Optional dict with pre-computed tracking analysis
            defer_thumbnails: Deprecated parameter, kept for backward compatibility
            
        Returns:
            VisualFeatures dataclass with all extracted features
        """
        if clip:
            self.clip = clip
        
        if not self.clip:
            return self.features
        
        # 1. Generate clip fingerprint
        self.features.clip_fingerprint = self._generate_fingerprint()
        
        # 2. Extract motion features (uses tracking if available)
        if tracking_data:
            self._extract_motion_from_tracking(tracking_data)
            
            # Check if SmartTracker already computed feature density during detection
            pre_detected_density = tracking_data.get('detected_feature_density')
            if pre_detected_density:
                # Use pre-computed density (avoid duplicate detect_features call)
                # Note: This is from a single frame (detection frame), not multi-frame sampling
                self.features.feature_density = pre_detected_density
                self.features.feature_density_total = sum(pre_detected_density.values())
                if len(pre_detected_density) > 0:
                    self.features.feature_density_mean = self.features.feature_density_total / len(pre_detected_density)
                # Store in timeline format for consistency
                self.features.feature_density_timeline = {"detection_frame": pre_detected_density}
                print(f"AutoSolve: Using pre-detected feature density - {self.features.feature_density_total} features")
            else:
                # 3. Compute feature density per region using Blender's detect_features
                self._compute_feature_density()
        else:
            # No tracking data - compute from scratch
            self._compute_feature_density()
        
        # 4. Compute edge density per region (proxy from track survival)
        self._compute_edge_density()
        
        return self.features
    
    def _compute_feature_density(self):
        """
        Compute feature density per region using Blender's detect_features operator.
        
        Samples at 3 frames (25%, 50%, 75% of clip duration) to detect:
        - Temporal changes (moving objects, transient occlusions)
        - Consistent dead zones (regions that are always low)
        """
        if not self.clip or not bpy:
            return
        
        try:
            scene = bpy.context.scene
            original_frame = scene.frame_current
            
            # Calculate sample frames
            duration = self.clip.frame_duration
            start = self.clip.frame_start
            sample_frames = {
                "frame_25pct": start + int(duration * 0.25),
                "frame_50pct": start + int(duration * 0.50),
                "frame_75pct": start + int(duration * 0.75)
            }
            
            # Store settings to restore later
            settings = self.clip.tracking.settings
            original_margin = settings.default_margin
            
            # Handle API change: default_correlation_min (new) vs default_minimum_correlation (old)
            correlation_attr = 'default_correlation_min' if hasattr(settings, 'default_correlation_min') else 'default_minimum_correlation'
            original_threshold = getattr(settings, correlation_attr, 0.75)
            
            # Use low threshold to find ALL features
            settings.default_margin = 8
            setattr(settings, correlation_attr, 0.3)
            
            timeline = {}
            all_counts = {region: [] for region in self.REGIONS}
            
            for frame_label, frame_num in sample_frames.items():
                try:
                    scene.frame_set(frame_num)
                    
                    # Get existing tracks to know what's new
                    tracks = self.clip.tracking.tracks
                    existing_track_names = [t.name for t in tracks]
                    
                    # Detect features at this frame
                    bpy.ops.clip.detect_features(threshold=0.1, min_distance=8, margin=8, placement='FRAME')
                    
                    # Count per region
                    region_counts = {region: 0 for region in self.REGIONS}
                    new_tracks = [t for t in self.clip.tracking.tracks if t.name not in existing_track_names]
                    
                    for track in new_tracks:
                        marker = track.markers.find_frame(frame_num)
                        if marker and not marker.mute:
                            x, y = marker.co
                            region = self._get_region(x, y)
                            region_counts[region] += 1
                    
                    # Clean up detected tracks
                    for track in new_tracks:
                        tracks.remove(track)
                    
                    # Store for this frame
                    timeline[frame_label] = region_counts
                    
                    # Accumulate for averaging
                    for region, count in region_counts.items():
                        all_counts[region].append(count)
                        
                except Exception as e:
                    print(f"AutoSolve: Feature density at {frame_label} failed: {e}")
                    timeline[frame_label] = {region: 0 for region in self.REGIONS}
            
            # Restore settings
            settings.default_margin = original_margin
            setattr(settings, correlation_attr, original_threshold)
            scene.frame_set(original_frame)
            
            # Store multi-frame timeline
            self.features.feature_density_timeline = timeline
            
            # Compute average across all frames for the main feature_density
            avg_counts = {}
            for region in self.REGIONS:
                counts = all_counts[region]
                avg_counts[region] = int(sum(counts) / len(counts)) if counts else 0
            
            self.features.feature_density = avg_counts
            self.features.feature_density_total = sum(avg_counts.values())
            self.features.feature_density_mean = self.features.feature_density_total / len(avg_counts) if avg_counts else 0
            
            print(f"AutoSolve: Feature density - {self.features.feature_density_total} avg features (3-frame sampling)")
            
        except Exception as e:
            print(f"AutoSolve: Feature density computation failed: {e}")
            self.features.feature_density = {region: 0 for region in self.REGIONS}
    
    def _get_region(self, x: float, y: float) -> str:
        """Get region name for normalized coordinates."""
        # Divide frame into 3x3 grid
        col = 0 if x < 0.33 else (1 if x < 0.66 else 2)
        row = 2 if y < 0.33 else (1 if y < 0.66 else 0)  # Flip Y
        
        region_map = [
            ['top-left', 'top-center', 'top-right'],
            ['mid-left', 'center', 'mid-right'],
            ['bottom-left', 'bottom-center', 'bottom-right']
        ]
        return region_map[row][col]
    
    def _generate_fingerprint(self) -> str:
        """
        Generate a unique fingerprint for this clip.
        
        Combines filepath hash with resolution/fps/duration for collision resistance.
        Used to identify "same clip" across sessions.
        """
        if not self.clip:
            return ""
        
        try:
            # Access a property to trigger ReferenceError if clip was deleted
            _ = self.clip.name
            
            # Get absolute filepath
            filepath = bpy.path.abspath(self.clip.filepath) if self.clip.filepath else ""
            
            # Combine with clip properties for robustness
            fingerprint_data = f"{filepath}:{self.clip.size[0]}:{self.clip.size[1]}:{self.clip.fps}:{self.clip.frame_duration}"
            
            # Generate hash
            return hashlib.md5(fingerprint_data.encode()).hexdigest()[:16]
        except (ReferenceError, AttributeError) as e:
            print(f"AutoSolve: Error generating fingerprint (clip may have been deleted): {e}")
            return ""
        except Exception as e:
            print(f"AutoSolve: Error generating fingerprint: {e}")
            return ""
    
    def _extract_motion_from_tracking(self, tracking_data: Dict):
        """
        Extract motion classification from tracking analysis.
        
        Uses velocity statistics from tracked markers to classify motion.
        Handles both dict format (new: {avg, max}) and list format (legacy).
        """
        if not tracking_data:
            return
        
        velocities_data = tracking_data.get('velocities')
        
        # Edge case: None or missing velocities
        if velocities_data is None:
            return
        
        mean_vel = 0.0  # Default for classification
        
        # Handle dict format (new) vs list format (legacy)
        if isinstance(velocities_data, dict):
            # Edge case: empty dict
            if not velocities_data:
                return
            # New format: {'avg': float, 'max': float}
            mean_vel = float(velocities_data.get('avg', 0.0) or 0.0)
            max_vel = float(velocities_data.get('max', 0.0) or 0.0)
            self.features.motion_magnitude = mean_vel
            # Use max-avg as proxy for variance
            self.features.motion_variance = max_vel - mean_vel if max_vel > mean_vel else 0.0
        elif isinstance(velocities_data, list):
            # Edge case: empty list
            if not velocities_data:
                return
            # Filter out None/invalid values
            valid_velocities = [v for v in velocities_data if isinstance(v, (int, float)) and v is not None]
            if not valid_velocities:
                return
            # Legacy format: [float, float, ...]
            mean_vel = sum(valid_velocities) / len(valid_velocities)
            variance = sum((v - mean_vel) ** 2 for v in valid_velocities) / len(valid_velocities)
            self.features.motion_magnitude = mean_vel
            self.features.motion_variance = variance ** 0.5
        else:
            # Unexpected type - log and skip
            print(f"AutoSolve: Unexpected velocities type: {type(velocities_data)}")
            return
        
        # Classify motion based on mean velocity
        if mean_vel < self.MOTION_LOW_THRESHOLD:
            self.features.motion_class = "LOW"
        elif mean_vel > self.MOTION_HIGH_THRESHOLD:
            self.features.motion_class = "HIGH"
        else:
            self.features.motion_class = "MEDIUM"
        
        # Build temporal motion profile from frame samples
        frame_samples = tracking_data.get('frame_samples', [])
        if frame_samples:
            self.features.temporal_motion_profile = [
                s.get('avg_velocity', 0.0) for s in frame_samples[:10] if isinstance(s, dict)
            ]

    
    # NOTE: Thumbnail extraction methods removed - using feature_density instead
    # which uses Blender's detect_features for actual trackable feature detection
    
    def _compute_edge_density(self):
        """
        Compute edge density per region.
        
        Uses track survival rates and jitter as proxies for texture quality.
        High edge density = good texture for tracking.
        
        This is a proxy method - full implementation would use Sobel filters
        on actual pixel data, but that's expensive without GPU.
        """
        # Initialize with neutral values
        for region in self.REGIONS:
            self.features.edge_density[region] = 0.5
        
        # If we have tracking data, use track quality as proxy for texture
        # Regions where tracks survive longer = better texture
        if hasattr(self, '_tracking_data') and self._tracking_data:
            region_success = self._tracking_data.get('region_success', {})
            
            densities = []
            for region in self.REGIONS:
                if region in region_success:
                    stats = region_success[region]
                    total = stats.get('total', 0)
                    success = stats.get('success', 0)
                    if total > 0:
                        # Success rate as proxy for texture quality
                        density = success / total
                        self.features.edge_density[region] = round(density, 3)
                        densities.append(density)
            
            if densities:
                self.features.edge_density_mean = round(sum(densities) / len(densities), 3)
        else:
            self.features.edge_density_mean = 0.5
    
    def compute_from_tracking(self, tracking_data: Dict):
        """
        Compute visual features from tracking analysis data.
        
        Call this after motion probe to populate edge density and contrast
        based on track performance.
        
        Args:
            tracking_data: Dict with motion_class, region_success, velocities, etc.
        """
        self._tracking_data = tracking_data
        
        # Update motion features
        self._extract_motion_from_tracking(tracking_data)
        
        # Update edge density using tracking as proxy
        self._compute_edge_density()
        
        # Compute contrast stats from velocity variance
        self._compute_contrast_stats(tracking_data)
    
    def _compute_contrast_stats(self, tracking_data: Dict):
        """
        Compute contrast statistics per region.
        
        Uses velocity variance as proxy for contrast -
        low contrast regions tend to have more jittery/failed tracks.
        """
        region_success = tracking_data.get('region_success', {})
        velocities = tracking_data.get('velocities', {})
        
        for region in self.REGIONS:
            stats = {}
            
            if region in region_success:
                rs = region_success[region]
                total = rs.get('total', 0)
                success = rs.get('success', 0)
                
                # Estimated contrast: success rate correlates with trackable texture
                stats['estimated_contrast'] = round(success / max(total, 1), 3)
                stats['track_count'] = total
                stats['success_rate'] = round(success / max(total, 1), 3)
            else:
                stats['estimated_contrast'] = 0.5
                stats['track_count'] = 0
                stats['success_rate'] = 0.0
            
            self.features.contrast_stats[region] = stats
    
    def record_track_failure(self, track_name: str, frame: int, x: float, y: float):
        """
        Record when and where a track failed.
        
        Args:
            track_name: Name of the failed track
            frame: Frame number where track lost lock
            x, y: Normalized position of track at failure
        """
        self.features.failure_frames.append(frame)
        self.features.failure_positions.append([round(x, 4), round(y, 4)])
    
    def compute_flow_histograms(self, motion_vectors: List[Tuple[float, float]]):
        """
        Compute optical flow histograms from motion vectors.
        
        Args:
            motion_vectors: List of (dx, dy) tuples
        """
        if not motion_vectors:
            return
        
        # Direction histogram (8 bins = 8 directions)
        direction_bins = [0] * 8
        # Magnitude histogram (5 bins: very_slow, slow, medium, fast, very_fast)
        magnitude_bins = [0] * 5
        magnitude_thresholds = [0.003, 0.01, 0.025, 0.05]
        
        for dx, dy in motion_vectors:
            magnitude = (dx**2 + dy**2) ** 0.5
            
            # Direction bin (use atan2, convert to 0-7)
            angle = math.atan2(dy, dx)  # -pi to pi
            bin_idx = int((angle + math.pi) / (2 * math.pi / 8)) % 8
            direction_bins[bin_idx] += 1
            
            # Magnitude bin
            mag_bin = 0
            for i, threshold in enumerate(magnitude_thresholds):
                if magnitude > threshold:
                    mag_bin = i + 1
            magnitude_bins[mag_bin] += 1
        
        # Normalize to fractions
        total = len(motion_vectors)
        self.features.flow_direction_histogram = [b / total for b in direction_bins]
        self.features.flow_magnitude_histogram = [b / total for b in magnitude_bins]
    
    def get_motion_subclass(self) -> str:
        """
        Get motion-based sub-classification string.
        
        Returns string like "LOW_MOTION" or "HIGH_MOTION" to append to footage class.
        """
        return f"{self.features.motion_class}_MOTION"
    
    def to_dict(self) -> Dict:
        """Convert features to dictionary for JSON serialization."""
        return asdict(self.features)


def create_enhanced_footage_class(base_class: str, motion_class: str) -> str:
    """
    Create enhanced footage class with motion sub-classification.
    
    Args:
        base_class: Base footage class like "HD_30fps"
        motion_class: Motion classification like "LOW", "MEDIUM", "HIGH"
        
    Returns:
        Enhanced class like "HD_30fps_HIGH_MOTION"
    """
    return f"{base_class}_{motion_class}_MOTION"
