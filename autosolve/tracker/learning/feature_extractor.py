# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
FeatureExtractor - CPU-only visual feature extraction for ML training.

Extracts content-aware features from video frames without requiring GPU.
Features are designed for:
1. Immediate statistical model improvement (motion-based sub-classification)
2. Future neural network training (thumbnails, edge maps, histograms)
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
    
    # Frame thumbnails (base64 encoded, 64x64 RGB)
    # List of 3-5 representative frames
    thumbnails: List[str] = field(default_factory=list)
    thumbnail_frames: List[int] = field(default_factory=list)
    
    # Edge density per region (0-1, higher = more texture)
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
                    tracking_data: Optional[Dict] = None) -> VisualFeatures:
        """
        Extract all visual features from clip.
        
        Args:
            clip: Blender MovieClip (optional, uses self.clip if not provided)
            tracking_data: Optional dict with pre-computed tracking analysis
            
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
        
        # 3. Extract frame thumbnails (3 representative frames)
        self._extract_thumbnails()
        
        # 4. Compute edge density per region
        self._compute_edge_density()
        
        return self.features
    
    def _generate_fingerprint(self) -> str:
        """
        Generate a unique fingerprint for this clip.
        
        Combines filepath hash with resolution/fps/duration for collision resistance.
        Used to identify "same clip" across sessions.
        """
        if not self.clip:
            return ""
        
        try:
            # Get absolute filepath
            filepath = bpy.path.abspath(self.clip.filepath) if self.clip.filepath else ""
            
            # Combine with clip properties for robustness
            fingerprint_data = f"{filepath}:{self.clip.size[0]}:{self.clip.size[1]}:{self.clip.fps}:{self.clip.frame_duration}"
            
            # Generate hash
            return hashlib.md5(fingerprint_data.encode()).hexdigest()[:16]
        except Exception as e:
            print(f"AutoSolve: Error generating fingerprint: {e}")
            return ""
    
    def _extract_motion_from_tracking(self, tracking_data: Dict):
        """
        Extract motion classification from tracking analysis.
        
        Uses velocity statistics from tracked markers to classify motion.
        """
        velocities = tracking_data.get('velocities', [])
        
        if not velocities:
            return
        
        # Compute velocity statistics
        mean_vel = sum(velocities) / len(velocities)
        variance = sum((v - mean_vel) ** 2 for v in velocities) / len(velocities)
        
        self.features.motion_magnitude = mean_vel
        self.features.motion_variance = variance ** 0.5
        
        # Classify motion
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
                s.get('avg_velocity', 0.0) for s in frame_samples[:10]
            ]
    
    def _extract_thumbnails(self, sample_count: int = 3):
        """
        Extract frame thumbnails for NN training.
        
        Samples frames at 25%, 50%, 75% of clip duration.
        Extracts actual RGB pixels, downscales to 64x64, encodes as base64 JPEG.
        
        Total overhead: ~200-500ms for 3 thumbnails.
        Storage: ~2-4KB per thumbnail (JPEG compressed).
        """
        if not self.clip or not bpy:
            return
        
        try:
            frame_count = self.clip.frame_duration
            if frame_count < 1:
                return
            
            # Sample at 25%, 50%, 75% of clip duration (more representative than evenly spaced)
            percentages = [0.25, 0.50, 0.75]
            sample_frames = []
            for pct in percentages[:sample_count]:
                frame = self.clip.frame_start + int(frame_count * pct)
                sample_frames.append(min(frame, self.clip.frame_start + frame_count - 1))
            
            self.features.thumbnail_frames = sample_frames
            
            for frame in sample_frames:
                thumbnail = self._render_thumbnail(frame)
                if thumbnail:
                    self.features.thumbnails.append(thumbnail)
            
            if self.features.thumbnails:
                print(f"AutoSolve: Extracted {len(self.features.thumbnails)} thumbnails "
                      f"(~{sum(len(t) for t in self.features.thumbnails) // 1024}KB)")
                
        except Exception as e:
            print(f"AutoSolve: Error extracting thumbnails: {e}")
    
    def _render_thumbnail(self, frame: int, size: int = 64) -> Optional[str]:
        """
        Render a single frame thumbnail with actual pixel data.
        
        Uses Blender's MovieClip preview system to extract frame pixels,
        downscales to target size, and encodes as base64 JPEG.
        
        Args:
            frame: Frame number to render
            size: Output size (64x64 default)
            
        Returns:
            Base64 encoded JPEG data or None on failure
        """
        if not self.clip or not bpy:
            return None
        
        try:
            import base64
            import tempfile
            import os
            
            # Get clip dimensions
            width, height = self.clip.size
            if width == 0 or height == 0:
                return None
            
            # Calculate aspect-correct dimensions
            aspect = width / height
            if aspect > 1:
                thumb_w = size
                thumb_h = max(1, int(size / aspect))
            else:
                thumb_h = size
                thumb_w = max(1, int(size * aspect))
            
            # Method: Use Blender's image loading for the clip's source
            filepath = bpy.path.abspath(self.clip.filepath)
            if not filepath:
                return None
            
            # For movie files, we need to extract the specific frame
            # Create a temporary image to hold the frame
            temp_img_name = f"__autosolve_thumb_{frame}"
            
            # Check if this is a movie or image sequence
            is_movie = filepath.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.mxf'))
            
            if is_movie:
                # For movies, use the compositor to extract frame
                # This is the most reliable method
                thumbnail_data = self._extract_movie_frame(frame, thumb_w, thumb_h)
            else:
                # For image sequences, load the specific frame file
                thumbnail_data = self._extract_sequence_frame(frame, thumb_w, thumb_h)
            
            return thumbnail_data
            
        except Exception as e:
            print(f"AutoSolve: Error rendering thumbnail frame {frame}: {e}")
            return None
    
    def _extract_movie_frame(self, frame: int, width: int, height: int) -> Optional[str]:
        """
        Extract a frame from a movie file using FFmpeg through Blender.
        
        Falls back to a simpler hash-based identifier if extraction fails.
        """
        try:
            import base64
            import tempfile
            import os
            import subprocess
            
            filepath = bpy.path.abspath(self.clip.filepath)
            fps = self.clip.fps if self.clip.fps > 0 else 24
            
            # Calculate timestamp from frame
            time_sec = (frame - self.clip.frame_start) / fps
            
            # Try to use FFmpeg directly (faster than compositor)
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "thumb.jpg")
                
                # FFmpeg command to extract and scale frame
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(time_sec),
                    '-i', filepath,
                    '-vframes', '1',
                    '-vf', f'scale={width}:{height}',
                    '-q:v', '5',  # JPEG quality (2-31, lower is better)
                    output_path
                ]
                
                try:
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        timeout=5,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                    )
                    
                    if result.returncode == 0 and os.path.exists(output_path):
                        with open(output_path, 'rb') as f:
                            jpeg_data = f.read()
                        return base64.b64encode(jpeg_data).decode('utf-8')
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    # FFmpeg not available or timeout
                    pass
            
            # Fallback: use frame identifier if FFmpeg fails
            return self._create_frame_identifier(frame)
            
        except Exception as e:
            return self._create_frame_identifier(frame)
    
    def _extract_sequence_frame(self, frame: int, width: int, height: int) -> Optional[str]:
        """
        Extract a frame from an image sequence.
        """
        try:
            import base64
            
            # Get the specific frame's filepath
            filepath = bpy.path.abspath(self.clip.filepath)
            
            # For sequences, Blender uses # notation or frame numbers
            # Try to construct the frame path
            if '#' in filepath:
                # Replace # with frame number
                num_hashes = filepath.count('#')
                frame_str = str(frame).zfill(num_hashes)
                frame_path = filepath.replace('#' * num_hashes, frame_str)
            else:
                # Try common naming patterns
                frame_path = filepath
            
            # Load and resize image
            if os.path.exists(frame_path):
                # Load image
                img = bpy.data.images.load(frame_path, check_existing=False)
                try:
                    # Scale and save as JPEG
                    img.scale(width, height)
                    
                    # Save to temp file and read back
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    img.save_render(tmp_path)
                    
                    with open(tmp_path, 'rb') as f:
                        jpeg_data = f.read()
                    
                    os.unlink(tmp_path)
                    return base64.b64encode(jpeg_data).decode('utf-8')
                finally:
                    bpy.data.images.remove(img)
            
            return self._create_frame_identifier(frame)
            
        except Exception:
            return self._create_frame_identifier(frame)
    
    def _create_frame_identifier(self, frame: int) -> str:
        """
        Create a lightweight frame identifier as fallback.
        
        Used when actual pixel extraction fails.
        """
        import hashlib
        filepath = bpy.path.abspath(self.clip.filepath) if self.clip else ""
        width, height = self.clip.size if self.clip else (0, 0)
        frame_id = f"{filepath}:{frame}:{width}x{height}"
        frame_hash = hashlib.md5(frame_id.encode()).hexdigest()[:16]
        return f"fallback:{frame}:{frame_hash}"
    
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
