# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
FeatureExtractor - Slim utility focused on density analysis for detection decisions.
"""

from typing import Dict, Optional
from types import SimpleNamespace
import bpy


class FeatureExtractor:
    """
    Slim utility to compute feature density per region.
    """
    
    REGIONS = [
        'top-left', 'top-center', 'top-right',
        'mid-left', 'center', 'mid-right',
        'bottom-left', 'bottom-center', 'bottom-right'
    ]
    
    def __init__(self, clip: Optional[bpy.types.MovieClip] = None):
        self.clip = clip
        self.feature_density: Dict[str, int] = {r: 0 for r in self.REGIONS}
        self.motion_class: str = "MEDIUM"
        self.features = SimpleNamespace(clip_fingerprint="", motion_class="MEDIUM")
        
    def extract_all(self, clip: Optional[bpy.types.MovieClip] = None, 
                    tracking_data: Optional[Dict] = None,
                    force_recompute: bool = False) -> Dict:
        """Slim signature compat for extracting features (stub/density analysis only)."""
        if clip:
            self.clip = clip
            
        if tracking_data and 'detected_feature_density' in tracking_data:
            self.feature_density = tracking_data['detected_feature_density']
            
        return self.feature_density

    def compute_from_tracking(self, tracking_data: Dict):
        """No-op stub for compatibility."""
        pass
        
    def compute_flow_histograms(self, vectors):
        """No-op stub for compatibility."""
        pass
        
    def to_dict(self) -> Dict:
        """Compatibility dict wrapper."""
        return {
            'feature_density': self.feature_density,
            'motion_class': self.motion_class
        }
