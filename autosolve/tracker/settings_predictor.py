# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
SettingsPredictor - Predicts optimal tracking settings.

Uses pretrained community data to recommend the best settings for new footage.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Set
import bpy
import numpy as np

from .constants import TIERED_SETTINGS, DEFAULT_SETTINGS
from .utils import classify_footage as classify_footage_util


class SettingsPredictor:
    """
    Predicts optimal tracking settings based on static community-trained presets.
    """
    
    @property
    def DEFAULT_SETTINGS(self):
        return DEFAULT_SETTINGS
    
    @property 
    def TIERED_SETTINGS(self):
        return TIERED_SETTINGS
    
    def __init__(self):
        self.model = self._load_defaults()
    
    def _load_defaults(self) -> Dict:
        """Load bundled pretrained model/presets from disk."""
        model = None
        pretrained_path = Path(__file__).parent / 'presets' / 'defaults.json'
        if pretrained_path.exists():
            try:
                with open(pretrained_path, encoding='utf-8') as f:
                    model = json.load(f)
            except Exception as e:
                print(f"AutoSolve: Error loading presets/defaults.json: {e}")
        
        if model is None:
            # Safe fallback structure
            model = {
                'version': 1,
                'footage_classes': {},
                'footage_type_adjustments': {},
                'region_models': {},
            }
        return model
    
    def classify_footage(self, clip: bpy.types.MovieClip) -> str:
        """Classify footage into a category (e.g., "HD_30fps")."""
        return classify_footage_util(clip)
    
    def predict_settings(self, clip: bpy.types.MovieClip, 
                          robust_mode: bool = False,
                          footage_type: str = 'AUTO') -> Dict:
        """
        Predict optimal settings for the given clip using presets.
        
        Args:
            clip: The Movie Clip to analyze
            robust_mode: Use more aggressive settings for difficult footage
            footage_type: User-specified footage type (INDOOR, DRONE, etc.)
        """
        # Try ONNX Settings Optimizer first
        try:
            from .onnx_predictor import OnnxPredictor
            opt = OnnxPredictor.get_instance()
            if opt.settings_model_available and clip.size[0] > 0 and clip.size[1] > 0:
                def feature_fn(cand):
                    # 1. Clip features (4)
                    width = float(clip.size[0])
                    height = float(clip.size[1])
                    fps = float(clip.fps if clip.fps > 0 else 24.0)
                    frame_count = float(clip.frame_duration)
                    
                    # 2. Boolean switches (2)
                    tripod = 1.0 if (footage_type == 'TRIPOD' or getattr(clip.tracking, 'use_tripod', False)) else 0.0
                    robust = 1.0 if robust_mode else 0.0
                    
                    # 3. Candidate settings (4)
                    pattern = float(cand['pattern_size'])
                    search = float(cand['search_size'])
                    corr = float(cand['correlation'])
                    thresh = float(cand['threshold'])
                    
                    # 4. Footage type one-hot (10)
                    FOOTAGE_TYPES = ['AUTO', 'INDOOR', 'OUTDOOR', 'DRONE', 'HANDHELD', 'GIMBAL', 'ACTION', 'VFX', 'SCREEN', 'CINEMATIC']
                    f_type_oh = [0.0] * len(FOOTAGE_TYPES)
                    f_type = footage_type if footage_type in FOOTAGE_TYPES else 'AUTO'
                    f_type_oh[FOOTAGE_TYPES.index(f_type)] = 1.0
                    
                    # 5. Motion model one-hot (4)
                    MOTION_MODELS = ['Loc', 'LocRot', 'Affine', 'Perspective']
                    m_model_oh = [0.0] * len(MOTION_MODELS)
                    m_model = cand['motion_model']
                    if m_model in MOTION_MODELS:
                        m_model_oh[MOTION_MODELS.index(m_model)] = 1.0
                    else:
                        m_model_oh[1] = 1.0
                        
                    # 6. Video features (4) (estimating motion dynamically from tracks)
                    mean_motion = 0.5
                    try:
                        if clip.tracking.tracks:
                            disps = []
                            current_frame = bpy.context.scene.frame_current
                            for track in clip.tracking.tracks:
                                markers = [m for m in track.markers if not m.mute]
                                recent_markers = [m for m in markers if current_frame - 5 <= m.frame <= current_frame]
                                if len(recent_markers) >= 2:
                                    recent_markers.sort(key=lambda x: x.frame)
                                    track_disps = []
                                    for i in range(1, len(recent_markers)):
                                        dx = recent_markers[i].co.x - recent_markers[i-1].co.x
                                        dy = recent_markers[i].co.y - recent_markers[i-1].co.y
                                        track_disps.append((dx**2 + dy**2) ** 0.5)
                                    if track_disps:
                                        disps.append(np.mean(track_disps))
                            if disps:
                                # Scale by 100 to match feature normalization scale in training
                                mean_motion = float(np.mean(disps)) * 100.0
                    except Exception:
                        pass

                    v_feats = [mean_motion, 0.0, 0.0, 0.003, 0.0] # mean_motion, zoom_divergence, distortion_factor, grain_noise, dynamic_area_ratio
                    if footage_type == 'ACTION' and mean_motion == 0.5:
                        v_feats[0] = 3.5
                    elif footage_type == 'DRONE':
                        v_feats[1] = 0.5
                        
                    return np.array(
                        [width, height, fps, frame_count, tripod, robust, pattern, search, corr, thresh] +
                        f_type_oh + m_model_oh + v_feats,
                        dtype=np.float32
                    )

                patterns = [11, 15, 17, 21, 31, 55]
                searches = [51, 71, 91, 121, 231]
                correlations = [0.55, 0.65, 0.70, 0.75, 0.85]
                thresholds = [0.1, 0.2, 0.3, 0.4]
                motion_models = ['Loc', 'LocRot', 'Affine', 'Perspective']
                
                candidates = []
                for p in patterns:
                    for s in searches:
                        for c in correlations:
                            for t in thresholds:
                                for m in motion_models:
                                    candidates.append({
                                        'pattern_size': p,
                                        'search_size': s,
                                        'correlation': c,
                                        'threshold': t,
                                        'motion_model': m
                                    })
                
                ranked = opt.rank_settings_candidates(candidates, feature_fn)
                if ranked:
                    best_reward, best_settings = ranked[0]
                    print(f"AutoSolve: Neural Engine selected settings with predicted reward {best_reward:.4f}")
                    return best_settings.copy()
        except Exception as e:
            print(f"AutoSolve: Neural Engine settings prediction failed, falling back: {e}")

        # Fallback to heuristics / presets
        footage_class = self.classify_footage(clip)
        
        # 1. Get base settings for footage class
        class_data = self.model.get('footage_classes', {}).get(footage_class, {})
        if class_data and 'best_settings' in class_data:
            settings = class_data['best_settings'].copy()
        else:
            settings = self._predict_heuristic(clip, robust_mode)
        
        # 2. Apply footage type adjustments
        if footage_type != 'AUTO':
            settings = self._apply_footage_type_adjustment(settings, footage_type)
        
        # 3. Estimate motion and adjust
        motion_factor = self._estimate_motion_factor(clip)
        settings = self._adjust_for_motion(settings, motion_factor)
        
        # 4. Apply robust mode overrides if requested
        if robust_mode:
            settings['pattern_size'] = int(settings.get('pattern_size', 15) * 1.4) | 1
            settings['search_size'] = int(settings.get('search_size', 71) * 1.4) | 1
            settings['correlation'] = max(0.5, settings.get('correlation', 0.7) - 0.15)
            settings['motion_model'] = 'Affine'
            
        return settings
    
    def _apply_footage_type_adjustment(self, settings: Dict, footage_type: str) -> Dict:
        """Apply multipliers and offsets based on footage type (DRONE, INDOOR, etc.)."""
        adjustments = self.model.get('footage_type_adjustments', {}).get(footage_type, {})
        if not adjustments:
            return settings
        
        adjusted = settings.copy()
        
        if 'pattern_size_mult' in adjustments:
            adjusted['pattern_size'] = int(settings['pattern_size'] * adjustments['pattern_size_mult'])
            
        if 'search_size_mult' in adjustments:
            adjusted['search_size'] = int(settings['search_size'] * adjustments['search_size_mult'])
            
        if 'threshold_mult' in adjustments:
            adjusted['threshold'] = settings['threshold'] * adjustments['threshold_mult']
            
        if 'correlation_offset' in adjustments:
            adjusted['correlation'] = max(0.4, min(0.9, settings['correlation'] + adjustments['correlation_offset']))
            
        if 'motion_model' in adjustments:
            adjusted['motion_model'] = adjustments['motion_model']
            
        return adjusted
    
    def _estimate_motion_factor(self, clip: bpy.types.MovieClip) -> float:
        """Estimate motion multiplier based on frame rate and duration."""
        fps = clip.fps if clip.fps > 0 else 24
        duration = clip.frame_duration
        
        fps_factor = 30 / fps
        if duration < 100:
            duration_factor = 1.3
        elif duration < 300:
            duration_factor = 1.0
        else:
            duration_factor = 0.9
            
        return max(0.5, min(2.0, fps_factor * duration_factor))
    
    def _adjust_for_motion(self, settings: Dict, motion_factor: float) -> Dict:
        """Scale search sizes dynamically according to motion factor."""
        if motion_factor == 1.0:
            return settings
            
        adjusted = settings.copy()
        adjusted['search_size'] = int(settings['search_size'] * motion_factor)
        
        if motion_factor > 1.2:
            adjusted['correlation'] = max(0.45, settings['correlation'] - 0.05)
            
        if adjusted['search_size'] % 2 == 0:
            adjusted['search_size'] += 1
            
        return adjusted
    
    def _predict_heuristic(self, clip: bpy.types.MovieClip, robust_mode: bool) -> Dict:
        """Fallback rule-based heuristic prediction."""
        width = clip.size[0]
        fps = clip.fps if clip.fps > 0 else 24
        
        if robust_mode:
            base = self.TIERED_SETTINGS['aggressive'].copy()
        else:
            base = self.TIERED_SETTINGS['balanced'].copy()
            
        if width >= 3840:
            base['pattern_size'] = int(base['pattern_size'] * 1.5)
        elif width >= 1920:
            base['pattern_size'] = int(base['pattern_size'] * 1.2)
            
        if fps < 30:
            base['search_size'] = int(base['search_size'] * 1.3)
        elif fps >= 60:
            base['search_size'] = int(base['search_size'] * 0.8)
            
        return base
        
    def get_dead_zones_for_class(self, footage_class: str) -> Set[str]:
        """Identify region dead zones from defaults database."""
        dead_zones = set()
        for region, data in self.model.get('region_models', {}).items():
            success_rate = data.get('success_rate', 1.0)
            if success_rate < 0.25:
                dead_zones.add(region)
        return dead_zones
        
    def get_region_advice(self) -> Dict[str, str]:
        """Provides prioritization advice for regions based on success rate."""
        advice = {}
        for region, data in self.model.get('region_models', {}).items():
            success_rate = data.get('success_rate', 1.0)
            if success_rate < 0.3:
                advice[region] = 'avoid'
            elif success_rate > 0.7:
                advice[region] = 'prioritize'
            else:
                advice[region] = 'normal'
        return advice
