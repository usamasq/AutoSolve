# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
SettingsPredictor - Predicts optimal tracking settings.

Uses historical data to recommend the best settings for new footage.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import bpy


class SettingsPredictor:
    """
    Predicts optimal tracking settings based on historical data.
    
    Uses a combination of:
    1. Rule-based heuristics (fallback)
    2. Statistical aggregation of successful sessions
    3. Similarity matching for footage characteristics
    """
    
    # Default settings for cold start
    DEFAULT_SETTINGS = {
        'pattern_size': 15,
        'search_size': 71,
        'correlation': 0.70,
        'threshold': 0.30,
        'motion_model': 'LocRot',
    }
    
    # Bucketed settings for different success rates
    TIERED_SETTINGS = {
        'aggressive': {
            'pattern_size': 27,
            'search_size': 130,
            'correlation': 0.50,
            'threshold': 0.10,
            'motion_model': 'Affine',
        },
        'moderate': {
            'pattern_size': 21,
            'search_size': 100,
            'correlation': 0.65,
            'threshold': 0.25,
            'motion_model': 'Affine',
        },
        'balanced': {
            'pattern_size': 15,
            'search_size': 71,
            'correlation': 0.70,
            'threshold': 0.30,
            'motion_model': 'LocRot',
        },
        'selective': {
            'pattern_size': 13,
            'search_size': 61,
            'correlation': 0.75,
            'threshold': 0.40,
            'motion_model': 'LocRot',
        },
    }
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(bpy.utils.user_resource('DATAFILES')) / 'eztrack'
        
        self.model_path = self.data_dir / 'model.json'
        self.model: Dict = self._load_model()
    
    def _load_model(self) -> Dict:
        """Load trained model from disk."""
        if self.model_path.exists():
            try:
                with open(self.model_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"AutoSolve: Error loading model: {e}")
        
        return {
            'version': 1,
            'footage_classes': {},
            'region_models': {},
            'global_stats': {
                'total_sessions': 0,
                'successful_sessions': 0,
            }
        }
    
    def _save_model(self):
        """Save model to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.model_path, 'w') as f:
            json.dump(self.model, f, indent=2)
    
    def classify_footage(self, clip: bpy.types.MovieClip) -> str:
        """
        Classify footage into a category.
        
        Returns a string key like "HD_30fps" or "4K_24fps".
        """
        width = clip.size[0]
        fps = clip.fps if clip.fps > 0 else 24
        
        # Resolution class
        if width >= 3840:
            res_class = '4K'
        elif width >= 1920:
            res_class = 'HD'
        else:
            res_class = 'SD'
        
        # FPS class
        if fps >= 50:
            fps_class = '60fps'
        elif fps >= 28:
            fps_class = '30fps'
        else:
            fps_class = '24fps'
        
        return f"{res_class}_{fps_class}"
    
    def predict_settings(self, clip: bpy.types.MovieClip, 
                         robust_mode: bool = False) -> Dict:
        """
        Predict optimal settings for the given clip.
        
        Uses historical data if available, otherwise falls back to heuristics.
        """
        footage_class = self.classify_footage(clip)
        
        # Check if we have historical data for this footage class
        if footage_class in self.model['footage_classes']:
            class_data = self.model['footage_classes'][footage_class]
            
            if class_data.get('sample_count', 0) >= 3:
                # We have enough data to make a prediction
                return self._predict_from_history(class_data, robust_mode)
        
        # Fall back to heuristics
        return self._predict_heuristic(clip, robust_mode)
    
    def _predict_from_history(self, class_data: Dict, robust_mode: bool) -> Dict:
        """Predict settings from historical data."""
        avg_success_rate = class_data.get('avg_success_rate', 0.5)
        best_settings = class_data.get('best_settings', {})
        
        # Start with best historical settings
        settings = self.DEFAULT_SETTINGS.copy()
        settings.update(best_settings)
        
        # Adjust based on robust mode
        if robust_mode:
            settings['pattern_size'] = int(settings.get('pattern_size', 15) * 1.4)
            settings['search_size'] = int(settings.get('search_size', 71) * 1.4)
            settings['correlation'] = max(0.5, settings.get('correlation', 0.7) - 0.15)
            settings['motion_model'] = 'Affine'
        
        return settings
    
    def _predict_heuristic(self, clip: bpy.types.MovieClip, 
                           robust_mode: bool) -> Dict:
        """Predict settings using rule-based heuristics."""
        width = clip.size[0]
        fps = clip.fps if clip.fps > 0 else 24
        
        if robust_mode:
            base = self.TIERED_SETTINGS['aggressive'].copy()
        else:
            base = self.TIERED_SETTINGS['balanced'].copy()
        
        # Adjust for resolution
        if width >= 3840:  # 4K
            base['pattern_size'] = int(base['pattern_size'] * 1.5)
        elif width >= 1920:  # HD
            base['pattern_size'] = int(base['pattern_size'] * 1.2)
        
        # Adjust for FPS
        if fps < 30:  # Low FPS = more motion per frame
            base['search_size'] = int(base['search_size'] * 1.3)
        elif fps >= 60:  # High FPS = less motion
            base['search_size'] = int(base['search_size'] * 0.8)
        
        return base
    
    def update_model(self, session_data: Dict):
        """
        Update model with new session data.
        
        Called after each tracking session completes.
        """
        if not session_data:
            return
        
        # Classify footage
        res = session_data.get('resolution', (1920, 1080))
        fps = session_data.get('fps', 24)
        
        width = res[0] if isinstance(res, (list, tuple)) else 1920
        
        if width >= 3840:
            res_class = '4K'
        elif width >= 1920:
            res_class = 'HD'
        else:
            res_class = 'SD'
        
        if fps >= 50:
            fps_class = '60fps'
        elif fps >= 28:
            fps_class = '30fps'
        else:
            fps_class = '24fps'
        
        footage_class = f"{res_class}_{fps_class}"
        
        # Update footage class data
        if footage_class not in self.model['footage_classes']:
            self.model['footage_classes'][footage_class] = {
                'sample_count': 0,
                'success_count': 0,
                'avg_success_rate': 0.0,
                'settings_history': [],
                'best_settings': {},
            }
        
        class_data = self.model['footage_classes'][footage_class]
        class_data['sample_count'] += 1
        
        if session_data.get('success'):
            class_data['success_count'] += 1
            
            # Record successful settings
            class_data['settings_history'].append({
                'settings': session_data.get('settings', {}),
                'solve_error': session_data.get('solve_error', 999),
                'success_rate': session_data.get('successful_tracks', 0) / 
                               max(session_data.get('total_tracks', 1), 1),
            })
            
            # Keep only last 20 successful sessions
            class_data['settings_history'] = class_data['settings_history'][-20:]
            
            # Calculate best settings (lowest error)
            if class_data['settings_history']:
                best = min(class_data['settings_history'], 
                          key=lambda x: x.get('solve_error', 999))
                class_data['best_settings'] = best.get('settings', {})
        
        # Update success rate
        class_data['avg_success_rate'] = (
            class_data['success_count'] / class_data['sample_count']
        )
        
        # Update global stats
        self.model['global_stats']['total_sessions'] += 1
        if session_data.get('success'):
            self.model['global_stats']['successful_sessions'] += 1
        
        # Update region models
        for region, stats in session_data.get('region_stats', {}).items():
            if region not in self.model['region_models']:
                self.model['region_models'][region] = {
                    'total_tracks': 0,
                    'successful_tracks': 0,
                    'avg_lifespan': 0.0,
                }
            
            rm = self.model['region_models'][region]
            rm['total_tracks'] += stats.get('total_tracks', 0)
            rm['successful_tracks'] += stats.get('successful_tracks', 0)
        
        # Save updated model
        self._save_model()
        print(f"AutoSolve: Updated model with session data")
    
    def get_region_advice(self) -> Dict[str, str]:
        """
        Get advice for each region based on historical performance.
        
        Returns dict like {'top-left': 'avoid', 'center': 'prioritize'}
        """
        advice = {}
        
        for region, data in self.model.get('region_models', {}).items():
            total = data.get('total_tracks', 0)
            successful = data.get('successful_tracks', 0)
            
            if total < 10:
                advice[region] = 'unknown'
            elif total > 0:
                rate = successful / total
                if rate < 0.3:
                    advice[region] = 'avoid'
                elif rate > 0.7:
                    advice[region] = 'prioritize'
                else:
                    advice[region] = 'normal'
        
        return advice
    
    def get_tier_for_success_rate(self, success_rate: float) -> str:
        """Get the appropriate settings tier for a success rate."""
        if success_rate < 0.3:
            return 'aggressive'
        elif success_rate < 0.5:
            return 'moderate'
        elif success_rate < 0.7:
            return 'balanced'
        else:
            return 'selective'
    
    def get_stats(self) -> Dict:
        """Get human-readable statistics."""
        global_stats = self.model.get('global_stats', {})
        
        total = global_stats.get('total_sessions', 0)
        successful = global_stats.get('successful_sessions', 0)
        
        return {
            'total_sessions': total,
            'successful_sessions': successful,
            'success_rate': successful / max(total, 1),
            'footage_classes_known': len(self.model.get('footage_classes', {})),
            'regions_analyzed': len(self.model.get('region_models', {})),
        }
