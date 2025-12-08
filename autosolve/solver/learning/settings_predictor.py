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
            self.data_dir = Path(bpy.utils.user_resource('DATAFILES')) / 'autosolve'
        
        self.model_path = self.data_dir / 'model.json'
        self.model: Dict = self._load_model()
    
    def _load_model(self) -> Dict:
        """Load trained model from disk."""
        model = None
        
        # First try user's model
        if self.model_path.exists():
            try:
                with open(self.model_path) as f:
                    model = json.load(f)
            except Exception as e:
                print(f"AutoSolve: Error loading model: {e}")
        
        # Fall back to bundled pretrained model
        if model is None:
            pretrained_path = Path(__file__).parent / 'pretrained_model.json'
            if pretrained_path.exists():
                try:
                    with open(pretrained_path) as f:
                        print("AutoSolve: Using pretrained model")
                        model = json.load(f)
                except Exception as e:
                    print(f"AutoSolve: Error loading pretrained model: {e}")
        
        # Default empty model
        if model is None:
            model = {
                'version': 1,
                'footage_classes': {},
                'region_models': {},
            }
        
        # Migrate older models - ensure required keys exist
        if 'global_stats' not in model:
            model['global_stats'] = {
                'total_sessions': 0,
                'successful_sessions': 0,
            }
        if 'failure_patterns' not in model:
            model['failure_patterns'] = {}
        if 'footage_type_adjustments' not in model:
            model['footage_type_adjustments'] = {}
        
        return model

    
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
                         robust_mode: bool = False,
                         footage_type: str = 'AUTO') -> Dict:
        """
        Predict optimal settings for the given clip.
        
        Uses historical data if available, otherwise falls back to heuristics.
        Applies footage type adjustments for specialized scenarios.
        
        Args:
            clip: The Movie Clip to analyze
            robust_mode: Use more aggressive settings for difficult footage
            footage_type: User-specified footage type (INDOOR, DRONE, etc.)
        """
        footage_class = self.classify_footage(clip)
        
        # Check if we have historical data for this footage class
        if footage_class in self.model.get('footage_classes', {}):
            class_data = self.model['footage_classes'][footage_class]
            
            # Use pretrained data even with sample_count >= 1
            if class_data.get('sample_count', 0) >= 1:
                settings = self._predict_from_history(class_data, robust_mode)
            else:
                settings = self._predict_heuristic(clip, robust_mode)
        else:
            settings = self._predict_heuristic(clip, robust_mode)
        
        # Apply footage type adjustments
        if footage_type != 'AUTO':
            settings = self._apply_footage_type_adjustment(settings, footage_type)
        
        # Estimate motion from clip properties
        motion_factor = self._estimate_motion_factor(clip)
        settings = self._adjust_for_motion(settings, motion_factor)
        
        return settings
    
    def _apply_footage_type_adjustment(self, settings: Dict, footage_type: str) -> Dict:
        """Apply adjustments based on footage type (DRONE, INDOOR, etc.)."""
        adjustments = self.model.get('footage_type_adjustments', {}).get(footage_type, {})
        
        if not adjustments:
            return settings
        
        adjusted = settings.copy()
        
        # Apply multipliers
        if 'pattern_size_mult' in adjustments:
            adjusted['pattern_size'] = int(settings['pattern_size'] * adjustments['pattern_size_mult'])
        
        if 'search_size_mult' in adjustments:
            adjusted['search_size'] = int(settings['search_size'] * adjustments['search_size_mult'])
        
        if 'threshold_mult' in adjustments:
            adjusted['threshold'] = settings['threshold'] * adjustments['threshold_mult']
        
        if 'correlation_offset' in adjustments:
            adjusted['correlation'] = max(0.4, min(0.9, 
                settings['correlation'] + adjustments['correlation_offset']))
        
        if 'motion_model' in adjustments:
            adjusted['motion_model'] = adjustments['motion_model']
        
        return adjusted
    
    def _estimate_motion_factor(self, clip: bpy.types.MovieClip) -> float:
        """
        Estimate motion amount based on clip properties.
        
        Returns a factor from 0.5 (very slow) to 2.0 (very fast motion).
        """
        fps = clip.fps if clip.fps > 0 else 24
        duration = clip.frame_duration
        
        # Higher FPS typically means smoother/less motion per frame
        fps_factor = 30 / fps  # Normalize to 30fps baseline
        
        # Very short clips often have more dramatic motion
        if duration < 100:
            duration_factor = 1.3
        elif duration < 300:
            duration_factor = 1.0
        else:
            duration_factor = 0.9
        
        # Combine factors
        motion_factor = fps_factor * duration_factor
        
        # Clamp to reasonable range
        return max(0.5, min(2.0, motion_factor))
    
    def _adjust_for_motion(self, settings: Dict, motion_factor: float) -> Dict:
        """Adjust settings based on estimated motion amount."""
        if motion_factor == 1.0:
            return settings
        
        adjusted = settings.copy()
        
        # Higher motion = larger search area
        adjusted['search_size'] = int(settings['search_size'] * motion_factor)
        
        # Higher motion = slightly more lenient correlation
        if motion_factor > 1.2:
            adjusted['correlation'] = max(0.45, settings['correlation'] - 0.05)
        
        # Ensure odd values for pattern/search size
        if adjusted['search_size'] % 2 == 0:
            adjusted['search_size'] += 1
        
        return adjusted

    
    def _predict_from_history(self, class_data: Dict, robust_mode: bool) -> Dict:
        """
        Predict settings from historical data using success-weighted averaging.
        
        Sessions with lower solve error get more weight in determining optimal settings.
        This is more robust than just using the single best session.
        """
        settings_history = class_data.get('settings_history', [])
        
        if not settings_history:
            # No history, use defaults
            settings = self.DEFAULT_SETTINGS.copy()
        else:
            # Success-weighted averaging: lower error = higher weight
            weighted_settings = {}
            total_weight = 0
            
            for entry in settings_history:
                error = entry.get('solve_error', 2.0)
                success_rate = entry.get('success_rate', 0.5)
                
                # Weight by inverse error AND success rate
                weight = (1.0 / max(error, 0.1)) * success_rate
                total_weight += weight
                
                entry_settings = entry.get('settings', {})
                for key in ['pattern_size', 'search_size', 'correlation', 'threshold']:
                    if key in entry_settings:
                        if key not in weighted_settings:
                            weighted_settings[key] = 0
                        weighted_settings[key] += entry_settings[key] * weight
            
            # Normalize by total weight
            if total_weight > 0:
                for key in weighted_settings:
                    weighted_settings[key] /= total_weight
            
            # Build final settings
            settings = self.DEFAULT_SETTINGS.copy()
            for key, value in weighted_settings.items():
                if key in ['pattern_size', 'search_size']:
                    # Round to odd integer
                    settings[key] = int(value) | 1  # Ensure odd
                else:
                    settings[key] = round(value, 2)
            
            # Keep motion_model from best session
            best_entry = min(settings_history, key=lambda x: x.get('solve_error', 999))
            if 'motion_model' in best_entry.get('settings', {}):
                settings['motion_model'] = best_entry['settings']['motion_model']
        
        # Adjust based on robust mode
        if robust_mode:
            settings['pattern_size'] = int(settings.get('pattern_size', 15) * 1.4) | 1
            settings['search_size'] = int(settings.get('search_size', 71) * 1.4) | 1
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
        
        # Update global stats (ensure exists for older model formats)
        if 'global_stats' not in self.model:
            self.model['global_stats'] = {
                'total_sessions': 0,
                'successful_sessions': 0,
            }
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
            
            # Handle both dict format and legacy integer format
            if isinstance(stats, dict):
                new_tracks = stats.get('total_tracks', 0)
                new_successful = stats.get('successful_tracks', 0)
                new_lifespan = stats.get('avg_lifespan', 0.0)
                
                # Update running average for lifespan
                old_count = rm['total_tracks']
                rm['total_tracks'] += new_tracks
                rm['successful_tracks'] += new_successful
                
                # Compute weighted average of lifespans
                if rm['total_tracks'] > 0 and new_tracks > 0:
                    old_weight = old_count / rm['total_tracks']
                    new_weight = new_tracks / rm['total_tracks']
                    rm['avg_lifespan'] = (rm['avg_lifespan'] * old_weight) + (new_lifespan * new_weight)
            elif isinstance(stats, (int, float)):
                # Legacy format: just a count
                rm['total_tracks'] += int(stats)
        
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
    
    def record_failure(self, footage_class: str, failure_pattern: str, settings: Dict):
        """
        Learn from failure: record what settings DON'T work for this footage type + failure.
        
        This helps avoid repeating mistakes on similar footage.
        
        Args:
            footage_class: From classify_footage() e.g. "HD_30fps"
            failure_pattern: From FailureDiagnostics e.g. "rapid_motion"
            settings: The settings that led to failure
        """
        if 'failure_patterns' not in self.model:
            self.model['failure_patterns'] = {}
        
        key = f"{footage_class}_{failure_pattern}"
        
        if key not in self.model['failure_patterns']:
            self.model['failure_patterns'][key] = {
                'count': 0,
                'avoid_settings': []
            }
        
        self.model['failure_patterns'][key]['count'] += 1
        
        # Store simplified settings to avoid
        avoid_entry = {
            'search_size': settings.get('search_size', 71),
            'pattern_size': settings.get('pattern_size', 15),
            'correlation': settings.get('correlation', 0.7),
        }
        self.model['failure_patterns'][key]['avoid_settings'].append(avoid_entry)
        
        # Keep only last 10 failure records per pattern
        self.model['failure_patterns'][key]['avoid_settings'] = \
            self.model['failure_patterns'][key]['avoid_settings'][-10:]
        
        self._save_model()
        print(f"AutoSolve: Recorded failure - {failure_pattern} for {footage_class}")
    
    def should_avoid_settings(self, footage_class: str, settings: Dict) -> bool:
        """
        Check if settings should be avoided based on past failures.
        
        Returns True if these settings are too similar to settings that
        have failed multiple times for similar footage.
        """
        if 'failure_patterns' not in self.model:
            return False
        
        # Check all failure patterns for this footage class
        for key, data in self.model['failure_patterns'].items():
            if not key.startswith(footage_class):
                continue
            
            if data['count'] < 2:
                continue  # Need at least 2 failures to warn
            
            # Check if current settings are similar to failed settings
            for failed in data['avoid_settings']:
                if self._settings_similar(settings, failed):
                    return True
        
        return False
    
    def _settings_similar(self, settings1: Dict, settings2: Dict, tolerance: float = 0.15) -> bool:
        """Check if two settings configurations are similar."""
        for key in ['search_size', 'pattern_size', 'correlation']:
            v1 = settings1.get(key, 0)
            v2 = settings2.get(key, 0)
            
            if v1 == 0 or v2 == 0:
                continue
            
            # Check if values are within tolerance
            diff = abs(v1 - v2) / max(v1, v2)
            if diff > tolerance:
                return False
        
        return True
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LocalLearningModel Compatibility Methods (for backwards compatibility)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_settings_for_class(self, footage_class: str) -> Optional[Dict]:
        """Get learned settings for a footage class (LocalLearningModel compat)."""
        class_data = self.model.get('footage_classes', {}).get(footage_class, {})
        if class_data.get('settings_history'):
            # Use weighted prediction logic
            return self._predict_from_history(class_data, False)
        return None
    
    def get_dead_zones_for_class(self, footage_class: str) -> set:
        """Get dead zones for a footage class (LocalLearningModel compat)."""
        dead_zones = set()
        
        # Check region models for low success rates
        for region, data in self.model.get('region_models', {}).items():
            total = data.get('total_tracks', 0)
            successful = data.get('successful_tracks', 0)
            
            if total >= 10:  # Need enough samples
                rate = successful / total
                if rate < 0.25:
                    dead_zones.add(region)
        
        return dead_zones
    
    def get_data(self, footage_class: str) -> Optional[Dict]:
        """Get raw data for a footage class (LocalLearningModel compat)."""
        return self.model.get('footage_classes', {}).get(footage_class)
    
    def update(self, footage_class: str, data: Dict):
        """Update data for a footage class (LocalLearningModel compat)."""
        if 'footage_classes' not in self.model:
            self.model['footage_classes'] = {}
        self.model['footage_classes'][footage_class] = data
        self._save_model()
    
    def update_from_session(self, footage_class: str, success: bool, settings: Dict, 
                            error: float = 999.0, region_stats: Dict = None):
        """Update from session data (LocalLearningModel compat)."""
        session_data = {
            'footage_class': footage_class,
            'success': success,
            'settings': settings,
            'solve_error': error,
            'region_stats': region_stats or {},
        }
        self.update_model(session_data)
    
    def save(self):
        """Save model (LocalLearningModel compat)."""
        self._save_model()
