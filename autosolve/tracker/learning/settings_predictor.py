# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
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

from ..constants import TIERED_SETTINGS, DEFAULT_SETTINGS
from ..utils import classify_footage as classify_footage_util, get_autosolve_data_dir, get_model_path


class SettingsPredictor:
    """
    Predicts optimal tracking settings based on historical data.
    
    Uses a combination of:
    1. Rule-based heuristics (fallback)
    2. Statistical aggregation of successful sessions
    3. Similarity matching for footage characteristics
    """
    
    # Using DEFAULT_SETTINGS and TIERED_SETTINGS from constants.py
    
    # Method-level access for compatibility
    @property
    def DEFAULT_SETTINGS(self):
        return DEFAULT_SETTINGS
    
    @property 
    def TIERED_SETTINGS(self):
        return TIERED_SETTINGS
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = get_autosolve_data_dir()
        
        self.model_path = get_model_path()
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
        
        # Ensure required keys exist
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
        """Save model to disk atomically to prevent corruption on crash."""
        import tempfile
        import os
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Write to temp file first, then rename (atomic on most filesystems)
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.json', 
                dir=self.data_dir, 
                delete=False
            ) as tmp:
                json.dump(self.model, tmp, indent=2)
                tmp_path = tmp.name
            
            # Atomic replace (os.replace is atomic on POSIX and Windows)
            os.replace(tmp_path, self.model_path)
        except (OSError, IOError) as e:
            print(f"AutoSolve: Error saving model: {e}")
            # Clean up temp file if it exists
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HER REWARD COMPUTATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _compute_reward_static(success: bool, solve_error: float, bundle_count: int) -> float:
        """
        Compute reward signal for ML training (static version for migration).
        
        This creates a continuous reward signal from 0.0 to 1.0 that works for
        both successes and failures, enabling Hindsight Experience Replay.
        
        Reward = error_score * 0.5 + bundle_score * 0.3 + success_bonus * 0.2
        
        Args:
            success: Whether the solve succeeded
            solve_error: Reprojection error in pixels
            bundle_count: Number of 3D bundles created
            
        Returns:
            Reward value between 0.0 and 1.0
        """
        # Error score: 0 error = 1.0, 10+ error = 0.0
        error_score = max(0.0, 1.0 - solve_error / 10.0)
        
        # Bundle score: 100+ bundles = 1.0, 0 bundles = 0.0
        bundle_score = min(1.0, bundle_count / 100.0)
        
        # Success bonus
        success_bonus = 1.0 if success else 0.0
        
        # Weighted combination
        reward = error_score * 0.5 + bundle_score * 0.3 + success_bonus * 0.2
        
        return round(reward, 3)
    
    def compute_reward(self, success: bool, solve_error: float, bundle_count: int) -> float:
        """
        Compute reward signal for ML training.
        
        Instance method wrapper for _compute_reward_static.
        """
        return self._compute_reward_static(success, solve_error, bundle_count)
    
    def classify_footage(self, clip: bpy.types.MovieClip) -> str:
        """
        Classify footage into a category.
        
        Returns a string key like "HD_30fps" or "4K_24fps".
        """
        return classify_footage_util(clip)
    
    def predict_settings(self, clip: bpy.types.MovieClip, 
                         robust_mode: bool = False,
                         footage_type: str = 'AUTO',
                         motion_class: str = None,
                         clip_fingerprint: str = None) -> Dict:
        """
        Predict optimal settings for the given clip.
        
        Enhanced with:
        1. Per-clip fingerprinting (exact clip reuse)
        2. Motion-based sub-classification (LOW/MEDIUM/HIGH)
        3. HER weighted prediction
        
        Args:
            clip: The Movie Clip to analyze
            robust_mode: Use more aggressive settings for difficult footage
            footage_type: User-specified footage type (INDOOR, DRONE, etc.)
            motion_class: Optional motion classification (LOW, MEDIUM, HIGH)
            clip_fingerprint: Optional clip fingerprint for per-clip lookup
        """
        footage_class = self.classify_footage(clip)
        
        # Priority 1: Check for per-clip learned settings (exact match)
        if clip_fingerprint:
            clip_settings = self._get_clip_specific_settings(clip_fingerprint)
            if clip_settings:
                print(f"AutoSolve: Using per-clip learned settings (fingerprint: {clip_fingerprint[:8]})")
                return clip_settings
        
        # Priority 2: Check for motion-based sub-classification
        if motion_class:
            enhanced_class = f"{footage_class}_{motion_class}_MOTION"
            if enhanced_class in self.model.get('footage_classes', {}):
                class_data = self.model['footage_classes'][enhanced_class]
                if class_data.get('sample_count', 0) >= 1:
                    settings = self._predict_from_history(class_data, robust_mode)
                    print(f"AutoSolve: Using motion-enhanced class: {enhanced_class}")
                    # Apply remaining adjustments and return
                    if footage_type != 'AUTO':
                        settings = self._apply_footage_type_adjustment(settings, footage_type)
                    motion_factor = self._estimate_motion_factor(clip)
                    settings = self._adjust_for_motion(settings, motion_factor)
                    settings = self._apply_behavior_adjustments(settings, enhanced_class)
                    return settings
        
        # Priority 3: Check base footage class
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
        
        # Apply learned behavior adjustments (cautious: only when confident)
        settings = self._apply_behavior_adjustments(settings, footage_class)
        
        return settings
    
    def _get_clip_specific_settings(self, clip_fingerprint: str) -> Optional[Dict]:
        """
        Get settings for a specific clip by fingerprint.
        
        Returns the best-performing settings for this exact clip if available.
        """
        clip_data = self.model.get('clip_specific', {}).get(clip_fingerprint)
        if not clip_data:
            return None
        
        # Return best settings if we have them
        if clip_data.get('best_settings'):
            return clip_data['best_settings'].copy()
        
        return None
    
    def save_clip_specific_settings(self, clip_fingerprint: str, settings: Dict, 
                                     success: bool, solve_error: float):
        """
        Save settings for a specific clip by fingerprint.
        
        Only saves if this is better than previous attempts.
        """
        if 'clip_specific' not in self.model:
            self.model['clip_specific'] = {}
        
        if clip_fingerprint not in self.model['clip_specific']:
            self.model['clip_specific'][clip_fingerprint] = {
                'attempts': 0,
                'best_error': 999.0,
                'best_settings': None,
                'last_success': False,
            }
        
        clip_data = self.model['clip_specific'][clip_fingerprint]
        clip_data['attempts'] += 1
        clip_data['last_success'] = success
        
        # Update best settings if this was better
        if success and solve_error < clip_data.get('best_error', 999.0):
            clip_data['best_error'] = solve_error
            clip_data['best_settings'] = settings.copy()
            print(f"AutoSolve: Saved best settings for clip (error: {solve_error:.2f}px)")
        
        # Limit clip-specific storage to avoid unbounded growth
        # Keep only last 100 clips
        if len(self.model['clip_specific']) > 100:
            # Remove oldest entries (by lowest attempt count)
            sorted_clips = sorted(
                self.model['clip_specific'].items(),
                key=lambda x: x[1].get('attempts', 0)
            )
            for fingerprint, _ in sorted_clips[:10]:  # Remove 10 oldest
                del self.model['clip_specific'][fingerprint]
        
        self._save_model()
    
    def _apply_behavior_adjustments(self, settings: Dict, footage_class: str) -> Dict:
        """
        Apply learned adjustments from user behavior.
        
        Only applies if:
        - 3+ similar behaviors observed
        - 0.7+ confidence (improvements were consistent)
        """
        adjusted = settings.copy()
        applied = []
        
        for setting_name in ['pattern_size', 'search_size', 'correlation', 'threshold']:
            delta = self.get_behavior_adjustment(footage_class, setting_name)
            if delta is not None and setting_name in adjusted:
                old_val = adjusted[setting_name]
                adjusted[setting_name] = old_val + delta
                applied.append(f"{setting_name}: {old_val:.1f}→{adjusted[setting_name]:.1f}")
        
        if applied:
            print(f"AutoSolve: Applied learned behavior adjustments: {', '.join(applied)}")
        
        return adjusted
    
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
        Predict settings from historical data using HER reward-weighted averaging.
        
        HER Approach + Recency Weighting:
        - High reward experiences → positive weight → pull settings toward them
        - Low reward experiences → used to compute "settings to avoid"
        - Recent experiences → 2x weight (recency decay factor)
        - Net effect: settings converge to optimal while avoiding known failures
        """
        # Prefer new experiences array, fall back to legacy settings_history
        experiences = class_data.get('experiences', [])
        settings_history = class_data.get('settings_history', [])
        
        if not experiences and not settings_history:
            # No history, use defaults
            settings = self.DEFAULT_SETTINGS.copy()
        elif experiences:
            # ═══════════════════════════════════════════════════════════════════
            # HER REWARD-WEIGHTED PREDICTION WITH RECENCY WEIGHTING
            # ═══════════════════════════════════════════════════════════════════
            weighted_settings = {}
            total_weight = 0
            
            # Compute settings to avoid from low-reward experiences
            avoid_settings = self._compute_settings_to_avoid(experiences)
            
            # Apply recency weighting: newer experiences (later in list) get higher weight
            # Decay factor 0.85 means exp[-5] has 44% weight of exp[-1]
            recency_decay = 0.85
            num_exp = len(experiences)
            
            for idx, exp in enumerate(experiences):
                reward = exp.get('reward', 0.5)
                
                # Only use experiences with reward > 0.3 for positive contribution
                # Lower reward experiences contribute via avoid_settings
                if reward <= 0.3:
                    continue
                
                # Recency weight: newer = higher (idx closer to num_exp = newer)
                recency_weight = recency_decay ** (num_exp - 1 - idx)
                
                # Weight by reward (higher reward = more influence) AND recency
                weight = (reward ** 2) * recency_weight
                total_weight += weight
                
                exp_settings = exp.get('settings', {})
                for key in ['pattern_size', 'search_size', 'correlation', 'threshold']:
                    if key in exp_settings:
                        if key not in weighted_settings:
                            weighted_settings[key] = 0
                        weighted_settings[key] += exp_settings[key] * weight
            
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
            
            # ═══════════════════════════════════════════════════════════════════
            # FAILURE AVOIDANCE: Adjust settings AWAY from known-bad values
            # ═══════════════════════════════════════════════════════════════════
            if avoid_settings:
                settings = self._apply_failure_avoidance(settings, avoid_settings)
                print(f"AutoSolve: Applied failure avoidance from {len(avoid_settings)} bad experiences")
            
            # Keep motion_model from highest reward experience
            best_exp = max(experiences, key=lambda x: x.get('reward', 0))
            if 'motion_model' in best_exp.get('settings', {}):
                settings['motion_model'] = best_exp['settings']['motion_model']
        else:
            # Legacy path: use old settings_history format
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
                    settings[key] = int(value) | 1
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
    
    def _compute_settings_to_avoid(self, experiences: List[Dict]) -> List[Dict]:
        """
        Extract settings from low-reward experiences that should be avoided.
        
        Returns list of settings dicts that led to failures.
        """
        avoid = []
        for exp in experiences:
            if exp.get('reward', 1.0) < 0.3:  # Threshold for "failure"
                avoid.append(exp.get('settings', {}))
        return avoid
    
    def _apply_failure_avoidance(self, settings: Dict, avoid_settings: List[Dict]) -> Dict:
        """
        Adjust settings to move AWAY from known-bad configurations.
        
        For each setting that matches a failure, nudge it in the opposite direction.
        """
        adjusted = settings.copy()
        
        for bad in avoid_settings:
            # Check if current settings are too similar to failed settings
            for key in ['search_size', 'pattern_size']:
                if key in bad and key in adjusted:
                    bad_val = bad[key]
                    current_val = adjusted[key]
                    
                    # If within 20% of bad value, increase by 30%
                    if abs(current_val - bad_val) / max(bad_val, 1) < 0.2:
                        adjusted[key] = int(current_val * 1.3) | 1  # Increase and ensure odd
            
            # For correlation, move opposite direction
            if 'correlation' in bad and 'correlation' in adjusted:
                bad_corr = bad['correlation']
                current_corr = adjusted['correlation']
                
                # If correlation is similar to failed, adjust
                if abs(current_corr - bad_corr) < 0.1:
                    # If failure had high correlation, try lower (more lenient)
                    # If failure had low correlation, try higher (stricter)
                    if bad_corr > 0.6:
                        adjusted['correlation'] = max(0.45, current_corr - 0.1)
                    else:
                        adjusted['correlation'] = min(0.85, current_corr + 0.1)
        
        return adjusted
    
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
        Update model with new session data using Hindsight Experience Replay.
        
        HER Approach: ALL sessions are valuable training data.
        - Successes: High reward, directly inform optimal settings
        - Failures: Low reward, inform what settings to AVOID
        
        Called after each tracking session completes.
        """
        if not session_data:
            return
        
        # BUG 1 FIX: Use passed footage_class if available, only compute if not provided
        footage_class = session_data.get('footage_class')
        if not footage_class:
            # Fall back to computing from resolution/fps (for backwards compat)
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
        
        # Ensure footage class exists with new schema
        if footage_class not in self.model['footage_classes']:
            self.model['footage_classes'][footage_class] = {
                'sample_count': 0,
                'success_count': 0,
                'avg_success_rate': 0.0,
                'settings_history': [],  # Legacy, kept for backward compat
                'experiences': [],       # NEW: HER unified experience storage
                'best_settings': {},
            }
        
        class_data = self.model['footage_classes'][footage_class]
        
        # Ensure experiences array exists (migration)
        if 'experiences' not in class_data:
            class_data['experiences'] = []
        
        class_data['sample_count'] += 1
        
        # Extract metrics
        success = session_data.get('success', False)
        solve_error = session_data.get('solve_error', 999.0)
        bundle_count = session_data.get('bundle_count', 0)
        settings = session_data.get('settings', {})
        failure_type = session_data.get('failure_type', None)
        
        # Compute HER reward (works for both success and failure)
        reward = self.compute_reward(success, solve_error, bundle_count)
        
        # Determine outcome category
        if success:
            outcome = 'SUCCESS'
            class_data['success_count'] += 1
        elif reward > 0.3:
            outcome = 'PARTIAL'  # Got some progress but didn't fully succeed
        else:
            outcome = 'FAILURE'
        
        # ═══════════════════════════════════════════════════════════════════
        # HER: Store ALL sessions as experiences (not just successes!)
        # ═══════════════════════════════════════════════════════════════════
        experience = {
            'settings': settings,
            'outcome': outcome,
            'reward': reward,
            'solve_error': solve_error,
            'bundle_count': bundle_count,
            'failure_type': failure_type,
            'success_rate': session_data.get('successful_tracks', 0) / 
                           max(session_data.get('total_tracks', 1), 1),
        }
        
        class_data['experiences'].append(experience)
        
        # Keep only last 50 experiences (more than before to capture failures)
        class_data['experiences'] = class_data['experiences'][-50:]
        
        # Also update legacy settings_history for backward compatibility
        if success:
            class_data['settings_history'].append({
                'settings': settings,
                'solve_error': solve_error,
                'success_rate': experience['success_rate'],
            })
            class_data['settings_history'] = class_data['settings_history'][-20:]
            
            # Update best settings
            if class_data['settings_history']:
                best = min(class_data['settings_history'], 
                          key=lambda x: x.get('solve_error', 999))
                class_data['best_settings'] = best.get('settings', {})
        
        # Update success rate
        class_data['avg_success_rate'] = (
            class_data['success_count'] / class_data['sample_count']
        )
        
        # Update global stats
        if 'global_stats' not in self.model:
            self.model['global_stats'] = {
                'total_sessions': 0,
                'successful_sessions': 0,
            }
        self.model['global_stats']['total_sessions'] += 1
        if success:
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
        """
        Get learned settings for a footage class.
        
        BUG 2 FIX: Check experiences (HER) and best_settings, not just settings_history.
        """
        class_data = self.model.get('footage_classes', {}).get(footage_class, {})
        
        # Check for HER experiences or legacy settings_history
        if class_data.get('experiences') or class_data.get('settings_history'):
            # Use weighted prediction logic (handles both arrays)
            return self._predict_from_history(class_data, False)
        
        # Fall back to best_settings (for pretrained model - BUG 3 FIX)
        if class_data.get('best_settings'):
            return class_data['best_settings'].copy()
        
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
                            error: float = 999.0, region_stats: Dict = None,
                            bundle_count: int = 0, failure_type: str = None):
        """Update from session data (LocalLearningModel compat)."""
        
        # Extract total and successful tracks from region_stats
        # region_stats = {region: {total_tracks: N, successful_tracks: N}, ...}
        region_stats = region_stats or {}
        total_tracks = 0
        successful_tracks = 0
        for region, stats in region_stats.items():
            total_tracks += stats.get('total', stats.get('total_tracks', 0))
            successful_tracks += stats.get('success', stats.get('successful_tracks', 0))
        
        session_data = {
            'footage_class': footage_class,
            'success': success,
            'settings': settings,
            'solve_error': error,
            'region_stats': region_stats,
            'bundle_count': bundle_count,
            'failure_type': failure_type,
            # FIX: Include track counts for success_rate calculation
            'total_tracks': total_tracks,
            'successful_tracks': successful_tracks,
        }
        self.update_model(session_data)
    
    def save(self):
        """Save model (LocalLearningModel compat)."""
        self._save_model()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BEHAVIOR LEARNING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def learn_from_behavior(self, footage_class: str, behavior_data: Dict):
        """
        Learn from user behavior patterns.
        
        Captures what settings users adjust, which tracks they delete,
        and uses this to improve future predictions.
        
        Only applies adjustments when:
        - 3+ similar behaviors observed
        - Confidence >= 0.7 (improvements were consistent)
        
        Args:
            footage_class: Footage classification (e.g., 'HD_30fps')
            behavior_data: BehaviorData dict from BehaviorRecorder
        """
        if not behavior_data:
            return
        
        # Ensure behavior_patterns exists in model
        if 'behavior_patterns' not in self.model:
            self.model['behavior_patterns'] = {}
        
        # Learn from settings adjustments
        settings_adj = behavior_data.get('settings_adjustments', {})
        re_solve = behavior_data.get('re_solve', {})
        
        for setting_name, adjustment in settings_adj.items():
            # Key: footage_class:setting_name:direction
            direction = 'increase' if adjustment.get('delta', 0) > 0 else 'decrease'
            key = f"{footage_class}:{setting_name}_{direction}"
            
            if key not in self.model['behavior_patterns']:
                self.model['behavior_patterns'][key] = {
                    'count': 0,
                    'improvements': [],
                    'avg_improvement': 0.0,
                    'confidence': 0.0,
                    'avg_delta': 0.0,
                }
            
            pattern = self.model['behavior_patterns'][key]
            pattern['count'] += 1
            
            # Track if this adjustment helped
            if re_solve.get('attempted') and re_solve.get('improved'):
                improvement = re_solve.get('improvement', 0)
                pattern['improvements'].append(improvement)
                pattern['improvements'] = pattern['improvements'][-10:]  # Keep last 10
                pattern['avg_improvement'] = sum(pattern['improvements']) / len(pattern['improvements'])
            
            # Update average delta
            delta = adjustment.get('delta', 0)
            old_avg = pattern['avg_delta']
            pattern['avg_delta'] = (old_avg * (pattern['count'] - 1) + delta) / pattern['count']
            
            # Compute confidence (% of times this adjustment helped)
            if pattern['improvements']:
                positive = sum(1 for i in pattern['improvements'] if i > 0)
                pattern['confidence'] = positive / len(pattern['improvements'])
            
            print(f"AutoSolve: Learned behavior pattern {key} "
                  f"(count={pattern['count']}, confidence={pattern['confidence']:.2f})")
        
        # Learn from deletions (reduce region confidence)
        deletions = behavior_data.get('track_deletions', [])
        for deletion in deletions:
            region = deletion.get('region', 'unknown')
            key = f"{footage_class}:region_{region}_penalty"
            
            if key not in self.model['behavior_patterns']:
                self.model['behavior_patterns'][key] = {
                    'count': 0,
                    'reasons': {},
                }
            
            pattern = self.model['behavior_patterns'][key]
            pattern['count'] += 1
            
            reason = deletion.get('inferred_reason', 'unknown')
            pattern['reasons'][reason] = pattern['reasons'].get(reason, 0) + 1
        
        self._save_model()
    
    def get_behavior_adjustment(self, footage_class: str, setting_name: str) -> Optional[float]:
        """
        Get learned adjustment for a setting based on user behavior.
        
        Only returns adjustment if confidence threshold is met.
        
        Returns:
            Delta to apply to setting, or None if no confident adjustment
        """
        if 'behavior_patterns' not in self.model:
            return None
        
        # Check for increase pattern
        inc_key = f"{footage_class}:{setting_name}_increase"
        dec_key = f"{footage_class}:{setting_name}_decrease"
        
        inc_pattern = self.model['behavior_patterns'].get(inc_key, {})
        dec_pattern = self.model['behavior_patterns'].get(dec_key, {})
        
        # Need 3+ observations and 0.7+ confidence
        if inc_pattern.get('count', 0) >= 3 and inc_pattern.get('confidence', 0) >= 0.7:
            print(f"AutoSolve: Applying learned adjustment for {setting_name} (increase)")
            return inc_pattern.get('avg_delta', 0)
        
        if dec_pattern.get('count', 0) >= 3 and dec_pattern.get('confidence', 0) >= 0.7:
            print(f"AutoSolve: Applying learned adjustment for {setting_name} (decrease)")
            return dec_pattern.get('avg_delta', 0)
        
        return None
    
    def get_region_penalty(self, footage_class: str, region: str) -> float:
        """
        Get penalty factor for a region based on user deletion patterns.
        
        Returns:
            Penalty factor 0.0-1.0 (lower = more deletions = avoid this region)
        """
        if 'behavior_patterns' not in self.model:
            return 1.0
        
        key = f"{footage_class}:region_{region}_penalty"
        pattern = self.model['behavior_patterns'].get(key, {})
        
        count = pattern.get('count', 0)
        if count == 0:
            return 1.0
        
        # More deletions = lower score
        # 10+ deletions = 0.5 penalty
        penalty = max(0.5, 1.0 - (count / 20))
        return penalty
