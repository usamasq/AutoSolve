# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
TrackQualityPredictor - Predict track quality before solve.

Uses learned patterns to identify tracks likely to have high reprojection
error, enabling filtering BEFORE the solver runs.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import bpy


@dataclass
class TrackPrediction:
    """Prediction result for a single track."""
    track_name: str
    quality_score: float  # 0-1, higher = better predicted quality
    predicted_error: float  # Estimated reprojection error
    confidence: float  # How confident is this prediction
    risk_factors: List[str]  # Why this track might be bad


class TrackQualityPredictor:
    """
    Predict track quality before solve using learned patterns.
    
    Uses historical data to identify tracks likely to degrade solve quality.
    Can filter bad tracks BEFORE running the expensive solve operation.
    """
    
    # Default thresholds learned from data analysis
    DEFAULT_THRESHOLDS = {
        'min_lifespan': 20,  # Tracks shorter than this are risky
        'max_jitter': 3.0,  # Tracks with jitter above this are risky
        'max_velocity': 0.01,  # Unusually fast tracks are risky
        'error_threshold': 0.5,  # Predicted error above this = low quality
    }
    
    # Region reliability from your data (mid-left, center, bottom are best)
    REGION_RELIABILITY = {
        'mid-left': 1.0,
        'center': 1.0,
        'bottom-left': 1.0,
        'bottom-center': 1.0,
        'bottom-right': 1.0,
        'top-center': 0.97,
        'top-left': 0.97,
        'top-right': 0.96,
        'mid-right': 0.96,
    }
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(bpy.utils.user_resource('DATAFILES')) / 'autosolve'
        
        self.model_path = self.data_dir / 'track_quality_model.json'
        self.model = self._load_model()
    
    def _load_model(self) -> Dict:
        """Load or initialize the track quality model."""
        if self.model_path.exists():
            try:
                with open(self.model_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"AutoSolve: Error loading track quality model: {e}")
        
        # Initialize with defaults
        return {
            'version': 1,
            'thresholds': self.DEFAULT_THRESHOLDS.copy(),
            'region_reliability': self.REGION_RELIABILITY.copy(),
            'learned_correlations': {
                'lifespan_vs_error': -0.3,  # Longer lifespan = lower error
                'jitter_vs_error': 0.5,  # Higher jitter = higher error
                'velocity_vs_error': 0.2,  # Higher velocity = slightly higher error
            },
            'training_samples': 0,
        }
    
    def _save_model(self):
        """Save model to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, 'w') as f:
            json.dump(self.model, f, indent=2)
    
    def predict_track_quality(self, track_features: Dict) -> TrackPrediction:
        """
        Predict quality of a single track before solve.
        
        Args:
            track_features: Dict with keys:
                - lifespan: int
                - jitter_score: float
                - avg_velocity: float
                - region: str
                
        Returns:
            TrackPrediction with quality score, predicted error, and risk factors
        """
        thresholds = self.model.get('thresholds', self.DEFAULT_THRESHOLDS)
        region_reliability = self.model.get('region_reliability', self.REGION_RELIABILITY)
        
        lifespan = track_features.get('lifespan', 0)
        jitter = track_features.get('jitter_score', 0)
        velocity = track_features.get('avg_velocity', 0)
        region = track_features.get('region', 'center')
        
        # Start with base quality from region
        quality = region_reliability.get(region, 0.9)
        risk_factors = []
        
        # Penalize short tracks
        if lifespan < thresholds['min_lifespan']:
            penalty = (thresholds['min_lifespan'] - lifespan) / thresholds['min_lifespan']
            quality *= (1 - penalty * 0.5)
            risk_factors.append(f"Short lifespan ({lifespan} frames)")
        
        # Penalize high jitter
        if jitter > thresholds['max_jitter']:
            penalty = min(1.0, (jitter - thresholds['max_jitter']) / thresholds['max_jitter'])
            quality *= (1 - penalty * 0.4)
            risk_factors.append(f"High jitter ({jitter:.2f})")
        
        # Penalize unusually high velocity
        if velocity > thresholds['max_velocity']:
            penalty = min(1.0, (velocity - thresholds['max_velocity']) / thresholds['max_velocity'])
            quality *= (1 - penalty * 0.3)
            risk_factors.append(f"High velocity ({velocity:.4f})")
        
        # Bonus for very long tracks
        if lifespan > 100:
            quality = min(1.0, quality * 1.1)
        
        # Estimate error from quality (inverse relationship)
        # Based on your data: good tracks have ~0.1-0.4 error, bad tracks have 1.0+
        predicted_error = 0.1 + (1 - quality) * 2.0
        
        # Confidence based on how extreme the features are
        confidence = 0.5 + min(0.4, len(risk_factors) * 0.15)
        
        return TrackPrediction(
            track_name=track_features.get('name', 'unknown'),
            quality_score=round(quality, 3),
            predicted_error=round(predicted_error, 3),
            confidence=round(confidence, 3),
            risk_factors=risk_factors
        )
    
    def analyze_all_tracks(self, tracking) -> List[TrackPrediction]:
        """
        Analyze all tracks in a tracking object.
        
        Args:
            tracking: Blender tracking object
            
        Returns:
            List of TrackPrediction for each track, sorted by quality (worst first)
        """
        predictions = []
        
        for track in tracking.tracks:
            markers = [m for m in track.markers if not m.mute]
            if len(markers) < 2:
                continue
            
            markers.sort(key=lambda x: x.frame)
            lifespan = markers[-1].frame - markers[0].frame
            
            # Calculate average position for region
            avg_x = sum(m.co.x for m in markers) / len(markers)
            avg_y = sum(m.co.y for m in markers) / len(markers)
            region = self._get_region(avg_x, avg_y)
            
            # Calculate velocity
            from mathutils import Vector
            displacement = (Vector(markers[-1].co) - Vector(markers[0].co)).length
            avg_velocity = displacement / max(lifespan, 1)
            
            # Calculate jitter
            jitter = self._calculate_jitter(markers)
            
            features = {
                'name': track.name,
                'lifespan': lifespan,
                'jitter_score': jitter,
                'avg_velocity': avg_velocity,
                'region': region,
            }
            
            prediction = self.predict_track_quality(features)
            predictions.append(prediction)
        
        # Sort by quality (worst first for easy filtering)
        predictions.sort(key=lambda p: p.quality_score)
        
        return predictions
    
    def filter_low_quality_tracks(self, tracking, quality_threshold: float = 0.4,
                                   min_keep: int = 20) -> Tuple[int, List[str]]:
        """
        Remove tracks predicted to have low quality.
        
        Args:
            tracking: Blender tracking object
            quality_threshold: Remove tracks below this quality score
            min_keep: Always keep at least this many tracks
            
        Returns:
            Tuple of (tracks_removed, list of removed track names)
        """
        predictions = self.analyze_all_tracks(tracking)
        
        if not predictions:
            return 0, []
        
        # Find tracks below threshold
        low_quality = [p for p in predictions if p.quality_score < quality_threshold]
        
        # Ensure we keep minimum tracks
        current_count = len(tracking.tracks)
        max_remove = max(0, current_count - min_keep)
        
        to_remove = low_quality[:max_remove]
        
        if not to_remove:
            return 0, []
        
        # Print what we're filtering
        print(f"AutoSolve: Filtering {len(to_remove)} low-quality tracks (threshold: {quality_threshold})")
        for pred in to_remove[:5]:  # Show first 5
            print(f"  → {pred.track_name}: quality={pred.quality_score:.2f}, "
                  f"predicted_error={pred.predicted_error:.2f} ({', '.join(pred.risk_factors)})")
        
        # Select and delete
        removed_names = [p.track_name for p in to_remove]
        for track in tracking.tracks:
            track.select = track.name in removed_names
        
        try:
            bpy.ops.clip.delete_track()
        except Exception as e:
            print(f"AutoSolve: Error removing tracks: {e}")
            return 0, []
        
        return len(removed_names), removed_names
    
    def learn_from_solve(self, predictions: List[TrackPrediction], 
                         actual_errors: Dict[str, float]):
        """
        Update model based on actual solve results.
        
        Args:
            predictions: Predictions made before solve
            actual_errors: Dict mapping track_name -> actual reprojection error
        """
        if not predictions or not actual_errors:
            return
        
        # Compare predictions to actuals
        correct_predictions = 0
        total = 0
        
        for pred in predictions:
            if pred.track_name in actual_errors:
                actual = actual_errors[pred.track_name]
                
                # Check if prediction direction was correct
                # (low quality predicted → high actual error, and vice versa)
                predicted_bad = pred.quality_score < 0.5
                actually_bad = actual > 0.5
                
                if predicted_bad == actually_bad:
                    correct_predictions += 1
                total += 1
        
        if total > 0:
            accuracy = correct_predictions / total
            print(f"AutoSolve: Track quality prediction accuracy: {accuracy:.1%} ({total} tracks)")
            
            # Update model sample count
            self.model['training_samples'] = self.model.get('training_samples', 0) + total
            self._save_model()
    
    def get_pre_solve_summary(self, tracking) -> Dict:
        """
        Get a summary for pre-solve quality assessment.
        
        Returns:
            Dict with 'good_tracks', 'risky_tracks', 'recommendations'
        """
        predictions = self.analyze_all_tracks(tracking)
        
        good = [p for p in predictions if p.quality_score >= 0.7]
        medium = [p for p in predictions if 0.4 <= p.quality_score < 0.7]
        risky = [p for p in predictions if p.quality_score < 0.4]
        
        recommendations = []
        
        if len(risky) > len(predictions) * 0.3:
            recommendations.append("Many risky tracks - consider filtering before solve")
        
        if len(good) < 15:
            recommendations.append("Few high-quality tracks - solve may have high error")
        
        # Check region distribution in good tracks
        good_regions = set(p.risk_factors for p in good)
        if len(good_regions) < 5:
            recommendations.append("Good tracks concentrated in few regions - add more markers")
        
        return {
            'total_tracks': len(predictions),
            'good_tracks': len(good),
            'medium_tracks': len(medium),
            'risky_tracks': len(risky),
            'average_quality': sum(p.quality_score for p in predictions) / max(len(predictions), 1),
            'recommendations': recommendations,
        }
    
    def _get_region(self, x: float, y: float) -> str:
        """Get region name from normalized coordinates."""
        col = 0 if x < 0.33 else (1 if x < 0.66 else 2)
        row = 2 if y < 0.33 else (1 if y < 0.66 else 0)
        
        regions = [
            ['top-left', 'top-center', 'top-right'],
            ['mid-left', 'center', 'mid-right'],
            ['bottom-left', 'bottom-center', 'bottom-right']
        ]
        return regions[row][col]
    
    def _calculate_jitter(self, markers) -> float:
        """Calculate jitter score from markers."""
        if len(markers) < 3:
            return 0.0
        
        from mathutils import Vector
        velocities = []
        for i in range(1, len(markers)):
            v = (Vector(markers[i].co) - Vector(markers[i-1].co)).length
            velocities.append(v)
        
        if not velocities:
            return 0.0
        
        avg = sum(velocities) / len(velocities)
        if avg == 0:
            return 0.0
        
        variance = sum((v - avg) ** 2 for v in velocities) / len(velocities)
        return (variance ** 0.5) / avg
