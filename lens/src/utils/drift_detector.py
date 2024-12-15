import numpy as np
from typing import Dict, Any, Optional, List
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

class DriftDetector:
    """Detect and analyze model and data drift over time."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.reference_data = None
        self.reference_predictions = None
        
    def set_reference(self, X: np.ndarray, predictions: np.ndarray) -> None:
        """Set reference data and predictions for drift comparison."""
        self.reference_data = X
        self.reference_predictions = predictions
        
    def detect_drift(self, X: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """Detect both feature and prediction drift."""
        feature_drift = self._detect_feature_drift(X)
        prediction_drift = self._detect_prediction_drift(predictions)
        
        return {
            'feature_drift': feature_drift,
            'prediction_drift': prediction_drift,
            'overall_drift_score': self._compute_overall_drift(feature_drift, prediction_drift)
        }
        
    def _detect_feature_drift(self, X: np.ndarray) -> Dict[str, float]:
        """Detect drift in feature distributions using KS test."""
        drift_scores = {}
        
        for i in range(X.shape[1]):
            statistic, p_value = ks_2samp(
                self.reference_data[:, i],
                X[:, i]
            )
            drift_scores[f'feature_{i}'] = {
                'statistic': statistic,
                'p_value': p_value
            }
            
        return drift_scores
        
    def _detect_prediction_drift(self, predictions: np.ndarray) -> Dict[str, float]:
        """Detect drift in model predictions using Jensen-Shannon divergence."""
        ref_hist, _ = np.histogram(self.reference_predictions, bins=20, density=True)
        curr_hist, _ = np.histogram(predictions, bins=20, density=True)
        
        js_divergence = jensenshannon(ref_hist, curr_hist)
        
        return {
            'js_divergence': js_divergence,
            'significant_drift': js_divergence > 0.1  # threshold can be adjusted
        }
        
    def _compute_overall_drift(self, 
                             feature_drift: Dict[str, float], 
                             prediction_drift: Dict[str, float]) -> float:
        """Compute overall drift score combining feature and prediction drift."""
        feature_scores = [v['statistic'] for v in feature_drift.values()]
        return np.mean(feature_scores) * 0.5 + prediction_drift['js_divergence'] * 0.5 