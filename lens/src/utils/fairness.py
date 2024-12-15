from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import confusion_matrix

class FairnessAnalyzer:
    """Analyze model fairness and bias across sensitive attributes."""
    
    def __init__(self, sensitive_features: List[str]):
        self.sensitive_features = sensitive_features
        
    def analyze_fairness(self, X: np.ndarray, y_true: np.ndarray, 
                        y_pred: np.ndarray, 
                        feature_names: List[str]) -> Dict[str, Any]:
        """Compute various fairness metrics across sensitive attributes."""
        results = {}
        
        for feature in self.sensitive_features:
            feature_idx = feature_names.index(feature)
            groups = np.unique(X[:, feature_idx])
            
            group_metrics = {}
            for group in groups:
                mask = X[:, feature_idx] == group
                group_metrics[str(group)] = self._compute_group_metrics(
                    y_true[mask], 
                    y_pred[mask]
                )
                
            results[feature] = {
                'group_metrics': group_metrics,
                'disparate_impact': self._compute_disparate_impact(group_metrics),
                'equal_opportunity': self._compute_equal_opportunity(group_metrics),
                'demographic_parity': self._compute_demographic_parity(group_metrics)
            }
            
        return results
        
    def _compute_group_metrics(self, y_true: np.ndarray, 
                             y_pred: np.ndarray) -> Dict[str, float]:
        """Compute metrics for a specific group."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'selection_rate': (tp + fp) / (tp + tn + fp + fn)
        }
        
    def _compute_disparate_impact(self, group_metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate disparate impact ratio."""
        selection_rates = [m['selection_rate'] for m in group_metrics.values()]
        return min(selection_rates) / max(selection_rates) if max(selection_rates) > 0 else 0
        
    def _compute_equal_opportunity(self, group_metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate equal opportunity difference."""
        recalls = [m['recall'] for m in group_metrics.values()]
        return max(recalls) - min(recalls)
        
    def _compute_demographic_parity(self, group_metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate demographic parity difference."""
        selection_rates = [m['selection_rate'] for m in group_metrics.values()]
        return max(selection_rates) - min(selection_rates) 