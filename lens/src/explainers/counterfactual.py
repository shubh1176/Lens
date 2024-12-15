import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from ..core.base import LensBase
from scipy.optimize import minimize

class CounterfactualExplainer(LensBase):
    """Generate counterfactual explanations for model predictions."""
    
    def __init__(self, feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        super().__init__()
        self.feature_ranges = feature_ranges
        self.epsilon = 1e-4
        
    def fit(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'CounterfactualExplainer':
        """Initialize the explainer with model and training data."""
        self.model = model
        if self.feature_ranges is None:
            self._compute_feature_ranges(X)
        return self
        
    def explain(self, X: np.ndarray, desired_outcome: Any) -> Dict[str, Any]:
        """Generate counterfactual explanations for instances."""
        counterfactuals = []
        distances = []
        
        for instance in X:
            cf, dist = self._find_counterfactual(
                instance, 
                desired_outcome
            )
            counterfactuals.append(cf)
            distances.append(dist)
            
        return {
            'counterfactuals': np.array(counterfactuals),
            'distances': np.array(distances),
            'feature_changes': self._compute_feature_changes(X, counterfactuals)
        }
        
    def _find_counterfactual(self, instance: np.ndarray, desired_outcome: Any) -> Tuple[np.ndarray, float]:
        """Find the closest counterfactual example."""
        def objective(x):
            pred = self.model.predict(x.reshape(1, -1))[0]
            pred_diff = np.abs(pred - desired_outcome)
            distance = np.linalg.norm(x - instance)
            return pred_diff + self.epsilon * distance
            
        bounds = [(self.feature_ranges[f][0], self.feature_ranges[f][1]) 
                 for f in self.feature_names]
                 
        result = minimize(
            objective,
            instance,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        return result.x, result.fun
        
    def _compute_feature_ranges(self, X: np.ndarray) -> None:
        """Compute min/max ranges for each feature."""
        self.feature_ranges = {}
        for i, feature in enumerate(self.feature_names):
            self.feature_ranges[feature] = (
                np.min(X[:, i]),
                np.max(X[:, i])
            )
            
    def _compute_feature_changes(self, 
                               original: np.ndarray, 
                               counterfactuals: List[np.ndarray]) -> Dict[str, List[float]]:
        """Compute changes in feature values for counterfactuals."""
        changes = {feature: [] for feature in self.feature_names}
        
        for orig, cf in zip(original, counterfactuals):
            for i, feature in enumerate(self.feature_names):
                changes[feature].append(cf[i] - orig[i])
                
        return changes
