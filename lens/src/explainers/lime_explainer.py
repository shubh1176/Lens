import lime
import lime.lime_tabular
from typing import Dict, Any, Optional
import numpy as np
from ..core.base import LensBase

class LimeExplainer(LensBase):
    """LIME-based model explainer."""
    
    def __init__(self):
        super().__init__()
        self.explainer = None
        
    def fit(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LimeExplainer':
        """Initialize LIME explainer with training data."""
        self.model = model
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=self.feature_names,
            class_names=self.target_names,
            mode='classification' if len(self.target_names) > 0 else 'regression'
        )
        return self
        
    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        """Generate LIME explanations for the given instances."""
        explanations = []
        for instance in X:
            exp = self.explainer.explain_instance(
                instance, 
                self.model.predict,
                num_features=len(self.feature_names)
            )
            explanations.append(exp)
            
        return {
            'explanations': explanations,
            'feature_importance': self._aggregate_importance(explanations)
        }
        
    def _aggregate_importance(self, explanations):
        """Aggregate feature importance across multiple explanations."""
        importance_dict = {}
        for exp in explanations:
            for feature, importance in exp.as_list():
                if feature not in importance_dict:
                    importance_dict[feature] = []
                importance_dict[feature].append(abs(importance))
                
        return {k: np.mean(v) for k, v in importance_dict.items()}