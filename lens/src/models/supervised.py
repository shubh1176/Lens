from typing import Any, Dict, Optional, Union, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ..core.wrapper import ModelWrapper

class SupervisedModel:
    """Wrapper for supervised learning models with unified interface."""
    
    def __init__(self, model: Any):
        self.model_wrapper = ModelWrapper(model)
        self.metrics_history: Dict[str, List[float]] = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[tuple] = None) -> Dict[str, float]:
        """Fit the model and track metrics."""
        if hasattr(self.model_wrapper.model, 'fit'):
            self.model_wrapper.model.fit(X, y)
            
        metrics = self._compute_metrics(X, y)
        
        if validation_data:
            val_metrics = self._compute_metrics(
                validation_data[0], 
                validation_data[1], 
                prefix='val_'
            )
            metrics.update(val_metrics)
            
        self._update_metrics_history(metrics)
        return metrics
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the wrapped model."""
        return self.model_wrapper.predict(X)
        
    def _compute_metrics(self, X: np.ndarray, y: np.ndarray, 
                        prefix: str = '') -> Dict[str, float]:
        """Compute standard classification metrics."""
        y_pred = self.predict(X)
        
        return {
            f'{prefix}accuracy': accuracy_score(y, y_pred),
            f'{prefix}precision': precision_score(y, y_pred, average='weighted'),
            f'{prefix}recall': recall_score(y, y_pred, average='weighted'),
            f'{prefix}f1': f1_score(y, y_pred, average='weighted')
        }
        
    def _update_metrics_history(self, metrics: Dict[str, float]) -> None:
        """Update metrics history for tracking model performance."""
        for metric, value in metrics.items():
            if metric in self.metrics_history:
                self.metrics_history[metric].append(value)
