from typing import Any, Dict, Optional, Union, List
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from ..core.wrapper import ModelWrapper

class UnsupervisedModel:
    """Wrapper for unsupervised learning models with unified interface."""
    
    def __init__(self, model: Any, model_type: str = 'clustering'):
        self.model_wrapper = ModelWrapper(model)
        self.model_type = model_type  # 'clustering' or 'dimensionality_reduction'
        self.metrics_history: Dict[str, List[float]] = {
            'silhouette_score': [],
            'calinski_harabasz_score': [],
            'inertia': [] if hasattr(model, 'inertia_') else None
        }
        self.n_components = getattr(model, 'n_components', None)
        
    def fit(self, X: np.ndarray, validation_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Fit the model and track metrics."""
        if hasattr(self.model_wrapper.model, 'fit'):
            self.model_wrapper.model.fit(X)
            
        metrics = self._compute_metrics(X)
        
        if validation_data is not None:
            val_metrics = self._compute_metrics(validation_data, prefix='val_')
            metrics.update(val_metrics)
            
        self._update_metrics_history(metrics)
        return metrics
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using the wrapped model."""
        if hasattr(self.model_wrapper.model, 'transform'):
            return self.model_wrapper.model.transform(X)
        return self.predict(X)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (cluster assignments) using the wrapped model."""
        return self.model_wrapper.predict(X)
        
    def _compute_metrics(self, X: np.ndarray, prefix: str = '') -> Dict[str, float]:
        """Compute standard unsupervised learning metrics."""
        metrics = {}
        
        if self.model_type == 'clustering':
            predictions = self.predict(X)
            
            # Compute clustering metrics
            try:
                metrics[f'{prefix}silhouette'] = silhouette_score(X, predictions)
                metrics[f'{prefix}calinski_harabasz'] = calinski_harabasz_score(X, predictions)
            except ValueError:
                # Handle cases where metrics cannot be computed
                metrics[f'{prefix}silhouette'] = 0.0
                metrics[f'{prefix}calinski_harabasz'] = 0.0
                
            # Add inertia if available (K-means specific)
            if hasattr(self.model_wrapper.model, 'inertia_'):
                metrics[f'{prefix}inertia'] = self.model_wrapper.model.inertia_
                
        elif self.model_type == 'dimensionality_reduction':
            # Compute reconstruction error if applicable
            if hasattr(self.model_wrapper.model, 'score'):
                metrics[f'{prefix}reconstruction_error'] = -self.model_wrapper.model.score(X)
            
            # Compute explained variance if available
            if hasattr(self.model_wrapper.model, 'explained_variance_ratio_'):
                metrics[f'{prefix}explained_variance'] = sum(
                    self.model_wrapper.model.explained_variance_ratio_
                )
                
        return metrics
        
    def _update_metrics_history(self, metrics: Dict[str, float]) -> None:
        """Update metrics history for tracking model performance."""
        for metric, value in metrics.items():
            if metric in self.metrics_history:
                if self.metrics_history[metric] is None:
                    self.metrics_history[metric] = []
                self.metrics_history[metric].append(value)
