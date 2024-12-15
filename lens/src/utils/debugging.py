from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

class ModelDebugger:
    """Tools for model debugging and error analysis."""
    
    def __init__(self):
        self.model = None
        self.X = None
        self.y = None
        
    def analyze_errors(self, model: Any, X: np.ndarray, y: np.ndarray, 
                      y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze model errors and identify patterns."""
        self.model = model
        self.X = X
        self.y = y
        
        # Get misclassified examples
        incorrect_mask = y != y_pred
        X_incorrect = X[incorrect_mask]
        y_incorrect = y[incorrect_mask]
        y_pred_incorrect = y_pred[incorrect_mask]
        
        # Cluster misclassified examples
        error_clusters = self._cluster_errors(X_incorrect)
        
        return {
            'confusion_matrix': confusion_matrix(y, y_pred),
            'error_indices': np.where(incorrect_mask)[0],
            'error_clusters': error_clusters,
            'error_statistics': self._compute_error_statistics(y, y_pred),
            'edge_cases': self._identify_edge_cases(X, y, y_pred)
        }
        
    def _cluster_errors(self, X_incorrect: np.ndarray) -> Dict[str, Any]:
        """Cluster misclassified examples to identify patterns."""
        if len(X_incorrect) < 2:
            return {}
            
        kmeans = KMeans(n_clusters=min(5, len(X_incorrect)))
        clusters = kmeans.fit_predict(X_incorrect)
        
        return {
            'cluster_labels': clusters,
            'centroids': kmeans.cluster_centers_,
            'cluster_sizes': np.bincount(clusters)
        }
        
    def _compute_error_statistics(self, y_true: np.ndarray, 
                                y_pred: np.ndarray) -> Dict[str, float]:
        """Compute statistics about model errors."""
        errors = y_true != y_pred
        
        return {
            'error_rate': errors.mean(),
            'error_std': errors.std(),
            'consecutive_errors': self._count_consecutive_errors(errors)
        }
        
    def _identify_edge_cases(self, X: np.ndarray, y_true: np.ndarray, 
                           y_pred: np.ndarray) -> Dict[str, List[int]]:
        """Identify potential edge cases in the dataset."""
        # Find samples with high prediction uncertainty
        proba = getattr(self.model, "predict_proba", None)
        edge_cases = {}
        
        if proba is not None:
            probabilities = proba(X)
            max_probs = np.max(probabilities, axis=1)
            edge_cases['uncertain_predictions'] = np.where(max_probs < 0.6)[0]
            
        # Find outliers in feature space
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.1)
        outliers = iso_forest.fit_predict(X) == -1
        edge_cases['feature_outliers'] = np.where(outliers)[0]
        
        return edge_cases
        
    def _count_consecutive_errors(self, errors: np.ndarray) -> int:
        """Count the maximum number of consecutive errors."""
        max_consecutive = 0
        current_consecutive = 0
        
        for error in errors:
            if error:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive 