from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List, Optional, Union

class LensBase(ABC):
    """Base class for Lens explainability framework."""
    
    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.target_names: List[str] = []
        
    @abstractmethod
    def fit(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LensBase':
        """Fit the explainer to a model and dataset."""
        pass
    
    @abstractmethod
    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        """Generate explanations for the given instances."""
        pass
    
    def set_feature_names(self, names: List[str]) -> None:
        """Set feature names for better interpretability."""
        self.feature_names = names
        
    def set_target_names(self, names: List[str]) -> None:
        """Set target names for classification tasks."""
        self.target_names = names 