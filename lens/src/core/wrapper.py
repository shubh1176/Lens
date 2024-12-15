from typing import Any, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator
import torch
import tensorflow as tf

class ModelWrapper:
    """Universal wrapper for different types of ML models."""
    
    def __init__(self, model: Any):
        self.model = model
        self.model_type = self._detect_model_type()
        
    def _detect_model_type(self) -> str:
        """Detect the type of model (sklearn, pytorch, tensorflow, etc.)."""
        if isinstance(self.model, BaseEstimator):
            return 'sklearn'
        elif isinstance(self.model, torch.nn.Module):
            return 'pytorch'
        elif isinstance(self.model, tf.keras.Model):
            return 'tensorflow'
        else:
            raise ValueError("Unsupported model type")
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Unified prediction interface."""
        if self.model_type == 'sklearn':
            return self.model.predict(X)
        elif self.model_type == 'pytorch':
            with torch.no_grad():
                return self.model(torch.FloatTensor(X)).numpy()
        elif self.model_type == 'tensorflow':
            return self.model.predict(X)