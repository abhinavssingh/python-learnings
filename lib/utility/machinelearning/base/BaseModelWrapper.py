from abc import ABC, abstractmethod
from typing import Any, Dict

from sklearn.pipeline import Pipeline


class BaseModelWrapper(ABC):

    def __init__(self, model: Any):
        self.model = model
        self.pipeline: Pipeline | None = None   # ✅ typed properly

    @abstractmethod
    def build_pipeline(self, preprocessor: Any) -> None:
        """Attach preprocessor + model into pipeline"""
        pass

    @abstractmethod
    def evaluate(self, y_true, y_pred) -> Dict[str, float]:
        """Model-specific evaluation metrics"""
        pass

    # ✅ NEW: Safe accessor
    def get_pipeline(self) -> Pipeline:
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline first.")
        return self.pipeline

    def train(self, X, y) -> None:
        pipeline = self.get_pipeline()   # ✅ safer
        pipeline.fit(X, y)

    def predict(self, X):
        pipeline = self.get_pipeline()   # ✅ safer
        return pipeline.predict(X)
