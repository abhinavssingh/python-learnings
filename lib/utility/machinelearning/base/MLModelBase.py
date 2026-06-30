from abc import ABC, abstractmethod
from typing import Any


class MLModelBase(ABC):

    @abstractmethod
    def build_pipeline(self, preprocessor: Any, extra_steps: Any = None) -> None:
        pass

    @abstractmethod
    def train(self, X, y) -> None:
        pass

    @abstractmethod
    def predict(self, X):
        pass
