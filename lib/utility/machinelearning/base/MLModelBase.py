from abc import ABC, abstractmethod
from typing import Any


class MLModelBase(ABC):

    @abstractmethod
    def build_pipeline(self, preprocessor: Any) -> None:
        pass

    @abstractmethod
    def train(self, X, y) -> None:
        pass

    @abstractmethod
    def predict(self, X):
        pass
