from abc import ABC, abstractmethod
from typing import Any


from lib.utility.machinelearning._logging import ExceptionLoggingMixin


class MLModelBase(ExceptionLoggingMixin, ABC):

    @abstractmethod
    def build_pipeline(self, preprocessor: Any, extra_steps: Any = None) -> None:
        pass

    @abstractmethod
    def train(self, X, y) -> None:
        pass

    @abstractmethod
    def predict(self, X):
        pass

