from abc import ABC, abstractmethod
from typing import Any


class BaseTrainer(ABC):
    @abstractmethod
    def compile_model(
        self,
        metrics: list[str] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
        X_train: Any,
        y_train: Any,
        validation_data: Any = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        X_test: Any,
        y_test: Any,
    ):
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        X: Any,
    ):
        raise NotImplementedError
