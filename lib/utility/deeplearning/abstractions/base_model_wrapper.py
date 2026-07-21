from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TModel = TypeVar("TModel")


class BaseModelWrapper(Generic[TModel], ABC):

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Friendly model name.
        """
        pass

    @abstractmethod
    def build_model(self) -> TModel:
        """
        Build and return model.
        """
        pass

    @abstractmethod
    def get_model(self) -> TModel:
        """
        Return underlying framework model.
        """
        pass

    @abstractmethod
    def summary(self):
        """
        Display model summary.
        """
        pass
