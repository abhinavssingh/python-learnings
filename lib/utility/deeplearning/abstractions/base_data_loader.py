from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

TDataset = TypeVar("TDataset")


class BaseDataLoader(ABC, Generic[TDataset]):
    @abstractmethod
    def load_data(self) -> tuple[Any, Any]:
        pass

    @abstractmethod
    def preprocess(
        self,
        X: Any,
        y: Any = None,
    ) -> tuple[Any, Any]:
        pass

    @abstractmethod
    def get_dataset(self) -> TDataset:
        pass
