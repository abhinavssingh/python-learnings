from abc import ABC, abstractmethod

from .document import SearchResult


class BaseRetriever(ABC):
    """Returns the most relevant documents for a query."""

    name: str = "retriever"

    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> list[SearchResult]:
        pass

    def __call__(self, query: str, k: int = 4) -> list[SearchResult]:
        return self.retrieve(query, k)
