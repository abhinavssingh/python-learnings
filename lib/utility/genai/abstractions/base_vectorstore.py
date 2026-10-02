from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .base_embedder import BaseEmbedder
from .document import Document, SearchResult

MetadataFilter = dict[str, Any] | Callable[[dict[str, Any]], bool] | None


class BaseVectorStore(ABC):
    """Stores document embeddings and finds the nearest ones to a query."""

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder

    @abstractmethod
    def add_documents(self, documents: list[Document], embeddings: np.ndarray | None = None) -> None:
        pass

    @abstractmethod
    def similarity_search_by_vector(
        self, vector: np.ndarray, k: int = 4, filter: MetadataFilter = None
    ) -> list[SearchResult]:
        pass

    @abstractmethod
    def save(self, directory: str | Path) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def similarity_search(self, query: str, k: int = 4, filter: MetadataFilter = None) -> list[SearchResult]:
        return self.similarity_search_by_vector(self.embedder.embed_query(query), k=k, filter=filter)
