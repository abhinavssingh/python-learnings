import json
from pathlib import Path

import numpy as np

from ..abstractions.base_embedder import BaseEmbedder
from ..abstractions.base_vectorstore import BaseVectorStore, MetadataFilter
from ..abstractions.document import Document, SearchResult
from .filters import matches_filter


class InMemoryVectorStore(BaseVectorStore):
    """
    The simplest possible vector database: a NumPy matrix + brute-force search.

    metric="cosine"    -> normalised dot product (default)
    metric="dot"       -> raw dot product
    metric="euclidean" -> negative L2 distance (so higher is still better)
    """

    def __init__(self, embedder: BaseEmbedder, metric: str = "cosine"):
        super().__init__(embedder)
        self.metric = metric
        self.documents: list[Document] = []
        self.vectors: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.documents)

    def add_documents(self, documents: list[Document], embeddings: np.ndarray | None = None) -> None:
        if not documents:
            return
        if embeddings is None:
            embeddings = self.embedder.embed_documents([d.page_content for d in documents])
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.documents.extend(documents)
        self.vectors = embeddings if self.vectors is None else np.vstack([self.vectors, embeddings])

    def _scores(self, vector: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            matrix = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True).clip(min=1e-12)
            return matrix @ (vector / max(np.linalg.norm(vector), 1e-12))
        if self.metric == "dot":
            return self.vectors @ vector
        if self.metric == "euclidean":
            return -np.linalg.norm(self.vectors - vector, axis=1)
        raise ValueError(f"Unknown metric '{self.metric}'")

    def similarity_search_by_vector(self, vector: np.ndarray, k: int = 4, filter: MetadataFilter = None) -> list[SearchResult]:
        if self.vectors is None:
            return []
        scores = self._scores(np.asarray(vector, dtype=np.float32))
        allowed = np.array([matches_filter(d.metadata, filter) for d in self.documents])
        scores = np.where(allowed, scores, -np.inf)
        order = [i for i in np.argsort(scores)[::-1] if np.isfinite(scores[i])][:k]
        return [SearchResult(self.documents[i], float(scores[i]), self.vectors[i], "dense") for i in order]

    def delete(self, filter: MetadataFilter) -> int:
        keep = [i for i, d in enumerate(self.documents) if not matches_filter(d.metadata, filter)]
        removed = len(self.documents) - len(keep)
        self.documents = [self.documents[i] for i in keep]
        self.vectors = self.vectors[keep] if self.vectors is not None and keep else None
        return removed

    # ------------------------------------------------------
    # Persistence
    # ------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors if self.vectors is not None else np.zeros((0, 0)))
        payload = {"metric": self.metric, "documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in self.documents]}
        (directory / "documents.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder: BaseEmbedder) -> "InMemoryVectorStore":
        directory = Path(directory)
        payload = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
        store = cls(embedder, metric=payload.get("metric", "cosine"))
        store.documents = [Document(**d) for d in payload["documents"]]
        vectors = np.load(directory / "vectors.npy")
        store.vectors = vectors if vectors.size else None
        return store
