import json
from pathlib import Path

import numpy as np

from ..abstractions.base_embedder import BaseEmbedder
from ..abstractions.base_vectorstore import BaseVectorStore, MetadataFilter
from ..abstractions.document import Document, SearchResult
from .filters import matches_filter

try:
    import faiss
except ImportError:
    faiss = None


class FaissVectorStore(BaseVectorStore):
    """
    Vector store backed by FAISS (Facebook AI Similarity Search).

    index_type="flat" -> exact search (IndexFlatIP on normalised vectors = cosine)
    index_type="hnsw" -> approximate graph-based search (fast on millions of vectors)
    """

    def __init__(self, embedder: BaseEmbedder, index_type: str = "flat", hnsw_m: int = 32):
        if faiss is None:
            raise ImportError("faiss-cpu is not installed. Run `pip install faiss-cpu`.")
        super().__init__(embedder)
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.index = None
        self.documents: list[Document] = []

    def __len__(self) -> int:
        return len(self.documents)

    def _new_index(self, dim: int):
        if self.index_type == "hnsw":
            return faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexFlatIP(dim)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    def add_documents(self, documents: list[Document], embeddings: np.ndarray | None = None) -> None:
        if not documents:
            return
        if embeddings is None:
            embeddings = self.embedder.embed_documents([d.page_content for d in documents])
        vectors = self._normalize(np.atleast_2d(embeddings))
        if self.index is None:
            self.index = self._new_index(vectors.shape[1])
        self.index.add(vectors)
        self.documents.extend(documents)

    def similarity_search_by_vector(self, vector: np.ndarray, k: int = 4, filter: MetadataFilter = None) -> list[SearchResult]:
        if self.index is None:
            return []
        query = self._normalize(np.atleast_2d(vector))
        # Over-fetch when filtering, because FAISS filters after the search.
        fetch = len(self.documents) if filter is not None else min(k, len(self.documents))
        scores, ids = self.index.search(query, fetch)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or not matches_filter(self.documents[idx].metadata, filter):
                continue
            results.append(SearchResult(self.documents[idx], float(score), self.index.reconstruct(int(idx)), "faiss"))
            if len(results) == k:
                break
        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        payload = {"index_type": self.index_type, "documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in self.documents]}
        (directory / "documents.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder: BaseEmbedder) -> "FaissVectorStore":
        directory = Path(directory)
        payload = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
        store = cls(embedder, index_type=payload.get("index_type", "flat"))
        store.index = faiss.read_index(str(directory / "index.faiss"))
        store.documents = [Document(**d) for d in payload["documents"]]
        return store
