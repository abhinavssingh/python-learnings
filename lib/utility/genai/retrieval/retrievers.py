import re

import numpy as np

from ..abstractions.base_retriever import BaseRetriever
from ..abstractions.base_vectorstore import BaseVectorStore, MetadataFilter
from ..abstractions.document import Document, SearchResult
from ..foundations.similarity import BM25
from ..foundations.text_cleaning import TextCleaner


class DenseRetriever(BaseRetriever):
    """Semantic search: embed the query and find nearest vectors."""

    name = "dense"

    def __init__(self, vectorstore: BaseVectorStore, filter: MetadataFilter = None, score_threshold: float | None = None):
        self.vectorstore = vectorstore
        self.filter = filter
        self.score_threshold = score_threshold

    def retrieve(self, query: str, k: int = 4) -> list[SearchResult]:
        results = self.vectorstore.similarity_search(query, k=k, filter=self.filter)
        if self.score_threshold is not None:
            results = [r for r in results if r.score >= self.score_threshold]
        return results


class BM25Retriever(BaseRetriever):
    """Lexical (keyword) search with BM25 - great for exact names, codes and rare terms."""

    name = "bm25"

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75, stem: bool = True):
        self.documents = documents
        self.cleaner = TextCleaner()
        self.stem = stem
        self.bm25 = BM25(k1=k1, b=b).fit([self._tokens(d.page_content) for d in documents])

    def _tokens(self, text: str) -> list[str]:
        return self.cleaner.clean(text, remove_stopwords=True, stem=self.stem)

    def retrieve(self, query: str, k: int = 4) -> list[SearchResult]:
        return [
            SearchResult(self.documents[i], score, None, self.name)
            for i, score in self.bm25.top_n(self._tokens(query), k)
            if score > 0
        ]


def reciprocal_rank_fusion(result_lists: list[list[SearchResult]], k: int = 60,
                           weights: list[float] | None = None) -> list[SearchResult]:
    """
    RRF: score(d) = sum_i  w_i / (k + rank_i(d))

    Works on ranks only, so scores from different retrievers
    (cosine vs BM25) never need to be normalised.
    """
    weights = weights or [1.0] * len(result_lists)
    fused: dict[str, float] = {}
    best: dict[str, SearchResult] = {}
    for weight, results in zip(weights, result_lists):
        for rank, result in enumerate(results, start=1):
            key = result.document.id
            fused[key] = fused.get(key, 0.0) + weight / (k + rank)
            if key not in best or (best[key].vector is None and result.vector is not None):
                best[key] = result
    ordered = sorted(fused, key=fused.get, reverse=True)
    return [SearchResult(best[key].document, fused[key], best[key].vector, "rrf") for key in ordered]


class HybridRetriever(BaseRetriever):
    """Combine several retrievers (typically dense + BM25) with reciprocal rank fusion."""

    name = "hybrid"

    def __init__(self, retrievers: list[BaseRetriever], weights: list[float] | None = None,
                 fetch_k: int = 10, rrf_k: int = 60):
        self.retrievers = retrievers
        self.weights = weights
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 4) -> list[SearchResult]:
        lists = [r.retrieve(query, self.fetch_k) for r in self.retrievers]
        return reciprocal_rank_fusion(lists, k=self.rrf_k, weights=self.weights)[:k]


def mmr_select(query_vector: np.ndarray, candidate_vectors: np.ndarray, k: int = 4, lambda_mult: float = 0.5) -> list[int]:
    """
    Maximal Marginal Relevance:
        argmax_d [ lambda * sim(q, d) - (1 - lambda) * max_{s in selected} sim(d, s) ]

    lambda=1 -> pure relevance, lambda=0 -> pure diversity.
    """
    if len(candidate_vectors) == 0:
        return []
    norm = candidate_vectors / np.linalg.norm(candidate_vectors, axis=1, keepdims=True).clip(min=1e-12)
    q = query_vector / max(np.linalg.norm(query_vector), 1e-12)
    relevance = norm @ q
    selected = [int(np.argmax(relevance))]
    while len(selected) < min(k, len(candidate_vectors)):
        redundancy = (norm @ norm[selected].T).max(axis=1)
        mmr = lambda_mult * relevance - (1 - lambda_mult) * redundancy
        mmr[selected] = -np.inf
        selected.append(int(np.argmax(mmr)))
    return selected


class MMRRetriever(BaseRetriever):
    """Fetch `fetch_k` nearest chunks, then pick `k` that are relevant *and* diverse."""

    name = "mmr"

    def __init__(self, vectorstore: BaseVectorStore, fetch_k: int = 12, lambda_mult: float = 0.5):
        self.vectorstore = vectorstore
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult

    def retrieve(self, query: str, k: int = 4) -> list[SearchResult]:
        query_vector = self.vectorstore.embedder.embed_query(query)
        candidates = self.vectorstore.similarity_search_by_vector(query_vector, k=self.fetch_k)
        if not candidates:
            return []
        vectors = np.vstack([c.vector for c in candidates])
        chosen = mmr_select(query_vector, vectors, k=k, lambda_mult=self.lambda_mult)
        return [SearchResult(candidates[i].document, candidates[i].score, candidates[i].vector, self.name) for i in chosen]


def keyword_tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))
