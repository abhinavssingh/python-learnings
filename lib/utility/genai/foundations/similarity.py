import math
from collections import Counter

import numpy as np


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = _as_2d(matrix)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


def cosine_similarity_matrix(A, B=None) -> np.ndarray:
    A = l2_normalize(A)
    B = A if B is None else l2_normalize(B)
    return A @ B.T


def dot_product(a, b) -> float:
    return float(np.dot(a, b))


def euclidean_distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def manhattan_distance(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).sum())


def jaccard_similarity(a, b) -> float:
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


class BM25:
    """
    Okapi BM25 - the classic lexical ranking function used by search engines.

        score(D, Q) = sum_{q in Q} IDF(q) * f(q, D) * (k1 + 1)
                      / (f(q, D) + k1 * (1 - b + b * |D| / avgdl))

    k1 controls term-frequency saturation, b controls length normalisation.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter] = []
        self.doc_len: np.ndarray = np.array([])
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def fit(self, tokenized_corpus: list[list[str]]) -> "BM25":
        self.doc_freqs = [Counter(doc) for doc in tokenized_corpus]
        self.doc_len = np.array([len(doc) for doc in tokenized_corpus], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        n_docs = len(tokenized_corpus)
        df: Counter = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        self.idf = {term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
        return self

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.doc_freqs))
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([freqs.get(term, 0) for freqs in self.doc_freqs], dtype=float)
            denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1))
            scores += idf * tf * (self.k1 + 1) / np.where(denom == 0, 1, denom)
        return scores

    def top_n(self, query_tokens: list[str], n: int = 5) -> list[tuple[int, float]]:
        scores = self.get_scores(query_tokens)
        order = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in order]
