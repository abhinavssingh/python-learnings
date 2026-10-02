import math
import re
from collections import Counter
from typing import Callable

import numpy as np


def default_tokenizer(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BagOfWordsVectorizer:
    """Document -> vector of word counts (or 0/1 when binary=True)."""

    def __init__(self, tokenizer: Callable[[str], list[str]] = default_tokenizer, binary: bool = False, ngram_range: tuple[int, int] = (1, 1)):
        self.tokenizer = tokenizer
        self.binary = binary
        self.ngram_range = ngram_range
        self.vocabulary_: dict[str, int] = {}

    def _features(self, text: str) -> list[str]:
        tokens = self.tokenizer(text)
        features: list[str] = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            features.extend(" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1))
        return features

    def fit(self, documents: list[str]) -> "BagOfWordsVectorizer":
        vocab = sorted({f for doc in documents for f in self._features(doc)})
        self.vocabulary_ = {term: i for i, term in enumerate(vocab)}
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        matrix = np.zeros((len(documents), len(self.vocabulary_)), dtype=float)
        for row, doc in enumerate(documents):
            for term, count in Counter(self._features(doc)).items():
                col = self.vocabulary_.get(term)
                if col is not None:
                    matrix[row, col] = 1.0 if self.binary else count
        return matrix

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)

    def get_feature_names(self) -> list[str]:
        return sorted(self.vocabulary_, key=self.vocabulary_.get)


class TfidfVectorizerScratch(BagOfWordsVectorizer):
    """
    TF-IDF from scratch (same formula as scikit-learn defaults).

        tf(t, d)  = count of t in d          (or 1 + log(count) if sublinear)
        idf(t)    = ln((1 + N) / (1 + df(t))) + 1     (smooth_idf=True)
        tfidf     = tf * idf, then L2-normalised per document
    """

    def __init__(self, tokenizer: Callable[[str], list[str]] = default_tokenizer, smooth_idf: bool = True,
                 sublinear_tf: bool = False, norm: str | None = "l2", ngram_range: tuple[int, int] = (1, 1)):
        super().__init__(tokenizer=tokenizer, binary=False, ngram_range=ngram_range)
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.idf_: np.ndarray = np.array([])

    def fit(self, documents: list[str]) -> "TfidfVectorizerScratch":
        super().fit(documents)
        counts = super().transform(documents)
        n_docs = len(documents)
        df = (counts > 0).sum(axis=0)
        if self.smooth_idf:
            self.idf_ = np.log((1 + n_docs) / (1 + df)) + 1
        else:
            self.idf_ = np.log(n_docs / df) + 1
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        tf = super().transform(documents)
        if self.sublinear_tf:
            tf = np.where(tf > 0, 1 + np.log(np.maximum(tf, 1e-12)), 0)
        tfidf = tf * self.idf_
        if self.norm == "l2":
            norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
            tfidf = tfidf / np.where(norms == 0, 1, norms)
        return tfidf

    def top_terms(self, vector: np.ndarray, top: int = 10) -> list[tuple[str, float]]:
        names = self.get_feature_names()
        order = np.argsort(vector)[::-1][:top]
        return [(names[i], float(vector[i])) for i in order if vector[i] > 0]

    def idf_table(self) -> list[tuple[str, float]]:
        return sorted(zip(self.get_feature_names(), self.idf_.tolist()), key=lambda x: -x[1])


def log_idf(n_docs: int, doc_freq: int) -> float:
    return math.log((1 + n_docs) / (1 + doc_freq)) + 1
