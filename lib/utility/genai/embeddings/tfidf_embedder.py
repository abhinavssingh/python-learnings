import numpy as np

from ..abstractions.base_embedder import BaseEmbedder
from ..foundations.vectorization import TfidfVectorizerScratch


class TfidfEmbedder(BaseEmbedder):
    """
    Sparse TF-IDF vectors exposed through the embedder interface.

    Call `fit(corpus)` first; otherwise the first `embed_documents`
    call fits the vocabulary on the documents it receives.
    """

    name = "tfidf"

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizerScratch(ngram_range=ngram_range, sublinear_tf=True)
        self.fitted = False

    def fit(self, corpus: list[str]) -> "TfidfEmbedder":
        self.vectorizer.fit(corpus)
        self.fitted = True
        return self

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not self.fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).astype(np.float32)

    @property
    def dimension(self) -> int:
        return len(self.vectorizer.vocabulary_)
