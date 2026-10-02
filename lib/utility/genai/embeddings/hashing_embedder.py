import hashlib
import re

import numpy as np

from ..abstractions.base_embedder import BaseEmbedder


class HashingEmbedder(BaseEmbedder):
    """
    Offline, deterministic embedding using the "hashing trick".

    Each word (and word bigram) is hashed into one of `dim` buckets.
    No training and no model download - useful as a fast baseline
    and for unit tests, but it has no notion of meaning/synonyms.
    """

    def __init__(self, dim: int = 512, use_bigrams: bool = True):
        self.dim = dim
        self.use_bigrams = use_bigrams
        self.name = f"hashing:{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign

    def _embed_one(self, text: str) -> np.ndarray:
        tokens = re.findall(r"\w+", text.lower())
        features = tokens + ([f"{a}_{b}" for a, b in zip(tokens, tokens[1:])] if self.use_bigrams else [])
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature in features:
            index, sign = self._bucket(feature)
            vector[index] += sign
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self.dim
