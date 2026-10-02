from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Turns text into dense vectors (one row per text)."""

    name: str = "embedder"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        pass

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        return int(self.embed_query("dimension probe").shape[0])
