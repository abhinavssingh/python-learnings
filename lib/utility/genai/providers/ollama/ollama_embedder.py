import numpy as np
import ollama

from ...abstractions.base_embedder import BaseEmbedder
from ...config.genai_config import GenAIConfig


class OllamaEmbedder(BaseEmbedder):
    """
    Dense embeddings from a local Ollama embedding model (default: nomic-embed-text).

    nomic-embed-text is trained with task prefixes, so documents and
    queries are prefixed differently for better retrieval quality.
    """

    TASK_PREFIXES = {
        "nomic-embed-text": ("search_document: ", "search_query: "),
    }

    def __init__(self, config: GenAIConfig | None = None, model: str | None = None,
                 batch_size: int = 32, normalize: bool = True, use_prefixes: bool = True):
        self.config = config or GenAIConfig.from_env()
        self.model = model or self.config.embedding_model
        self.name = f"ollama:{self.model}"
        self.batch_size = batch_size
        self.normalize = normalize
        self.client = ollama.Client(host=self.config.ollama_host, timeout=self.config.request_timeout)
        prefixes = self.TASK_PREFIXES.get(self.model.split(":")[0], ("", "")) if use_prefixes else ("", "")
        self.doc_prefix, self.query_prefix = prefixes

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            vectors.extend(self.client.embed(model=self.model, input=batch).embeddings)
        matrix = np.asarray(vectors, dtype=np.float32)
        if self.normalize and len(matrix):
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
        return matrix

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]
