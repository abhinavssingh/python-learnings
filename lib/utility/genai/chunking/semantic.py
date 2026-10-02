import numpy as np

from ..abstractions.base_chunker import BaseChunker
from ..abstractions.base_embedder import BaseEmbedder
from ..foundations.tokenization import SentenceTokenizer


class SemanticChunker(BaseChunker):
    """
    Split where the *meaning* changes.

    1. Split into sentences and build a small window around each (buffer).
    2. Embed every window.
    3. Compute cosine distance between consecutive windows.
    4. Start a new chunk where distance > the `breakpoint_percentile` threshold
       (or when the chunk would exceed `max_chars`).
    """

    name = "semantic"

    def __init__(self, embedder: BaseEmbedder, breakpoint_percentile: float = 85,
                 buffer_size: int = 1, max_chars: int = 1200, min_chars: int = 150):
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.sentence_tokenizer = SentenceTokenizer()
        self.last_distances: np.ndarray = np.array([])
        self.last_threshold: float = 0.0
        self.last_sentences: list[str] = []

    def _distances(self, sentences: list[str]) -> np.ndarray:
        windows = [
            " ".join(sentences[max(0, i - self.buffer_size): i + self.buffer_size + 1])
            for i in range(len(sentences))
        ]
        vectors = self.embedder.embed_documents(windows)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        return 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)

    def split_text(self, text: str) -> list[str]:
        sentences = self.sentence_tokenizer.tokenize(text)
        self.last_sentences = sentences
        if len(sentences) < 3:
            self.last_distances = np.array([])
            return [" ".join(sentences)] if sentences else []

        distances = self._distances(sentences)
        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        self.last_distances, self.last_threshold = distances, threshold

        chunks: list[str] = []
        current = [sentences[0]]
        for i, sentence in enumerate(sentences[1:]):
            current_text = " ".join(current)
            is_breakpoint = distances[i] > threshold and len(current_text) >= self.min_chars
            too_long = len(current_text) + len(sentence) > self.max_chars
            if is_breakpoint or too_long:
                chunks.append(current_text)
                current = []
            current.append(sentence)
        if current:
            chunks.append(" ".join(current))
        return chunks
