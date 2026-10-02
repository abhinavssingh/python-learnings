import re

from ..abstractions.base_chunker import BaseChunker


class FixedSizeChunker(BaseChunker):
    """
    Sliding window of `chunk_size` units with `chunk_overlap` units of overlap.

    unit="char" -> characters, unit="word" -> whitespace tokens.
    Simple and predictable, but it happily cuts sentences in half.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, unit: str = "char"):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unit = unit
        self.name = f"fixed_{unit}"

    def split_text(self, text: str) -> list[str]:
        step = self.chunk_size - self.chunk_overlap
        if self.unit == "word":
            words = text.split()
            return [" ".join(words[i: i + self.chunk_size]) for i in range(0, max(len(words) - self.chunk_overlap, 1), step)]
        text = re.sub(r"\s+", " ", text).strip()
        return [text[i: i + self.chunk_size] for i in range(0, max(len(text) - self.chunk_overlap, 1), step)]
