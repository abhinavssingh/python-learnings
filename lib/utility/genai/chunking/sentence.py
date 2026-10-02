import re

from ..abstractions.base_chunker import BaseChunker
from ..foundations.tokenization import SentenceTokenizer


class SentenceChunker(BaseChunker):
    """Group whole sentences until `max_chars`; repeat the last `overlap_sentences` in the next chunk."""

    name = "sentence"

    def __init__(self, max_chars: int = 500, overlap_sentences: int = 1):
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences
        self.sentence_tokenizer = SentenceTokenizer()

    def split_text(self, text: str) -> list[str]:
        sentences = self.sentence_tokenizer.tokenize(text)
        chunks: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            if current and len(" ".join(current + [sentence])) > self.max_chars:
                chunks.append(" ".join(current))
                current = current[-self.overlap_sentences:] if self.overlap_sentences else []
                if current and len(" ".join(current + [sentence])) > self.max_chars:
                    current = []
            current.append(sentence)
        if current:
            chunks.append(" ".join(current))
        return chunks


class ParagraphChunker(BaseChunker):
    """Split on line breaks / blank lines, merging short paragraphs up to `max_chars`."""

    name = "paragraph"

    def __init__(self, max_chars: int = 800, min_chars: int = 80):
        self.max_chars = max_chars
        self.min_chars = min_chars

    def split_text(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            candidate = f"{current}\n{para}".strip()
            if current and len(candidate) > self.max_chars and len(current) >= self.min_chars:
                chunks.append(current)
                current = para
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks
