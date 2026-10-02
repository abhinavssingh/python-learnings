import numpy as np
import pandas as pd

from ..abstractions.document import Document


class ChunkInspector:
    """Quality statistics for a list of chunks (size distribution, overlap, sentence breaks)."""

    @staticmethod
    def _texts(chunks: list[str] | list[Document]) -> list[str]:
        return [c.page_content if isinstance(c, Document) else c for c in chunks]

    @staticmethod
    def _overlap(a: str, b: str, max_check: int = 300) -> int:
        for size in range(min(len(a), len(b), max_check), 0, -1):
            if a.endswith(b[:size]):
                return size
        return 0

    @classmethod
    def stats(cls, chunks: list[str] | list[Document]) -> dict:
        texts = cls._texts(chunks)
        if not texts:
            return {"chunks": 0}
        chars = np.array([len(t) for t in texts])
        words = np.array([len(t.split()) for t in texts])
        overlaps = [cls._overlap(a, b) for a, b in zip(texts, texts[1:])]
        clean_end = sum(t.rstrip().endswith((".", "!", "?", ":")) for t in texts)
        return {
            "chunks": len(texts),
            "avg_chars": round(float(chars.mean()), 1),
            "min_chars": int(chars.min()),
            "max_chars": int(chars.max()),
            "std_chars": round(float(chars.std()), 1),
            "avg_words": round(float(words.mean()), 1),
            "approx_tokens": int(words.sum() * 1.3),
            "avg_overlap_chars": round(float(np.mean(overlaps)), 1) if overlaps else 0.0,
            "ends_on_sentence_%": round(100 * clean_end / len(texts), 1),
        }

    @classmethod
    def to_frame(cls, chunks: list[str] | list[Document], preview: int = 90) -> pd.DataFrame:
        rows = []
        for i, chunk in enumerate(chunks):
            text = chunk.page_content if isinstance(chunk, Document) else chunk
            meta = chunk.metadata if isinstance(chunk, Document) else {}
            rows.append({
                "chunk": i,
                "page": meta.get("page", ""),
                "chars": len(text),
                "words": len(text.split()),
                "preview": " ".join(text.split())[:preview],
            })
        return pd.DataFrame(rows)
