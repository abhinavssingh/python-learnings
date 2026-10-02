from ..abstractions.base_chunker import BaseChunker

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


class RecursiveCharacterChunker(BaseChunker):
    """
    The most widely used splitter (same idea as LangChain's RecursiveCharacterTextSplitter).

    Try the "largest" separator first (paragraphs); any piece still larger
    than `chunk_size` is split again with the next separator (lines ->
    sentences -> words -> characters). Small pieces are then merged back
    together up to `chunk_size`, keeping `chunk_overlap` characters of context.
    """

    name = "recursive"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separators: list[str] | None = None):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def split_text(self, text: str) -> list[str]:
        return [c.strip() for c in self._split(text, self.separators) if c.strip()]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        separator, remaining = separators[-1], []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator, remaining = sep, separators[i + 1:]
                break

        if separator:
            parts = text.split(separator)
            pieces = [p + separator for p in parts[:-1]] + [parts[-1]]
        else:
            pieces = list(text)

        chunks: list[str] = []
        small: list[str] = []
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                small.append(piece)
                continue
            if small:
                chunks.extend(self._merge(small))
                small = []
            chunks.extend(self._split(piece, remaining) if remaining else [piece])
        if small:
            chunks.extend(self._merge(small))
        return chunks

    def _merge(self, pieces: list[str]) -> list[str]:
        merged: list[str] = []
        window: list[str] = []
        total = 0
        for piece in pieces:
            if window and total + len(piece) > self.chunk_size:
                merged.append("".join(window))
                while window and (total > self.chunk_overlap or total + len(piece) > self.chunk_size):
                    total -= len(window.pop(0))
            window.append(piece)
            total += len(piece)
        if window:
            merged.append("".join(window))
        return merged
