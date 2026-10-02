import ast
import re

from ..abstractions.base_chunker import BaseChunker
from ..abstractions.document import Document
from .recursive import RecursiveCharacterChunker


class MarkdownHeaderChunker(BaseChunker):
    """Split Markdown by headers; each chunk keeps its header path in metadata."""

    name = "markdown_header"
    HEADER = re.compile(r"^(#{1,6})\s+(.*)$")

    def __init__(self, max_level: int = 3, max_chars: int | None = 1500):
        self.max_level = max_level
        self.max_chars = max_chars

    def sections(self, text: str) -> list[tuple[dict[str, str], str]]:
        headers: dict[str, str] = {}
        sections: list[tuple[dict[str, str], str]] = []
        buffer: list[str] = []

        def flush():
            content = "\n".join(buffer).strip()
            if content:
                sections.append((dict(headers), content))
            buffer.clear()

        for line in text.splitlines():
            match = self.HEADER.match(line)
            if match and len(match.group(1)) <= self.max_level:
                flush()
                level = len(match.group(1))
                headers = {k: v for k, v in headers.items() if int(k[1:]) < level}
                headers[f"h{level}"] = match.group(2).strip()
            else:
                buffer.append(line)
        flush()
        return sections

    def split_text(self, text: str) -> list[str]:
        return [content for _, content in self.sections(text)]

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        sub_splitter = RecursiveCharacterChunker(self.max_chars, min(100, self.max_chars // 5)) if self.max_chars else None
        for doc in documents:
            for headers, content in self.sections(doc.page_content):
                pieces = sub_splitter.split_text(content) if sub_splitter and len(content) > self.max_chars else [content]
                for piece in pieces:
                    metadata = {**doc.metadata, **headers, "chunker": self.name,
                                "header_path": " > ".join(headers.values())}
                    chunks.append(Document(piece, metadata))
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        return chunks


class PythonCodeChunker(BaseChunker):
    """Code-aware splitting: one chunk per top-level function / class (via the `ast` module)."""

    name = "python_code"

    def split_text(self, text: str) -> list[str]:
        return [chunk for _, chunk in self.blocks(text)]

    def blocks(self, text: str) -> list[tuple[str, str]]:
        tree = ast.parse(text)
        lines = text.splitlines()
        blocks: list[tuple[str, str]] = []
        preamble_end = len(lines)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
                preamble_end = min(preamble_end, start)
                blocks.append((node.name, "\n".join(lines[start: node.end_lineno])))
        preamble = "\n".join(lines[:preamble_end]).strip()
        return ([("<module>", preamble)] if preamble else []) + blocks
