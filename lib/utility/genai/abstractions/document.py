import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Document:
    """A piece of text plus metadata (source, page, chunk index ...)."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        if "id" in self.metadata:
            return str(self.metadata["id"])
        digest = hashlib.sha1(self.page_content.encode("utf-8")).hexdigest()
        return digest[:16]

    def preview(self, length: int = 120) -> str:
        text = " ".join(self.page_content.split())
        return text if len(text) <= length else text[:length] + "..."


@dataclass
class SearchResult:
    """A retrieved document with its score (higher = more relevant)."""

    document: Document
    score: float
    vector: np.ndarray | None = None
    retriever: str = ""

    @property
    def text(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata
