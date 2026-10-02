import re
from pathlib import Path

from pypdf import PdfReader

from ..abstractions.document import Document


def clean_pdf_text(text: str) -> str:
    """Fix common PDF extraction artefacts (hyphenated breaks, hard line wraps)."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"(?<![.!?:\n])\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


class PDFLoader:
    """
    Load a PDF with pypdf.

    mode="page"   -> one Document per page (metadata: source, page)
    mode="single" -> the whole PDF as one Document

    remove_patterns: regexes (e.g. running page headers) removed from every page.
    """

    def __init__(self, path: str | Path, mode: str = "page", clean: bool = True, min_chars: int = 20,
                 remove_patterns: list[str] | None = None):
        self.path = Path(path)
        self.mode = mode
        self.clean = clean
        self.min_chars = min_chars
        self.remove_patterns = [re.compile(p, re.MULTILINE) for p in (remove_patterns or [])]

    def load(self) -> list[Document]:
        reader = PdfReader(str(self.path))
        pages: list[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            for pattern in self.remove_patterns:
                raw = pattern.sub("", raw)
            text = clean_pdf_text(raw) if self.clean else raw
            if len(text.strip()) < self.min_chars:
                continue
            pages.append(Document(text, {"source": self.path.name, "page": page_number, "type": "pdf"}))

        if self.mode == "single":
            combined = "\n\n".join(p.page_content for p in pages)
            return [Document(combined, {"source": self.path.name, "pages": len(pages), "type": "pdf"})]
        return pages
