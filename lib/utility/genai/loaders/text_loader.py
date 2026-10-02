from pathlib import Path

from ..abstractions.document import Document


class TextLoader:
    """Load .txt / .md / .py files as a single Document."""

    def __init__(self, path: str | Path, encoding: str = "utf-8"):
        self.path = Path(path)
        self.encoding = encoding

    def load(self) -> list[Document]:
        text = self.path.read_text(encoding=self.encoding, errors="ignore")
        return [Document(text, {"source": self.path.name, "type": self.path.suffix.lstrip(".")})]
