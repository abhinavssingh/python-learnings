from pathlib import Path

from ..abstractions.document import Document
from .pdf_loader import PDFLoader
from .tabular_loader import TabularLoader
from .text_loader import TextLoader


class DirectoryLoader:
    """Load every supported file in a folder, dispatching by file extension."""

    LOADERS = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".py": TextLoader,
        ".csv": TabularLoader,
        ".xlsx": TabularLoader,
    }

    def __init__(self, directory: str | Path, pattern: str = "*", recursive: bool = False,
                 extensions: set[str] | None = None):
        self.directory = Path(directory)
        self.pattern = pattern
        self.recursive = recursive
        self.extensions = extensions or set(self.LOADERS)

    def files(self) -> list[Path]:
        iterator = self.directory.rglob(self.pattern) if self.recursive else self.directory.glob(self.pattern)
        return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in self.extensions)

    def load(self) -> list[Document]:
        documents: list[Document] = []
        for path in self.files():
            documents.extend(self.LOADERS[path.suffix.lower()](path).load())
        return documents
