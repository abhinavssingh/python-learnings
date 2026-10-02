from abc import ABC, abstractmethod

from .document import Document


class BaseChunker(ABC):
    """Splits text into smaller pieces that fit an embedding / LLM context."""

    name: str = "chunker"

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        pass

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in documents:
            for i, piece in enumerate(self.split_text(doc.page_content)):
                if not piece.strip():
                    continue
                metadata = {**doc.metadata, "chunk_index": i, "chunker": self.name}
                chunks.append(Document(page_content=piece, metadata=metadata))
        for global_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = global_index
        return chunks
