from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever as LCBaseRetriever

from ...abstractions.base_embedder import BaseEmbedder
from ...abstractions.base_retriever import BaseRetriever
from ...abstractions.document import Document


def to_lc_documents(documents: list[Document]) -> list[LCDocument]:
    return [LCDocument(page_content=d.page_content, metadata=dict(d.metadata)) for d in documents]


def from_lc_documents(documents: list[LCDocument]) -> list[Document]:
    return [Document(d.page_content, dict(d.metadata)) for d in documents]


class EmbedderAdapter(Embeddings):
    """Use any of our embedders (Ollama, cached, hashing ...) inside LangChain."""

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_query(text).tolist()


class RetrieverAdapter(LCBaseRetriever):
    """Expose one of our retrievers (hybrid, MMR, BM25 ...) as a LangChain Runnable retriever."""

    retriever: Any
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[LCDocument]:
        results = self.retriever.retrieve(query, self.k)
        return [LCDocument(page_content=r.text, metadata={**r.metadata, "score": r.score}) for r in results]


def as_langchain_retriever(retriever: BaseRetriever, k: int = 4) -> RetrieverAdapter:
    return RetrieverAdapter(retriever=retriever, k=k)
