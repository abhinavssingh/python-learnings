import hashlib
import json
from pathlib import Path
from typing import Callable

from ..abstractions.base_embedder import BaseEmbedder
from ..abstractions.document import Document
from .in_memory_store import InMemoryVectorStore


class IndexManager:
    """
    Build a vector index once and reload it on the next run.

    The index is rebuilt automatically when the chunks or the embedding
    model change (detected with a fingerprint hash).
    """

    def __init__(self, base_dir: str | Path, store_cls=InMemoryVectorStore):
        self.base_dir = Path(base_dir)
        self.store_cls = store_cls

    @staticmethod
    def fingerprint(documents: list[Document], embedder: BaseEmbedder) -> str:
        h = hashlib.sha256(embedder.name.encode("utf-8"))
        for doc in documents:
            h.update(doc.page_content.encode("utf-8"))
            h.update(json.dumps(doc.metadata, sort_keys=True, default=str).encode("utf-8"))
        return h.hexdigest()[:16]

    def build_or_load(self, name: str, documents: list[Document] | Callable[[], list[Document]],
                      embedder: BaseEmbedder, force_rebuild: bool = False):
        docs = documents() if callable(documents) else documents
        fingerprint = self.fingerprint(docs, embedder)
        directory = self.base_dir / name
        meta_file = directory / "index_meta.json"

        if not force_rebuild and meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fingerprint:
                store = self.store_cls.load(directory, embedder)
                store.loaded_from_disk = True
                return store

        store = self.store_cls(embedder)
        store.add_documents(docs)
        store.save(directory)
        meta_file.write_text(json.dumps({"fingerprint": fingerprint, "embedder": embedder.name,
                                         "documents": len(docs), "store": self.store_cls.__name__}, indent=2), encoding="utf-8")
        store.loaded_from_disk = False
        return store
