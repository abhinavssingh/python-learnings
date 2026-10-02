import hashlib
import json
from pathlib import Path

import numpy as np

from ..abstractions.base_embedder import BaseEmbedder


class CachedEmbedder(BaseEmbedder):
    """
    Disk cache around any embedder - the same text is never embedded twice.

    Cache layout:  <cache_dir>/<embedder-name>.npz  (keys + vectors)
    Queries are cached separately because some models use query prefixes.
    """

    def __init__(self, embedder: BaseEmbedder, cache_dir: str | Path):
        self.embedder = embedder
        self.name = f"cached:{embedder.name}"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() else "_" for c in embedder.name)
        self.cache_file = self.cache_dir / f"{safe_name}.npz"
        self._cache: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        self._load()

    @staticmethod
    def _key(kind: str, text: str) -> str:
        return hashlib.sha256(f"{kind}::{text}".encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if self.cache_file.exists():
            data = np.load(self.cache_file, allow_pickle=False)
            keys = json.loads(str(data["keys"]))
            self._cache = dict(zip(keys, data["vectors"]))

    def _save(self) -> None:
        if not self._cache:
            return
        keys = list(self._cache)
        np.savez(self.cache_file, keys=np.array(json.dumps(keys)), vectors=np.vstack([self._cache[k] for k in keys]))

    def _lookup(self, kind: str, texts: list[str], compute) -> np.ndarray:
        keys = [self._key(kind, t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            vectors = compute([texts[i] for i in missing])
            for i, vec in zip(missing, np.atleast_2d(vectors)):
                self._cache[keys[i]] = np.asarray(vec, dtype=np.float32)
            self._save()
        return np.vstack([self._cache[k] for k in keys]) if keys else np.zeros((0, 0), dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._lookup("doc", texts, self.embedder.embed_documents)

    def embed_query(self, text: str) -> np.ndarray:
        return self._lookup("query", [text], lambda ts: np.atleast_2d(self.embedder.embed_query(ts[0])))[0]

    def stats(self) -> dict[str, int]:
        return {"cached_vectors": len(self._cache), "hits": self.hits, "misses": self.misses}
