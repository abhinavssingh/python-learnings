"""
Vector stores — storing embeddings and searching them fast.

Learn:
  * brute-force NumPy store (cosine / dot / euclidean) vs FAISS (flat exact, HNSW approximate)
  * metadata filters (Mongo-style: $eq, $in, $gte, ...)
  * persisting an index to disk and reloading it
  * IndexManager: rebuild only when documents or embedder change (fingerprint)
"""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

import numpy as np  # noqa: E402

from lib.utility.genai.config import GenAIConfig  # noqa: E402
from lib.utility.genai.datasets import hr_policy_chunks  # noqa: E402
from lib.utility.genai.factory import get_embedder  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.vectorstores import FaissVectorStore, IndexManager, InMemoryVectorStore  # noqa: E402

QUERY = "How does the company support learning and development?"


def results_table(results):
    return [{"rank": i + 1, "score": round(r.score, 4), "page": r.metadata.get("page"), "text": r.document.preview(90)}
            for i, r in enumerate(results)]


def main():
    config = GenAIConfig.from_env()
    embedder = get_embedder(config)
    chunks = hr_policy_chunks(config.chunk_size, config.chunk_overlap)
    vectors = embedder.embed_documents([c.page_content for c in chunks])
    query_vec = embedder.embed_query(QUERY)

    stores = {
        "in-memory cosine": InMemoryVectorStore(embedder, metric="cosine"),
        "in-memory euclidean": InMemoryVectorStore(embedder, metric="euclidean"),
        "faiss flat (exact)": FaissVectorStore(embedder, index_type="flat"),
        "faiss hnsw (approx.)": FaissVectorStore(embedder, index_type="hnsw"),
    }
    timing_rows, top_ids = [], {}
    for name, store in stores.items():
        store.add_documents(chunks, embeddings=vectors)
        t0 = time.perf_counter()
        for _ in range(200):
            results = store.similarity_search_by_vector(query_vec, k=4)
        timing_rows.append({"store": name, "search time (ms, avg of 200)": round((time.perf_counter() - t0) / 200 * 1000, 3),
                            "top-4 pages": [r.metadata.get("page") for r in results]})
        top_ids[name] = [r.document.id for r in results]
    reference = top_ids["in-memory cosine"]
    for row, name in zip(timing_rows, stores):
        row["overlap with exact cosine"] = f"{len(set(top_ids[name]) & set(reference))}/4"

    store = stores["in-memory cosine"]
    filters = {
        "no filter": None,
        "{'page': 6}": {"page": 6},
        "{'page': {'$gte': 7}}": {"page": {"$gte": 7}},
        "{'page': {'$in': [4, 5]}}": {"page": {"$in": [4, 5]}},
    }
    filter_rows = []
    for label, flt in filters.items():
        res = store.similarity_search(QUERY, k=3, filter=flt)
        filter_rows.append({"filter": label, "pages": [r.metadata.get("page") for r in res],
                            "best score": round(res[0].score, 3) if res else None})

    with tempfile.TemporaryDirectory() as tmp:
        store.save(tmp)
        reloaded = InMemoryVectorStore.load(tmp, embedder)
        same = [r.document.id for r in reloaded.similarity_search_by_vector(query_vec, 4)] == reference
        files = sorted(p.name for p in Path(tmp).iterdir())

    manager = IndexManager(config.vectorstore_dir)
    t0 = time.perf_counter()
    first = manager.build_or_load("hr_policy_demo", chunks, embedder)
    first_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    second = manager.build_or_load("hr_policy_demo", chunks, embedder)
    second_s = time.perf_counter() - t0

    deleted = InMemoryVectorStore(embedder)
    deleted.add_documents(chunks, embeddings=vectors)
    removed = deleted.delete({"page": 7})

    report = GenAIReport("Vector stores: NumPy vs FAISS, filters, persistence")
    report.grid([
        report.kv("Index", {"chunks": len(chunks), "dimension": vectors.shape[1], "query": QUERY,
                            "vector matrix MB": round(vectors.nbytes / 1e6, 3)}),
        report.table("Top-4 (in-memory cosine)", results_table(store.similarity_search_by_vector(query_vec, 4))),
    ])
    report.full_table("Store comparison", timing_rows)
    report.full_table("Metadata filtering", filter_rows)
    report.grid([
        report.kv("Save / load round-trip", {"files written": files, "identical results after reload": same}),
        report.kv("IndexManager (fingerprinted cache)", {
            "1st call loaded_from_disk": first.loaded_from_disk, "1st call (s)": round(first_s, 3),
            "2nd call loaded_from_disk": second.loaded_from_disk, "2nd call (s)": round(second_s, 3),
            "location": str(config.vectorstore_dir)}),
        report.kv("Delete by filter", {"deleted chunks from page 7": removed, "remaining": len(deleted)}),
    ], columns=3)
    report.full_text("Key takeaways", "\n".join([
        "* A vector store = embeddings matrix + documents + metadata + a nearest-neighbour search.",
        "* Brute force is exact and fine for thousands of chunks; FAISS/HNSW scales to millions (approximate).",
        "* Filters let you scope search (department, date, access rights) BEFORE ranking.",
        "* Persist indexes and rebuild only when the data or embedding model changes.",
        f"* Vectors are L2-normalised ({np.allclose(np.linalg.norm(vectors, axis=1), 1)}) -> cosine, dot and euclidean give the same ranking.",
    ]))
    report.save(__file__, "vectorstore_basics_report.html")


if __name__ == "__main__":
    main()
