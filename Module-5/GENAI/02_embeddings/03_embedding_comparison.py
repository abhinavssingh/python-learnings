"""
Comparing embedding approaches: hashing vs TF-IDF vs neural (Ollama).

Learn:
  * how each embedder handles paraphrases (same meaning, different words)
  * topic separation: similarity within vs across topics
  * retrieval quality on the HR policy golden QA set (hit rate, MRR, nDCG)
  * cost / speed trade-offs
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

import numpy as np  # noqa: E402

from lib.utility.genai.datasets import SAMPLE_SENTENCES, hr_policy_chunks, load_hr_qa  # noqa: E402
from lib.utility.genai.embeddings import HashingEmbedder, TfidfEmbedder  # noqa: E402
from lib.utility.genai.evaluation import RetrievalEvaluator  # noqa: E402
from lib.utility.genai.factory import get_embedder  # noqa: E402
from lib.utility.genai.foundations import cosine_similarity  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.retrieval import DenseRetriever  # noqa: E402
from lib.utility.genai.vectorstores import InMemoryVectorStore  # noqa: E402
from lib.utility.genai.visualization import metric_bar  # noqa: E402

PARAPHRASES = [
    ("The employee received a salary increase.", "Staff compensation was raised."),
    ("Managers coach their teams.", "Supervisors mentor the people who report to them."),
    ("Harassment is not tolerated.", "Bullying at work is forbidden."),
]


def separation(embedder, texts, groups) -> float:
    vectors = embedder.embed_documents(texts)
    sims = vectors @ vectors.T
    same = np.array([[a == b for b in groups] for a in groups]) & ~np.eye(len(groups), dtype=bool)
    diff = ~np.array([[a == b for b in groups] for a in groups])
    return float(sims[same].mean() - sims[diff].mean())


def main():
    chunks = hr_policy_chunks()
    texts = [s for s, _ in SAMPLE_SENTENCES]
    groups = [g for _, g in SAMPLE_SENTENCES]

    embedders = {
        "hashing (512-d)": HashingEmbedder(512),
        "tfidf": TfidfEmbedder().fit([c.page_content for c in chunks] + texts),
        "ollama nomic-embed-text": get_embedder(),
    }

    info_rows, para_rows, retrievers = [], [], {}
    for name, emb in embedders.items():
        t0 = time.perf_counter()
        store = InMemoryVectorStore(emb)
        store.add_documents(chunks)
        index_s = time.perf_counter() - t0
        retrievers[name] = DenseRetriever(store)
        info_rows.append({"embedder": name, "dimension": emb.dimension, "index time (s)": round(index_s, 2),
                          "topic separation": round(separation(emb, texts, groups), 3)})
        for a, b in PARAPHRASES:
            va, vb = emb.embed_documents([a, b])
            para_rows.append({"embedder": name, "pair": f"{a} <-> {b}", "cosine": round(cosine_similarity(va, vb), 3)})

    summary, detail = RetrievalEvaluator(load_hr_qa(), k=4).compare(retrievers)
    summary = summary.reset_index()

    report = GenAIReport("Embedding Comparison: hashing vs TF-IDF vs neural")
    report.full_table("Embedders", info_rows)
    report.full_table("Paraphrase similarity (higher = understands meaning)", para_rows, rows=9)
    report.full_table("Retrieval quality on the HR golden QA set (k=4)", summary)
    report.full_plot(metric_bar(summary, "retriever", [c for c in summary.columns if c != "retriever"],
                                "Retrieval metrics by embedder"), "Retrieval metrics")
    report.full_table("Per-question detail", detail, rows=12)
    report.full_text("Key takeaways", "\n".join([
        "* Hashing & TF-IDF only match overlapping words -> paraphrases score low.",
        "* Neural embeddings capture meaning, which usually wins on natural-language questions.",
        "* Keyword methods remain competitive on exact terms (names, codes) -> combine them (hybrid search).",
        "* Always measure on YOUR data with a golden set before choosing an embedder.",
    ]))
    report.save(__file__, "embedding_comparison_report.html")


if __name__ == "__main__":
    main()
