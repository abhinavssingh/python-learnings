"""
Retrieval strategies — dense, sparse, hybrid and diverse — measured on a golden QA set.

Learn:
  * dense (embedding) retrieval vs BM25 keyword retrieval
  * hybrid retrieval with Reciprocal Rank Fusion (RRF)
  * MMR (maximal marginal relevance) to reduce redundant chunks
  * retrieval metrics: hit@k, precision@k, recall@k, MRR, nDCG
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.datasets import load_hr_qa  # noqa: E402
from lib.utility.genai.evaluation import RetrievalEvaluator  # noqa: E402
from lib.utility.genai.factory import build_hr_vectorstore  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, MMRRetriever  # noqa: E402
from lib.utility.genai.visualization import metric_bar  # noqa: E402

QUESTION = "What tools give employees feedback on their performance?"


def main():
    store, chunks = build_hr_vectorstore()
    dense = DenseRetriever(store)
    bm25 = BM25Retriever(chunks)
    retrievers = {
        "dense": dense,
        "bm25": bm25,
        "hybrid (RRF)": HybridRetriever([dense, bm25], weights=[1.0, 1.0], fetch_k=10),
        "mmr (lambda=0.5)": MMRRetriever(store, fetch_k=12, lambda_mult=0.5),
    }

    side_by_side = []
    for name, retriever in retrievers.items():
        for rank, r in enumerate(retriever.retrieve(QUESTION, k=4), start=1):
            side_by_side.append({"retriever": name, "rank": rank, "page": r.metadata.get("page"),
                                 "score": round(r.score, 4), "chunk": r.document.preview(90)})

    dataset = load_hr_qa()
    summary, detail = RetrievalEvaluator(dataset, k=4).compare(retrievers)
    summary = summary.reset_index()
    best = summary.sort_values("mrr", ascending=False).iloc[0]["retriever"] if "mrr" in summary else summary.iloc[0, 0]

    report = GenAIReport("Retrieval strategies: dense vs BM25 vs hybrid vs MMR")
    report.grid([
        report.kv("Setup", {"chunks": len(chunks), "golden questions": len(dataset), "k": 4,
                            "relevance": "chunk comes from one of the item's relevant_pages"}),
        report.table("Metrics (mean over questions)", summary),
    ])
    report.full_plot(metric_bar(summary, "retriever", [c for c in summary.columns if c != "retriever"],
                                "Retrieval quality by strategy"), "Retrieval metrics")
    report.full_table(f"Side-by-side results for: '{QUESTION}'", side_by_side, rows=16)
    report.full_table("Per-question detail", detail, rows=15)
    report.full_text("Key takeaways", "\n".join([
        f"* Best MRR on this dataset: {best}.",
        "* Dense retrieval understands paraphrases; BM25 nails exact terms (PDG, Rive-Reine, 360).",
        "* Hybrid (RRF) fuses RANKS, so the two score scales never need calibrating.",
        "* MMR trades a bit of relevance for diversity — useful when chunks overlap heavily.",
        "* Tiny golden sets are noisy: grow yours to 50-100 questions before trusting small differences.",
    ]))
    report.save(__file__, "retrieval_strategies_report.html")


if __name__ == "__main__":
    main()
