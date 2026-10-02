"""
Advanced RAG — improving each stage of the pipeline.

Learn:
  * pre-retrieval: query rewriting
  * retrieval: hybrid search (dense + BM25) with over-fetching
  * post-retrieval: LLM re-ranking and contextual compression
  * comparing naive vs advanced on the same questions (answer, pages, latency)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.datasets import load_hr_qa  # noqa: E402
from lib.utility.genai.evaluation import generation_scores  # noqa: E402
from lib.utility.genai.factory import build_hr_vectorstore, get_embedder, get_llm  # noqa: E402
from lib.utility.genai.rag import AdvancedRAG, NaiveRAG  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.retrieval import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402
from lib.utility.genai.visualization import mermaid_html  # noqa: E402

PIPELINE_DIAGRAM = """flowchart LR
  Q[question] --> RW[rewrite query]
  RW --> H[hybrid retrieve<br/>dense + BM25, fetch_k=8]
  H --> RR[LLM re-rank -> top 3]
  RR --> C[compress context]
  C --> G[generate with citations]
"""


def main():
    llm = get_llm()
    store, chunks = build_hr_vectorstore()
    dense = DenseRetriever(store)
    hybrid = HybridRetriever([dense, BM25Retriever(chunks)], fetch_k=10)

    naive = NaiveRAG(dense, llm, k=3, max_tokens=150)
    advanced = AdvancedRAG(hybrid, llm, k=3, fetch_k=8, rewrite_query=True, rerank=True,
                           compressor_embedder=get_embedder(), compression_threshold=0.5, max_tokens=150)

    items = [load_hr_qa()[i] for i in (6, 11)]  # feedback tools, gender balance

    rows = []
    report = GenAIReport("Advanced RAG vs Naive RAG")
    report.full("Advanced pipeline", mermaid_html(PIPELINE_DIAGRAM))
    for item in items:
        for name, pipeline in (("naive", naive), ("advanced", advanced)):
            print(f"{name}: {item['question']}")
            result = pipeline.ask(item["question"])
            scores = generation_scores(result.answer, item["reference_answer"], item["expected_keywords"])
            rows.append({"question": item["question"], "pipeline": name, "answer": result.answer,
                         "pages": [r.metadata.get("page") for r in result.sources],
                         "expected pages": item["relevant_pages"], "keyword recall": round(scores["keyword_recall"], 2),
                         "context chars": len(result.context), "latency (s)": round(result.latency_s, 1)})
            if name == "advanced":
                report.grid([
                    report.kv(f"Advanced steps: {item['question']}",
                              {k: (round(v, 2) if isinstance(v, float) else v) for k, v in result.steps.items()}),
                    report.text("Compressed context", result.context, max_lines=12),
                ])
    report.full_table("Side-by-side comparison", rows)
    report.full_text("Key takeaways", "\n".join([
        "* Every extra stage costs latency (each LLM call ~ seconds on a laptop) — add stages that measurably help.",
        "* Hybrid + re-rank mainly improves RETRIEVAL precision; compression shrinks the prompt.",
        "* Query rewriting shines for vague, conversational questions.",
        "* Use 11_evaluation to compare pipelines on the whole golden set, not two questions.",
    ]))
    report.save(__file__, "advanced_rag_report.html")


if __name__ == "__main__":
    main()
