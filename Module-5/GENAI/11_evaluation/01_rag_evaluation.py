"""
Evaluating RAG end-to-end on a golden QA set.

Learn:
  * retrieval metrics per question (hit, MRR against relevant pages)
  * reference-based generation metrics: token F1, ROUGE-L, BLEU, keyword recall
  * reference-free LLM-as-judge: faithfulness, answer relevance, context relevance (1-5)
  * comparing pipelines (naive vs advanced) on quality AND latency

Set GENAI_EVAL_LIMIT to evaluate more questions (default 4 — local LLMs are slow).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

import pandas as pd  # noqa: E402

from lib.utility.genai.datasets import load_hr_qa  # noqa: E402
from lib.utility.genai.evaluation import LLMJudge, RAGEvaluator, bleu, exact_match, keyword_recall, rouge_l, token_f1  # noqa: E402
from lib.utility.genai.factory import build_hr_vectorstore, get_llm  # noqa: E402
from lib.utility.genai.rag import AdvancedRAG, NaiveRAG  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.retrieval import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402
from lib.utility.genai.visualization import metric_bar  # noqa: E402

REFERENCE = "Promotions are based on sustained performance and future potential."
CANDIDATES = [
    "Promotions are based on sustained performance and future potential.",
    "Promotion depends on long-term performance, results and behaviour, plus potential.",
    "Employees are promoted based on seniority.",
]


def main():
    limit = int(os.getenv("GENAI_EVAL_LIMIT", "4"))
    llm = get_llm()
    store, chunks = build_hr_vectorstore()
    dense = DenseRetriever(store)

    pipelines = {
        "naive (dense k=3)": NaiveRAG(dense, llm, k=3, max_tokens=150),
        "advanced (hybrid + rerank)": AdvancedRAG(HybridRetriever([dense, BM25Retriever(chunks)], fetch_k=10), llm,
                                                  k=3, fetch_k=6, rewrite_query=False, rerank=True, max_tokens=150),
    }
    evaluator = RAGEvaluator(load_hr_qa(limit), judge=LLMJudge(llm))
    frames = [evaluator.evaluate(p, name)[0] for name, p in pipelines.items()]
    summary = RAGEvaluator.summary(frames)
    detail = pd.concat(frames, ignore_index=True)

    metric_rows = [{"candidate": c, "exact": exact_match(c, REFERENCE), "token F1": round(token_f1(c, REFERENCE), 3),
                    "ROUGE-L": round(rouge_l(c, REFERENCE), 3), "BLEU": round(bleu(c, REFERENCE), 3),
                    "keyword recall": keyword_recall(c, ["sustained performance", "potential"])} for c in CANDIDATES]

    quality_cols = [c for c in ("hit", "mrr", "token_f1", "rouge_l", "keyword_recall") if c in summary]
    judge_cols = [c for c in ("faithfulness", "answer_relevance", "context_relevance") if c in summary]

    report = GenAIReport("RAG evaluation: naive vs advanced")
    report.full_table("Understanding generation metrics (reference: '" + REFERENCE + "')", metric_rows)
    report.full_table(f"Pipeline summary ({limit} questions)", summary)
    report.plots([
        (metric_bar(summary, "pipeline", quality_cols, "Retrieval & reference metrics (0-1)"), "Quality"),
        (metric_bar(summary, "pipeline", judge_cols, "LLM-as-judge (1-5)"), "Judge"),
    ])
    report.full_table("Per-question detail", detail, rows=12)
    report.full_text("Key takeaways", "\n".join([
        "* Exact match is too strict for generated text; F1 / ROUGE reward overlap but miss paraphrases.",
        "* LLM judges capture meaning and faithfulness but are themselves noisy — use temperature 0 and rubrics.",
        "* Evaluate retrieval separately from generation: bad answers usually start with bad retrieval.",
        "* Track latency next to quality; run the eval on every change to prompts, chunking or models.",
    ]))
    report.save(__file__, "rag_evaluation_report.html")


if __name__ == "__main__":
    main()
