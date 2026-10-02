"""
Query transformations, re-ranking and contextual compression (LLM-assisted retrieval).

Learn:
  * query rewriting, multi-query expansion, HyDE, step-back and decomposition
  * LLM re-ranking of an over-fetched candidate list
  * contextual compression: keep only sentences relevant to the question
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.factory import build_hr_vectorstore, get_embedder, get_llm  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.retrieval import (  # noqa: E402
    DenseRetriever,
    EmbeddingContextCompressor,
    HyDERetriever,
    LLMReranker,
    MultiQueryRetriever,
    QueryDecomposer,
    QueryRewriter,
    StepBackGenerator,
)

VAGUE_QUESTION = "how do i get ahead here?"
COMPLEX_QUESTION = "How are employees rewarded and how are they trained?"


def pages(results):
    return [r.metadata.get("page") for r in results]


def main():
    llm = get_llm()
    store, _ = build_hr_vectorstore()
    embedder = get_embedder()
    dense = DenseRetriever(store)

    rewritten = QueryRewriter(llm)(VAGUE_QUESTION)
    step_back = StepBackGenerator(llm)(VAGUE_QUESTION)
    sub_questions = QueryDecomposer(llm, n=2)(COMPLEX_QUESTION)

    multi = MultiQueryRetriever(dense, llm, n=3)
    multi_results = multi.retrieve(VAGUE_QUESTION, k=4)
    hyde = HyDERetriever(store, llm)
    hyde_results = hyde.retrieve(VAGUE_QUESTION, k=4)

    comparison = [
        {"strategy": "original query", "query used": VAGUE_QUESTION, "pages": pages(dense.retrieve(VAGUE_QUESTION, 4))},
        {"strategy": "rewritten", "query used": rewritten, "pages": pages(dense.retrieve(rewritten, 4))},
        {"strategy": "step-back", "query used": step_back, "pages": pages(dense.retrieve(step_back, 4))},
        {"strategy": "multi-query (RRF)", "query used": " | ".join(multi.last_queries), "pages": pages(multi_results)},
        {"strategy": "HyDE", "query used": hyde.last_hypothesis[:200], "pages": pages(hyde_results)},
    ]

    candidates = dense.retrieve(COMPLEX_QUESTION, k=8)
    reranker = LLMReranker(llm)
    reranked = reranker.rerank(COMPLEX_QUESTION, candidates, top_n=4)
    compressed = EmbeddingContextCompressor(embedder, threshold=0.55).compress(COMPLEX_QUESTION, reranked)
    before_chars = sum(len(r.text) for r in reranked)
    after_chars = sum(len(r.text) for r in compressed)

    report = GenAIReport("Query transformations, re-ranking & compression")
    report.full_table("Same vague question, different query strategies (dense top-4 pages)", comparison)
    report.grid([
        report.kv("Decomposition", {"complex question": COMPLEX_QUESTION,
                                    **{f"sub-question {i + 1}": q for i, q in enumerate(sub_questions)}}),
        report.kv("Why these help", {"rewrite": "fix vague / conversational phrasing",
                                     "step-back": "ask the broader principle first",
                                     "multi-query": "several phrasings -> higher recall",
                                     "HyDE": "embed a fake ANSWER; answers look like documents",
                                     "decompose": "retrieve for each part separately"}),
    ])
    report.grid([
        report.table("Dense top-8 (before re-ranking)", [
            {"rank": i + 1, "page": r.metadata.get("page"), "score": round(r.score, 3), "chunk": r.document.preview(70)}
            for i, r in enumerate(candidates)]),
        report.table("LLM re-ranked top-4", [
            {"rank": i + 1, "page": r.metadata.get("page"), "llm score": r.score, "chunk": r.document.preview(70)}
            for i, r in enumerate(reranked)]),
    ])
    report.full_table(f"Contextual compression: {before_chars} -> {after_chars} characters", [
        {"page": r.metadata.get("page"), "kept text": r.text} for r in compressed])
    report.full_text("Key takeaways", "\n".join([
        "* Users ask vague questions — transforming the query is often cheaper than a better embedder.",
        "* Each transformation costs an LLM call: measure the gain before paying the latency.",
        "* Over-fetch (k=8..20) then re-rank to 4: recall from the retriever, precision from the re-ranker.",
        "* Compression shrinks the prompt -> faster, cheaper and less distracting context for the LLM.",
    ]))
    report.save(__file__, "query_transformations_report.html")


if __name__ == "__main__":
    main()
