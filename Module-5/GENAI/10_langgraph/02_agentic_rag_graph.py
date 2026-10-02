"""
Agentic RAG with LangGraph — adaptive routing + corrective retrieval + self-check.

Learn:
  * routing: does the question need the HR documents or not?
  * grading retrieved chunks with the LLM and rewriting the query when none are relevant
  * checking the answer is grounded in the documents (hallucination check)
  * reading the execution trace of a graph run
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.factory import build_hr_vectorstore, get_llm  # noqa: E402
from lib.utility.genai.frameworks.langgraph import build_agentic_rag_graph  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.retrieval import DenseRetriever  # noqa: E402
from lib.utility.genai.visualization import mermaid_html  # noqa: E402

QUESTIONS = [
    "How does the company support gender balance?",
    "What is 2 + 2?",
]


def main():
    llm = get_llm()
    store, _ = build_hr_vectorstore()
    app = build_agentic_rag_graph(DenseRetriever(store), llm, k=3, max_rewrites=1, check_hallucination=True)

    report = GenAIReport("Agentic RAG with LangGraph")
    report.full("Graph", mermaid_html(app.get_graph().draw_mermaid()))
    for question in QUESTIONS:
        print(f"Graph: {question}")
        final = app.invoke({"question": question})
        report.grid([
            report.kv(question, {"route": final.get("route"), "rewrites": final.get("rewrites", 0),
                                 "relevant docs": len(final.get("relevant_documents") or []),
                                 "grounded": final.get("grounded", "n/a"), "answer": final.get("answer")}),
            report.table("Execution trace", [{"step": i + 1, "event": e} for i, e in enumerate(final.get("trace", []))]),
        ])
    report.full_text("Key takeaways", "\n".join([
        "* Adaptive RAG: skip retrieval when the question does not need your documents.",
        "* Corrective RAG: grade chunks; if nothing is relevant, rewrite the query and retry (bounded loop).",
        "* Self-RAG: verify the answer against the context before returning it.",
        "* Each check is an LLM call — graphs make these trade-offs explicit and debuggable.",
    ]))
    report.save(__file__, "agentic_rag_graph_report.html")


if __name__ == "__main__":
    main()
