"""
Naive RAG — retrieve, stuff the context into a prompt, generate an answer with citations.

Learn:
  * the three RAG steps: retrieve -> augment (prompt) -> generate
  * grounding: answer ONLY from the context, say "I don't know" otherwise
  * inline citations [n] mapped back to source pages
  * latency breakdown of a RAG call
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.factory import build_hr_vectorstore, get_llm  # noqa: E402
from lib.utility.genai.prompting import PromptLibrary  # noqa: E402
from lib.utility.genai.rag import NaiveRAG, citation_table  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.retrieval import DenseRetriever  # noqa: E402

QUESTIONS = [
    "What are the key elements of Total Rewards?",
    "Who is responsible for hiring decisions and what is not considered?",
    "What is the capital of France?",  # out of scope -> should refuse
]


def main():
    llm = get_llm()
    store, chunks = build_hr_vectorstore()
    rag = NaiveRAG(DenseRetriever(store), llm, k=3, max_tokens=180)

    report = GenAIReport("Naive RAG over the HR policy")
    report.grid([
        report.kv("Pipeline", {"chunks indexed": len(chunks), "retriever": "dense (nomic-embed-text)", "k": 3,
                               "llm": llm.model, "store loaded from disk": store.loaded_from_disk}),
        report.text("RAG prompt template (config/prompts/rag_answer.txt)", PromptLibrary.get("rag_answer").template),
    ])
    for question in QUESTIONS:
        print(f"RAG: {question}")
        result = rag.ask(question)
        report.grid([
            report.kv(question, {"answer": result.answer, "latency (s)": round(result.latency_s, 2),
                                 **{f"{k} (s)": round(v, 2) for k, v in result.steps.items() if isinstance(v, float)}}),
            report.table("Citations", citation_table(result.answer, result.sources)),
        ])
        report.full_text(f"Context sent to the LLM for: {question}", result.context, max_lines=12)
    report.full_text("Key takeaways", "\n".join([
        "* RAG gives the model fresh / private knowledge without fine-tuning.",
        "* The prompt must force grounding and allow 'I don't know' — otherwise the model fills gaps.",
        "* Numbered context + [n] citations make answers verifiable.",
        "* Naive RAG fails when retrieval fails -> see advanced RAG and evaluation next.",
    ]))
    report.save(__file__, "naive_rag_report.html")


if __name__ == "__main__":
    main()
