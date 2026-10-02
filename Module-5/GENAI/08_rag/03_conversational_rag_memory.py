"""
Conversational RAG and conversation memory.

Learn:
  * memory strategies: buffer, sliding window, summary, vector (semantic recall)
  * condensing a follow-up ("what about promotions?") into a standalone question
  * multi-turn RAG where each turn retrieves with the condensed question
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.factory import build_hr_vectorstore, get_embedder, get_llm  # noqa: E402
from lib.utility.genai.memory import BufferMemory, SummaryMemory, VectorMemory, WindowMemory  # noqa: E402
from lib.utility.genai.rag import ConversationalRAG  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.retrieval import DenseRetriever  # noqa: E402

HISTORY = [
    ("I'm Priya, I joined the Pune office as a data analyst.", "Welcome Priya! How can I help?"),
    ("How many days of annual leave do I get?", "Annual leave depends on your country's local policy."),
    ("I prefer learning through online courses.", "Noted — online learning options are available."),
    ("Can I work from home on Fridays?", "Flexible working is supported; agree it with your manager."),
    ("What's the dress code?", "Business casual in most offices."),
]

CONVERSATION = [
    "How are employees trained at the company?",
    "And how is their performance evaluated?",
    "What are promotions based on then?",
]


def main():
    llm = get_llm()
    embedder = get_embedder()

    memories = {"buffer": BufferMemory(), "window (k=2)": WindowMemory(k=2), "vector (k=2)": VectorMemory(embedder, k=2),
                "summary (keep_last=2)": SummaryMemory(llm, keep_last=2)}
    for memory in memories.values():
        for user, assistant in HISTORY:
            memory.add_exchange(user, assistant)
    probe = "Where do I work and how do I like to learn?"
    memory_rows = [{"memory": name, "messages sent to LLM": len(m.messages(probe)), "context": m.as_text(probe)}
                   for name, m in memories.items()]

    store, _ = build_hr_vectorstore()
    rag = ConversationalRAG(DenseRetriever(store), llm, memory=WindowMemory(k=3), k=3, max_tokens=150)
    turns = []
    for question in CONVERSATION:
        print(f"Turn: {question}")
        result = rag.ask(question)
        turns.append({"user": question, "standalone question": result.steps.get("standalone_question", question),
                      "pages": [r.metadata.get("page") for r in result.sources], "answer": result.answer})

    report = GenAIReport("Conversational RAG & memory")
    report.full_table(f"Memory strategies after {len(HISTORY)} exchanges (probe: '{probe}')", memory_rows)
    report.full_table("Multi-turn RAG with question condensing", turns)
    report.full_text("Key takeaways", "\n".join([
        "* Buffer memory grows without bound -> cost and context-window problems.",
        "* Window memory is cheap but forgets early facts (Priya's name / office).",
        "* Summary memory compresses old turns with an extra LLM call.",
        "* Vector memory recalls only the turns relevant to the current question.",
        "* For RAG, ALWAYS rewrite follow-ups into standalone questions before retrieving —",
        "  'what about promotions then?' alone retrieves poorly.",
    ]))
    report.save(__file__, "conversational_rag_report.html")


if __name__ == "__main__":
    main()
