"""
Human-in-the-loop with LangGraph `interrupt()`.

Learn:
  * pausing a graph mid-run to wait for a human decision
  * inspecting the paused state (`get_state`, `.next`, interrupt payload)
  * resuming with `Command(resume=...)`: approve / edit / reject
  * why a checkpointer is mandatory for interrupts

Set GENAI_INTERACTIVE=1 to type the decisions yourself; otherwise decisions are simulated.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from langgraph.types import Command  # noqa: E402

from lib.utility.genai.factory import build_hr_vectorstore, get_llm  # noqa: E402
from lib.utility.genai.frameworks.langgraph import build_review_graph, get_checkpointer  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.retrieval import DenseRetriever  # noqa: E402
from lib.utility.genai.visualization import mermaid_html  # noqa: E402

SCENARIOS = [
    ("What are promotions based on?", {"action": "approve"}),
    ("Does the company tolerate harassment?",
     {"action": "edit", "text": "No. Harassment and discrimination are not tolerated. Report issues to HR or your manager."}),
]


def ask_human(payload: dict, simulated: dict) -> dict:
    if os.getenv("GENAI_INTERACTIVE") != "1":
        return simulated
    print(f"\nQuestion: {payload['question']}\nDraft: {payload['draft']}")
    action = input("approve / edit / reject > ").strip().lower() or "approve"
    return {"action": action, "text": input("Edited answer > ") if action == "edit" else ""}


def main():
    llm = get_llm()
    store, _ = build_hr_vectorstore()
    app = build_review_graph(DenseRetriever(store), llm, checkpointer=get_checkpointer("memory"))

    report = GenAIReport("Human-in-the-loop review (LangGraph interrupt)")
    report.full("Graph", mermaid_html(app.get_graph().draw_mermaid()))
    for i, (question, simulated) in enumerate(SCENARIOS, start=1):
        config = {"configurable": {"thread_id": f"review-{i}"}}
        print(f"Drafting: {question}")
        paused = app.invoke({"question": question}, config)
        payload = paused["__interrupt__"][0].value
        waiting_at = app.get_state(config).next

        decision = ask_human(payload, simulated)
        final = app.invoke(Command(resume=decision), config)
        report.grid([
            report.kv(f"Paused run: {question}", {"graph paused before": waiting_at, "draft": payload["draft"],
                                                  "source pages": final.get("sources")}),
            report.kv("After human decision", {"decision": decision.get("action"), "final answer": final["final_answer"],
                                               "trace": " -> ".join(final.get("trace", []))}),
        ])
    report.full_text("Key takeaways", "\n".join([
        "* interrupt() saves the state via the checkpointer and returns control to your app.",
        "* The same thread_id + Command(resume=...) continues exactly where it stopped — even hours later.",
        "* Use it for approvals: sending emails, updating records, answering sensitive HR questions.",
    ]))
    report.save(__file__, "human_in_the_loop_report.html")


if __name__ == "__main__":
    main()
