"""
LangGraph fundamentals — no LLM needed.

Learn:
  * StateGraph: typed state, nodes (functions returning partial updates), edges
  * reducers: `Annotated[list, operator.add]` appends instead of overwriting
  * conditional edges and loops (with a termination condition)
  * streaming node-by-node updates
  * checkpointers: thread_id memory and time-travel through state history
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

import operator  # noqa: E402
from typing import Annotated, TypedDict  # noqa: E402

from langgraph.graph import END, START, StateGraph  # noqa: E402

from lib.utility.genai.frameworks.langgraph import get_checkpointer  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.visualization import mermaid_html  # noqa: E402


class TicketState(TypedDict, total=False):
    ticket: str
    category: str
    attempts: int
    resolved: bool
    log: Annotated[list[str], operator.add]


def classify(state: TicketState) -> dict:
    text = state["ticket"].lower()
    category = "pay" if any(w in text for w in ("salary", "bonus", "pay")) else \
        "conduct" if any(w in text for w in ("harass", "bully", "rude")) else "general"
    return {"category": category, "attempts": 0, "log": [f"classified as {category}"]}


def payroll_bot(state: TicketState) -> dict:
    attempts = state.get("attempts", 0) + 1
    resolved = attempts >= 2  # pretend the first attempt fails
    return {"attempts": attempts, "resolved": resolved, "log": [f"payroll attempt {attempts}: {'ok' if resolved else 'retry'}"]}


def escalate(state: TicketState) -> dict:
    return {"resolved": True, "log": ["escalated to a human HR business partner"]}


def faq_bot(state: TicketState) -> dict:
    return {"resolved": True, "log": ["answered from FAQ"]}


def route(state: TicketState) -> str:
    return {"pay": "payroll_bot", "conduct": "escalate"}.get(state["category"], "faq_bot")


def retry_or_finish(state: TicketState) -> str:
    return END if state["resolved"] or state["attempts"] >= 3 else "payroll_bot"


def build_graph(checkpointer=None):
    graph = StateGraph(TicketState)
    graph.add_node("classify", classify)
    graph.add_node("payroll_bot", payroll_bot)
    graph.add_node("escalate", escalate)
    graph.add_node("faq_bot", faq_bot)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route, ["payroll_bot", "escalate", "faq_bot"])
    graph.add_conditional_edges("payroll_bot", retry_or_finish, ["payroll_bot", END])
    graph.add_edge("escalate", END)
    graph.add_edge("faq_bot", END)
    return graph.compile(checkpointer=checkpointer)


def main():
    app = build_graph()
    tickets = ["My bonus is missing from this month's salary.", "A colleague is rude and bullying me.",
               "Where can I find the holiday calendar?"]
    run_rows = []
    for ticket in tickets:
        final = app.invoke({"ticket": ticket})
        run_rows.append({"ticket": ticket, "category": final["category"], "resolved": final["resolved"],
                         "log": " -> ".join(final["log"])})

    stream_rows = [{"step": i + 1, "node": node, "update": str(update)}
                   for i, chunk in enumerate(app.stream({"ticket": tickets[0]}, stream_mode="updates"))
                   for node, update in chunk.items()]

    memory_app = build_graph(get_checkpointer("memory"))
    config = {"configurable": {"thread_id": "ticket-42"}}
    memory_app.invoke({"ticket": tickets[0]}, config)
    snapshot = memory_app.get_state(config)
    history = [{"checkpoint": h.config["configurable"]["checkpoint_id"][-8:], "next": h.next,
                "log entries": len(h.values.get("log", []))} for h in memory_app.get_state_history(config)]

    report = GenAIReport("LangGraph fundamentals (no LLM)")
    report.full("Graph", mermaid_html(app.get_graph().draw_mermaid()))
    report.full_table("Runs (conditional routing + retry loop)", run_rows)
    report.full_table("stream_mode='updates' — one event per node", stream_rows)
    report.grid([
        report.kv("Checkpointer: final state of thread 'ticket-42'", {k: v for k, v in snapshot.values.items()}),
        report.table("State history (newest first) — time travel", history),
    ])
    report.full_text("Key takeaways", "\n".join([
        "* Nodes return PARTIAL state updates; reducers decide how updates merge (e.g. append logs).",
        "* Conditional edges = routing functions; loops need an explicit stop condition.",
        "* Compile with a checkpointer to get per-thread memory, resumability and state history.",
        "* The same patterns power agents: replace the rule-based nodes with LLM calls (next scripts).",
    ]))
    report.save(__file__, "langgraph_basics_report.html")


if __name__ == "__main__":
    main()
