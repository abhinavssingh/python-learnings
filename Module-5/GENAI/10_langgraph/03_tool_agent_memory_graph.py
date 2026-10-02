"""
ReAct tool agent as a LangGraph graph, with persistent conversation memory.

Learn:
  * MessagesState + ToolNode + tools_condition (prebuilt building blocks)
  * thread_id: separate conversations share one compiled graph
  * SQLite checkpointer: memory that survives restarts (saved_models/genai/checkpoints)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

import uuid  # noqa: E402

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from lib.utility.genai.config import GenAIConfig  # noqa: E402
from lib.utility.genai.factory import build_hr_vectorstore  # noqa: E402
from lib.utility.genai.frameworks.langchain import get_chat_model  # noqa: E402
from lib.utility.genai.frameworks.langgraph import build_tool_agent_graph, get_checkpointer  # noqa: E402
from lib.utility.genai.rag import format_context  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.retrieval import DenseRetriever  # noqa: E402
from lib.utility.genai.tools import safe_eval  # noqa: E402
from lib.utility.genai.visualization import mermaid_html  # noqa: E402

_retriever = None


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '50000 * 0.12'."""
    return str(safe_eval(expression))


@tool
def search_hr_policy(query: str) -> str:
    """Search the company HR policy and return the most relevant passages."""
    return format_context(_retriever.retrieve(query, 2))


TURNS = [
    "Hi, I'm Arjun. My fixed pay is 50000. What would 12% variable pay on top of it make in total?",
    "What does the HR policy say promotions are based on?",
    "What is my name and what total pay did you calculate for me?",
]


def message_rows(messages):
    rows = []
    for m in messages:
        calls = ", ".join(f"{c['name']}({c['args']})" for c in getattr(m, "tool_calls", []) or [])
        rows.append({"type": m.type, "content": str(m.content)[:250], "tool calls": calls})
    return rows


def main():
    global _retriever
    store, _ = build_hr_vectorstore()
    _retriever = DenseRetriever(store)
    config = GenAIConfig.from_env()
    checkpointer = get_checkpointer("sqlite", config.checkpoint_dir / "tool_agent.sqlite")
    app = build_tool_agent_graph(get_chat_model(num_predict=200), [calculator, search_hr_policy], checkpointer=checkpointer)

    thread = {"configurable": {"thread_id": f"arjun-{uuid.uuid4().hex[:6]}"}}
    report = GenAIReport("Tool agent graph with memory (LangGraph)")
    report.full("Graph", mermaid_html(app.get_graph().draw_mermaid()))
    turn_rows = []
    for turn in TURNS:
        print(f"User: {turn}")
        state = app.invoke({"messages": [HumanMessage(turn)]}, thread)
        turn_rows.append({"user": turn, "assistant": state["messages"][-1].content})
    report.full_table("Conversation (same thread_id)", turn_rows)

    other = app.invoke({"messages": [HumanMessage("What is my name?")]},
                       {"configurable": {"thread_id": "someone-else"}})
    report.grid([
        report.kv("Different thread_id -> no shared memory", {"answer": other["messages"][-1].content}),
        report.kv("Persistence", {"checkpointer": "SqliteSaver", "database": str(config.checkpoint_dir / "tool_agent.sqlite"),
                                  "thread": thread["configurable"]["thread_id"],
                                  "messages stored": len(app.get_state(thread).values["messages"])}),
    ])
    report.full_table("Full message history of the thread", message_rows(app.get_state(thread).values["messages"]), rows=20)
    report.full_text("Key takeaways", "\n".join([
        "* agent -> (tool calls?) -> tools -> agent ... is the ReAct loop expressed as a graph.",
        "* The checkpointer stores every message per thread_id -> multi-user chat memory for free.",
        "* SQLite (or Postgres) checkpointers make conversations resumable after a restart.",
    ]))
    report.save(__file__, "tool_agent_graph_report.html")


if __name__ == "__main__":
    main()
