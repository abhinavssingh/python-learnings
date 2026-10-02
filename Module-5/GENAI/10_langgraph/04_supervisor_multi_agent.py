"""
Multi-agent supervisor pattern with LangGraph.

Learn:
  * a supervisor LLM that decides which worker acts next (JSON routing)
  * specialised workers: researcher (RAG), calculator, writer
  * shared state with accumulated findings and an iteration limit
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.factory import build_hr_vectorstore, get_llm  # noqa: E402
from lib.utility.genai.frameworks.langgraph import build_supervisor_graph  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.retrieval import DenseRetriever  # noqa: E402
from lib.utility.genai.visualization import mermaid_html  # noqa: E402

TASK = ("Find out which elements make up Total Rewards in the HR policy. Then calculate the total annual pay "
        "for someone with a fixed pay of 60000 and variable pay of 15% of fixed pay.")


def main():
    llm = get_llm()
    store, _ = build_hr_vectorstore()
    app = build_supervisor_graph(DenseRetriever(store), llm, max_iterations=4)

    print("Running supervisor graph...")
    final = app.invoke({"task": TASK})

    report = GenAIReport("Supervisor multi-agent graph (LangGraph)")
    report.full("Graph", mermaid_html(app.get_graph().draw_mermaid()))
    report.grid([
        report.kv("Task", {"task": TASK, "iterations": final.get("iterations"), "final answer": final.get("final_answer")}),
        report.table("Findings from workers", [{"finding": f} for f in final.get("findings", [])]),
    ])
    report.full_table("Execution trace", [{"step": i + 1, "event": e} for i, e in enumerate(final.get("trace", []))])
    report.full_text("Key takeaways", "\n".join([
        "* A supervisor decomposes the task and delegates to workers with narrow responsibilities.",
        "* Workers can be different prompts, tools, retrievers or even different models.",
        "* Always bound the number of iterations — LLM routers can loop.",
        "* Deterministic tools (calculator) beat the LLM at arithmetic — route to them.",
    ]))
    report.save(__file__, "supervisor_multi_agent_report.html")


if __name__ == "__main__":
    main()
