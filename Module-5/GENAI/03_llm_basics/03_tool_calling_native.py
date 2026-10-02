"""
Native tool (function) calling with Ollama — the core of every agent.

Learn:
  * describing Python functions as JSON-schema tools (@tool decorator)
  * the agent loop: model requests a tool -> we execute it -> result goes back -> final answer
  * combining a calculator, a clock and a document-search tool (RAG as a tool)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

import json  # noqa: E402

from lib.utility.genai.factory import build_hr_vectorstore, get_llm  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.retrieval import DenseRetriever  # noqa: E402
from lib.utility.genai.tools import (  # noqa: E402
    ToolCallingAgent,
    calculator,
    current_datetime,
    make_document_search_tool,
    word_count,
)

QUESTIONS = [
    "If my fixed pay is 48000 and variable pay is 15% of it, what is my total pay?",
    "According to the HR policy, what are promotions based on?",
]


def main():
    llm = get_llm()
    store, _ = build_hr_vectorstore()
    tools = [calculator, current_datetime, word_count, make_document_search_tool(DenseRetriever(store), k=2)]
    agent = ToolCallingAgent(llm, tools, max_steps=4)

    report = GenAIReport("Native tool calling with Ollama")
    report.full_text("Tool schemas sent to the model", json.dumps([t.to_schema() for t in tools], indent=2), max_lines=30)
    for question in QUESTIONS:
        print(f"Agent: {question}")
        result = agent.run(question)
        report.grid([
            report.kv("Question / answer", {"question": question, "answer": result.answer, "steps": len(result.steps)}),
            report.table("Agent trace", [{"step": s.step, "kind": s.kind, "content": s.content[:300]} for s in result.steps]),
        ])
    report.full_text("Key takeaways", "\n".join([
        "* The model never runs code — it returns a structured tool call; YOUR code executes it.",
        "* Good tool names, descriptions and typed parameters matter more than prompt tricks.",
        "* Always cap the loop (max_steps) and validate arguments (safe_eval instead of eval).",
        "* Retrieval exposed as a tool = 'agentic RAG': the model decides when to search.",
    ]))
    report.save(__file__, "tool_calling_native_report.html")


if __name__ == "__main__":
    main()
