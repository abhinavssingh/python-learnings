from langgraph.graph import END, START, StateGraph

from ....abstractions.base_llm import BaseLLM
from ....abstractions.base_retriever import BaseRetriever
from ....prompting.output_parsers import JsonOutputParser
from ....rag.pipelines import NaiveRAG
from ....tools.builtin_tools import safe_eval
from ..state import SupervisorState

WORKERS = {
    "researcher": "looks up facts in the company HR policy documents",
    "calculator": "evaluates arithmetic (numbers, percentages, totals)",
    "writer": "writes the final answer for the user once enough findings exist",
}

SUPERVISOR_PROMPT = """You are a supervisor coordinating workers to complete a task.
Workers:
{workers}

Task: {task}

Findings so far:
{findings}

Choose the next worker. Pick "writer" when the findings are enough to answer the task.
Respond with JSON only: {{"next": "<researcher|calculator|writer>", "instruction": "<what the worker should do>"}}"""


def build_supervisor_graph(retriever: BaseRetriever, llm: BaseLLM, max_iterations: int = 4, checkpointer=None):
    """
    Multi-agent supervisor pattern:

        START -> supervisor -> (researcher | calculator) -> supervisor -> ... -> writer -> END
    """
    rag = NaiveRAG(retriever, llm, k=3, max_tokens=150)
    parser = JsonOutputParser()

    def supervisor(state: SupervisorState) -> dict:
        iterations = state.get("iterations", 0)
        if iterations >= max_iterations:
            return {"next": "writer", "instruction": "Summarise the findings.", "iterations": iterations + 1,
                    "trace": ["supervisor -> writer (iteration limit)"]}
        findings = "\n".join(f"- {f}" for f in state.get("findings", [])) or "(none)"
        workers = "\n".join(f"- {name}: {desc}" for name, desc in WORKERS.items())
        response = llm.generate(SUPERVISOR_PROMPT.format(workers=workers, task=state["task"], findings=findings),
                                max_tokens=80, format="json")
        data = parser.safe_parse(response.text, default={}) or {}
        nxt = data.get("next") if data.get("next") in WORKERS else "writer"
        instruction = data.get("instruction") or state["task"]
        return {"next": nxt, "instruction": instruction, "iterations": iterations + 1,
                "trace": [f"supervisor -> {nxt}: {instruction}"]}

    def researcher(state: SupervisorState) -> dict:
        result = rag.ask(state["instruction"])
        return {"findings": [f"researcher: {result.answer}"], "trace": ["researcher answered"]}

    def calculator(state: SupervisorState) -> dict:
        response = llm.generate(
            "Convert the request into a single arithmetic expression using only numbers and + - * / ( ).\n"
            f"Request: {state['instruction']}\nContext: {' '.join(state.get('findings', []))}\n"
            'Respond with JSON only: {"expression": "..."}', max_tokens=60, format="json")
        expression = (parser.safe_parse(response.text, default={}) or {}).get("expression", "")
        try:
            value = safe_eval(expression)
            finding = f"calculator: {expression} = {round(value, 4)}"
        except Exception as exc:
            finding = f"calculator: could not evaluate '{expression}' ({exc})"
        return {"findings": [finding], "trace": [finding]}

    def writer(state: SupervisorState) -> dict:
        findings = "\n".join(state.get("findings", [])) or "(no findings)"
        answer = llm.generate(f"Task: {state['task']}\n\nFindings:\n{findings}\n\n"
                              "Write a concise final answer (max 4 sentences) using only the findings.",
                              max_tokens=180).text
        return {"final_answer": answer, "trace": ["writer produced final answer"]}

    graph = StateGraph(SupervisorState)
    for name, fn in (("supervisor", supervisor), ("researcher", researcher), ("calculator", calculator), ("writer", writer)):
        graph.add_node(name, fn)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: s["next"], list(WORKERS))
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("calculator", "supervisor")
    graph.add_edge("writer", END)
    return graph.compile(checkpointer=checkpointer)
