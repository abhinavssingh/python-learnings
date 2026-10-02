from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ....abstractions.base_llm import BaseLLM
from ....abstractions.base_retriever import BaseRetriever
from ....rag.pipelines import NaiveRAG
from ..state import ReviewState


def build_review_graph(retriever: BaseRetriever, llm: BaseLLM, checkpointer):
    """
    Human-in-the-loop: the graph drafts an answer, then *pauses* with `interrupt()`
    until a human approves, edits or rejects it.

        START -> draft_answer -> human_review (interrupt) -> finalize -> END

    Resume with:  graph.invoke(Command(resume={"action": "approve"}), config)
    """
    if checkpointer is None:
        raise ValueError("Human-in-the-loop graphs need a checkpointer to pause and resume.")
    rag = NaiveRAG(retriever, llm, k=3, max_tokens=150)

    def draft_answer(state: ReviewState) -> dict:
        result = rag.ask(state["question"])
        return {"draft": result.answer, "sources": [r.metadata.get("page") for r in result.sources],
                "trace": ["draft created"]}

    def human_review(state: ReviewState) -> dict:
        decision = interrupt({"question": state["question"], "draft": state["draft"],
                              "instructions": "Reply with {'action': 'approve'|'edit'|'reject', 'text': '...'}"})
        return {"decision": decision, "trace": [f"human decision: {decision.get('action')}"]}

    def finalize(state: ReviewState) -> dict:
        decision = state.get("decision") or {}
        action = decision.get("action", "approve")
        if action == "edit":
            final = decision.get("text", state["draft"])
        elif action == "reject":
            final = "This answer was rejected by a reviewer. Please contact HR directly."
        else:
            final = state["draft"]
        return {"final_answer": final, "trace": [f"finalized ({action})"]}

    graph = StateGraph(ReviewState)
    graph.add_node("draft_answer", draft_answer)
    graph.add_node("human_review", human_review)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "draft_answer")
    graph.add_edge("draft_answer", "human_review")
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
