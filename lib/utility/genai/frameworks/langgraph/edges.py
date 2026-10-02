from typing import Literal

from .state import RAGState


def route_after_question(state: RAGState) -> Literal["retrieve", "direct_answer"]:
    return "retrieve" if state.get("route") == "vectorstore" else "direct_answer"


def make_decide_after_grading(max_rewrites: int = 1):
    """Conditional edge: generate if we have relevant docs, otherwise rewrite the query (bounded)."""

    def decide_after_grading(state: RAGState) -> Literal["generate", "rewrite_query"]:
        if state.get("relevant_documents"):
            return "generate"
        if state.get("rewrites", 0) < max_rewrites:
            return "rewrite_query"
        return "generate"

    return decide_after_grading
