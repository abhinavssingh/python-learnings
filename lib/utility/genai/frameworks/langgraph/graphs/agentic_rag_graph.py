from langgraph.graph import END, START, StateGraph

from ....abstractions.base_llm import BaseLLM
from ....abstractions.base_retriever import BaseRetriever
from ..edges import make_decide_after_grading, route_after_question
from ..nodes import (
    make_direct_answer_node,
    make_generate_node,
    make_grade_node,
    make_hallucination_node,
    make_retrieve_node,
    make_rewrite_node,
    make_route_node,
)
from ..state import RAGState


def build_agentic_rag_graph(retriever: BaseRetriever, llm: BaseLLM, k: int = 4, max_rewrites: int = 1,
                            check_hallucination: bool = True, checkpointer=None):
    """
    Adaptive + Corrective + Self-RAG in one graph:

        START -> route_question -+-> retrieve -> grade_documents -+-> generate -> check_grounded -> END
                                 |        ^                       |
                                 |        +---- rewrite_query <---+  (no relevant docs)
                                 +-> direct_answer -> END
    """
    graph = StateGraph(RAGState)
    graph.add_node("route_question", make_route_node(llm))
    graph.add_node("retrieve", make_retrieve_node(retriever, k))
    graph.add_node("grade_documents", make_grade_node(llm))
    graph.add_node("rewrite_query", make_rewrite_node(llm))
    graph.add_node("generate", make_generate_node(llm))
    graph.add_node("direct_answer", make_direct_answer_node(llm))

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_question, ["retrieve", "direct_answer"])
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", make_decide_after_grading(max_rewrites), ["generate", "rewrite_query"])
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("direct_answer", END)

    if check_hallucination:
        graph.add_node("check_grounded", make_hallucination_node(llm))
        graph.add_edge("generate", "check_grounded")
        graph.add_edge("check_grounded", END)
    else:
        graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)
