import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import MessagesState


class RAGState(TypedDict, total=False):
    """Shared state that flows through the agentic RAG graph."""

    question: str
    search_query: str
    route: str
    documents: list[Any]
    relevant_documents: list[Any]
    answer: str
    grounded: bool
    rewrites: int
    # Reducer: every node's trace entries are appended instead of overwritten.
    trace: Annotated[list[str], operator.add]


class SupervisorState(TypedDict, total=False):
    task: str
    next: str
    instruction: str
    findings: Annotated[list[str], operator.add]
    iterations: int
    final_answer: str
    trace: Annotated[list[str], operator.add]


class ReviewState(TypedDict, total=False):
    question: str
    draft: str
    sources: list[Any]
    decision: dict[str, Any]
    final_answer: str
    trace: Annotated[list[str], operator.add]


class AgentState(MessagesState):
    """Chat messages (with the add_messages reducer) for tool-calling agents."""
