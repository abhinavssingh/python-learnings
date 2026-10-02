from typing import Callable

from ...abstractions.base_llm import BaseLLM
from ...abstractions.base_retriever import BaseRetriever
from ...prompting.output_parsers import JsonOutputParser, YesNoParser
from ...prompting.templates import PromptLibrary
from ...rag.citation import format_context
from ...retrieval.query_transform import QueryRewriter
from .state import RAGState

Node = Callable[[RAGState], dict]


def make_route_node(llm: BaseLLM) -> Node:
    prompt = PromptLibrary.get("route_question")

    def route_question(state: RAGState) -> dict:
        response = llm.generate(prompt.format(question=state["question"]), max_tokens=20, format="json")
        data = JsonOutputParser().safe_parse(response.text, default={}) or {}
        route = "direct" if str(data.get("datasource", "")).lower() == "direct" else "vectorstore"
        return {"route": route, "search_query": state["question"], "rewrites": 0, "trace": [f"route -> {route}"]}

    return route_question


def make_retrieve_node(retriever: BaseRetriever, k: int = 4) -> Node:
    def retrieve(state: RAGState) -> dict:
        query = state.get("search_query") or state["question"]
        docs = retriever.retrieve(query, k)
        return {"documents": docs, "trace": [f"retrieve '{query}' -> {len(docs)} docs"]}

    return retrieve


def make_grade_node(llm: BaseLLM) -> Node:
    """Corrective-RAG step: the LLM keeps only documents relevant to the question."""
    prompt = PromptLibrary.get("grade_document")
    parser = YesNoParser("relevant")

    def grade_documents(state: RAGState) -> dict:
        relevant = []
        for doc in state.get("documents", []):
            response = llm.generate(prompt.format(document=doc.text[:1200], question=state["question"]),
                                    max_tokens=15, format="json")
            if parser.parse(response.text):
                relevant.append(doc)
        total = len(state.get("documents", []))
        return {"relevant_documents": relevant, "trace": [f"grade -> {len(relevant)}/{total} relevant"]}

    return grade_documents


def make_rewrite_node(llm: BaseLLM) -> Node:
    rewriter = QueryRewriter(llm)

    def rewrite_query(state: RAGState) -> dict:
        new_query = rewriter(state["question"] if not state.get("rewrites") else state.get("search_query", state["question"]))
        return {"search_query": new_query, "rewrites": state.get("rewrites", 0) + 1,
                "trace": [f"rewrite -> '{new_query}'"]}

    return rewrite_query


def make_generate_node(llm: BaseLLM, max_tokens: int = 200) -> Node:
    prompt = PromptLibrary.get("rag_answer")

    def generate(state: RAGState) -> dict:
        docs = state.get("relevant_documents") or []
        if not docs:
            return {"answer": "I don't know based on the provided documents.", "trace": ["generate -> no relevant docs"]}
        answer = llm.generate(prompt.format(context=format_context(docs), question=state["question"]),
                              max_tokens=max_tokens).text
        return {"answer": answer, "trace": [f"generate -> {len(answer.split())} words"]}

    return generate


def make_direct_answer_node(llm: BaseLLM, max_tokens: int = 150) -> Node:
    def direct_answer(state: RAGState) -> dict:
        answer = llm.generate(state["question"], system="Answer briefly in at most 3 sentences.", max_tokens=max_tokens).text
        return {"answer": answer, "trace": ["direct answer (no retrieval)"]}

    return direct_answer


def make_hallucination_node(llm: BaseLLM) -> Node:
    """Self-RAG style check: is the answer grounded in the retrieved documents?"""
    prompt = PromptLibrary.get("grade_hallucination")
    parser = YesNoParser("grounded")

    def check_grounded(state: RAGState) -> dict:
        docs = state.get("relevant_documents") or []
        if not docs:
            return {"grounded": False, "trace": ["grounded -> no (no documents)"]}
        response = llm.generate(prompt.format(context=format_context(docs, max_chars=800), answer=state["answer"]),
                                max_tokens=15, format="json")
        grounded = parser.parse(response.text)
        return {"grounded": grounded, "trace": [f"grounded -> {'yes' if grounded else 'no'}"]}

    return check_grounded
