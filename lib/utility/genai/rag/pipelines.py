import time
from dataclasses import dataclass, field
from typing import Any

from ..abstractions.base_embedder import BaseEmbedder
from ..abstractions.base_llm import BaseLLM
from ..abstractions.base_retriever import BaseRetriever
from ..abstractions.document import SearchResult
from ..memory.conversation_memory import BufferMemory, ConversationMemory
from ..prompting.templates import PromptLibrary, PromptTemplate
from ..retrieval.query_transform import MultiQueryRetriever, QueryRewriter
from ..retrieval.reranking import EmbeddingContextCompressor, LLMReranker
from .citation import format_context


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[SearchResult]
    context: str
    latency_s: float
    steps: dict[str, Any] = field(default_factory=dict)


class NaiveRAG:
    """
    Retrieve -> stuff chunks into a prompt -> generate.

        question ──► retriever ──► top-k chunks ──► prompt ──► LLM ──► answer
    """

    name = "naive_rag"

    def __init__(self, retriever: BaseRetriever, llm: BaseLLM, k: int = 4, prompt: PromptTemplate | None = None,
                 max_tokens: int = 200):
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.prompt = prompt or PromptLibrary.get("rag_answer")
        self.max_tokens = max_tokens

    def generate_answer(self, question: str, results: list[SearchResult]) -> tuple[str, str]:
        context = format_context(results)
        answer = self.llm.generate(self.prompt.format(context=context, question=question), max_tokens=self.max_tokens).text
        return answer, context

    def ask(self, question: str) -> RAGResult:
        start = time.perf_counter()
        results = self.retriever.retrieve(question, self.k)
        answer, context = self.generate_answer(question, results)
        return RAGResult(question, answer, results, context, time.perf_counter() - start,
                         {"retrieved": len(results)})


class AdvancedRAG(NaiveRAG):
    """
    Pre-retrieval + post-retrieval optimisations:

        question ─► rewrite ─► (multi-query) retrieve fetch_k ─► LLM rerank ─► compress ─► generate
    """

    name = "advanced_rag"

    def __init__(self, retriever: BaseRetriever, llm: BaseLLM, k: int = 4, fetch_k: int = 8,
                 rewrite_query: bool = True, multi_query: bool = False, rerank: bool = True,
                 compressor_embedder: BaseEmbedder | None = None, compression_threshold: float = 0.5,
                 max_tokens: int = 200):
        super().__init__(retriever, llm, k=k, max_tokens=max_tokens)
        self.fetch_k = fetch_k
        self.rewriter = QueryRewriter(llm) if rewrite_query else None
        self.search = MultiQueryRetriever(retriever, llm, n=2) if multi_query else retriever
        self.reranker = LLMReranker(llm) if rerank else None
        self.compressor = EmbeddingContextCompressor(compressor_embedder, compression_threshold) if compressor_embedder else None

    def ask(self, question: str) -> RAGResult:
        start = time.perf_counter()
        steps: dict[str, Any] = {}

        query = self.rewriter(question) if self.rewriter else question
        steps["search_query"] = query

        candidates = self.search.retrieve(query, self.fetch_k)
        steps["candidates"] = len(candidates)
        if isinstance(self.search, MultiQueryRetriever):
            steps["queries"] = self.search.last_queries

        if self.reranker:
            results = self.reranker.rerank(question, candidates, top_n=self.k)
            steps["rerank_scores"] = self.reranker.last_scores
        else:
            results = candidates[: self.k]

        if self.compressor:
            before = sum(len(r.text) for r in results)
            results = self.compressor.compress(question, results)
            steps["context_chars_before"] = before
            steps["context_chars_after"] = sum(len(r.text) for r in results)

        answer, context = self.generate_answer(question, results)
        return RAGResult(question, answer, results, context, time.perf_counter() - start, steps)


class ConversationalRAG(NaiveRAG):
    """
    RAG with chat history: follow-up questions ("and what about women?")
    are first condensed into standalone questions before retrieval.
    """

    name = "conversational_rag"

    def __init__(self, retriever: BaseRetriever, llm: BaseLLM, memory: ConversationMemory | None = None,
                 k: int = 4, max_tokens: int = 200):
        super().__init__(retriever, llm, k=k, max_tokens=max_tokens)
        self.memory = memory or BufferMemory()
        self.condense_prompt = PromptLibrary.get("condense_question")

    def condense(self, question: str) -> str:
        history = self.memory.as_text(question)
        if not history:
            return question
        text = self.llm.generate(self.condense_prompt.format(history=history, question=question), max_tokens=80).text
        return text.strip().splitlines()[0] if text.strip() else question

    def ask(self, question: str) -> RAGResult:
        start = time.perf_counter()
        standalone = self.condense(question)
        results = self.retriever.retrieve(standalone, self.k)
        answer, context = self.generate_answer(standalone, results)
        self.memory.add_exchange(question, answer)
        return RAGResult(question, answer, results, context, time.perf_counter() - start,
                         {"standalone_question": standalone})
