from ..abstractions.base_llm import BaseLLM
from ..abstractions.base_retriever import BaseRetriever
from ..abstractions.base_vectorstore import BaseVectorStore
from ..abstractions.document import SearchResult
from ..prompting.output_parsers import ListOutputParser
from ..prompting.templates import PromptLibrary
from .retrievers import reciprocal_rank_fusion


class QueryRewriter:
    """LLM turns a chatty question into a clean search query."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt = PromptLibrary.get("query_rewrite")

    def __call__(self, question: str) -> str:
        text = self.llm.generate(self.prompt.format(question=question), max_tokens=60).text
        return text.strip().strip('"').splitlines()[0] if text.strip() else question


class MultiQueryGenerator:
    """LLM writes N paraphrases of the question (recall booster)."""

    def __init__(self, llm: BaseLLM, n: int = 3):
        self.llm = llm
        self.n = n
        self.prompt = PromptLibrary.get("multi_query")

    def __call__(self, question: str) -> list[str]:
        text = self.llm.generate(self.prompt.format(question=question, n=self.n), max_tokens=150).text
        return ListOutputParser().parse(text)[: self.n]


class HyDEGenerator:
    """Hypothetical Document Embeddings: embed a *fake answer* instead of the question."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt = PromptLibrary.get("hyde")

    def __call__(self, question: str) -> str:
        return self.llm.generate(self.prompt.format(question=question), max_tokens=150).text


class StepBackGenerator:
    """Ask a broader question first to fetch background context."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt = PromptLibrary.get("step_back")

    def __call__(self, question: str) -> str:
        return self.llm.generate(self.prompt.format(question=question), max_tokens=60).text.strip()


class QueryDecomposer:
    """Split a multi-part question into simple sub-questions."""

    def __init__(self, llm: BaseLLM, n: int = 3):
        self.llm = llm
        self.n = n
        self.prompt = PromptLibrary.get("decompose")

    def __call__(self, question: str) -> list[str]:
        text = self.llm.generate(self.prompt.format(question=question, n=self.n), max_tokens=150).text
        return ListOutputParser().parse(text)[: self.n]


class MultiQueryRetriever(BaseRetriever):
    """Retrieve for the original question + LLM paraphrases, then fuse with RRF."""

    name = "multi_query"

    def __init__(self, retriever: BaseRetriever, llm: BaseLLM, n: int = 3, include_original: bool = True):
        self.retriever = retriever
        self.generator = MultiQueryGenerator(llm, n)
        self.include_original = include_original
        self.last_queries: list[str] = []

    def retrieve(self, query: str, k: int = 4) -> list[SearchResult]:
        queries = ([query] if self.include_original else []) + self.generator(query)
        self.last_queries = queries
        return reciprocal_rank_fusion([self.retriever.retrieve(q, k * 2) for q in queries])[:k]


class HyDERetriever(BaseRetriever):
    """Search the vector store with the embedding of a hypothetical answer."""

    name = "hyde"

    def __init__(self, vectorstore: BaseVectorStore, llm: BaseLLM):
        self.vectorstore = vectorstore
        self.generator = HyDEGenerator(llm)
        self.last_hypothesis = ""

    def retrieve(self, query: str, k: int = 4) -> list[SearchResult]:
        self.last_hypothesis = self.generator(query)
        vector = self.vectorstore.embedder.embed_documents([self.last_hypothesis])[0]
        return self.vectorstore.similarity_search_by_vector(vector, k=k)
