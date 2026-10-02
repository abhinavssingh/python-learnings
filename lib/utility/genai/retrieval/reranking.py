import numpy as np

from ..abstractions.base_embedder import BaseEmbedder
from ..abstractions.base_llm import BaseLLM
from ..abstractions.document import Document, SearchResult
from ..foundations.tokenization import SentenceTokenizer
from ..prompting.output_parsers import JsonOutputParser
from ..prompting.templates import PromptLibrary


class LLMReranker:
    """
    Listwise LLM re-ranking: one call scores every candidate 0-10.

    (A cross-encoder model is the production alternative; an LLM keeps
    this project dependency-free and fully local.)
    """

    def __init__(self, llm: BaseLLM, max_chars_per_passage: int = 500):
        self.llm = llm
        self.max_chars = max_chars_per_passage
        self.prompt = PromptLibrary.get("rerank")
        self.last_scores: list[float] = []

    def rerank(self, query: str, results: list[SearchResult], top_n: int = 4) -> list[SearchResult]:
        if not results:
            return []
        passages = "\n\n".join(f"[{i}] {r.text[: self.max_chars]}" for i, r in enumerate(results, start=1))
        response = self.llm.generate(self.prompt.format(question=query, passages=passages),
                                     max_tokens=20 * len(results) + 40, format="json")
        data = JsonOutputParser().safe_parse(response.text, default=[])
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [])
        scores = {int(item.get("id", 0)): float(item.get("score", 0)) for item in data if isinstance(item, dict)}
        self.last_scores = [scores.get(i, 0.0) for i in range(1, len(results) + 1)]
        order = sorted(range(len(results)), key=lambda i: (-self.last_scores[i], i))
        return [SearchResult(results[i].document, self.last_scores[i], results[i].vector, "llm_rerank") for i in order[:top_n]]


class EmbeddingContextCompressor:
    """
    Contextual compression: keep only the sentences of each chunk that are
    semantically close to the question (smaller prompt, less noise).
    """

    def __init__(self, embedder: BaseEmbedder, threshold: float = 0.55, min_sentences: int = 1):
        self.embedder = embedder
        self.threshold = threshold
        self.min_sentences = min_sentences
        self.tokenizer = SentenceTokenizer()

    def compress(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        query_vec = self.embedder.embed_query(query)
        query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-12)
        compressed = []
        for result in results:
            sentences = self.tokenizer.tokenize(result.text)
            if len(sentences) <= self.min_sentences:
                compressed.append(result)
                continue
            vectors = self.embedder.embed_documents(sentences)
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
            sims = vectors @ query_vec
            keep = [i for i, s in enumerate(sims) if s >= self.threshold]
            if len(keep) < self.min_sentences:
                keep = sorted(np.argsort(sims)[::-1][: self.min_sentences].tolist())
            text = " ".join(sentences[i] for i in keep)
            meta = {**result.metadata, "compressed_from_chars": len(result.text)}
            compressed.append(SearchResult(Document(text, meta), result.score, result.vector, result.retriever))
        return compressed
