from typing import Callable

from ..abstractions.base_embedder import BaseEmbedder
from ..abstractions.base_llm import BaseLLM


class ProviderRegistry:
    """
    Name -> factory registry, so scripts pick providers by config:

        llm = ProviderRegistry.create_llm("ollama")
        embedder = ProviderRegistry.create_embedder("hashing", dim=256)
    """

    _llms: dict[str, Callable[..., BaseLLM]] = {}
    _embedders: dict[str, Callable[..., BaseEmbedder]] = {}

    @classmethod
    def register_llm(cls, name: str, factory: Callable[..., BaseLLM]) -> None:
        cls._llms[name] = factory

    @classmethod
    def register_embedder(cls, name: str, factory: Callable[..., BaseEmbedder]) -> None:
        cls._embedders[name] = factory

    @classmethod
    def create_llm(cls, name: str = "ollama", **kwargs) -> BaseLLM:
        cls._register_defaults()
        if name not in cls._llms:
            raise KeyError(f"Unknown LLM provider '{name}'. Available: {sorted(cls._llms)}")
        return cls._llms[name](**kwargs)

    @classmethod
    def create_embedder(cls, name: str = "ollama", **kwargs) -> BaseEmbedder:
        cls._register_defaults()
        if name not in cls._embedders:
            raise KeyError(f"Unknown embedder '{name}'. Available: {sorted(cls._embedders)}")
        return cls._embedders[name](**kwargs)

    @classmethod
    def available(cls) -> dict[str, list[str]]:
        cls._register_defaults()
        return {"llms": sorted(cls._llms), "embedders": sorted(cls._embedders)}

    @classmethod
    def _register_defaults(cls) -> None:
        if cls._llms and cls._embedders:
            return
        from ..embeddings.hashing_embedder import HashingEmbedder
        from ..embeddings.tfidf_embedder import TfidfEmbedder
        from .ollama.ollama_embedder import OllamaEmbedder
        from .ollama.ollama_llm import OllamaLLM

        cls._llms.setdefault("ollama", OllamaLLM)
        cls._embedders.setdefault("ollama", OllamaEmbedder)
        cls._embedders.setdefault("hashing", HashingEmbedder)
        cls._embedders.setdefault("tfidf", TfidfEmbedder)
