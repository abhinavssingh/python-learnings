from .abstractions.base_embedder import BaseEmbedder
from .config.genai_config import GenAIConfig
from .datasets import hr_policy_chunks
from .embeddings.cached_embedder import CachedEmbedder
from .providers.ollama.ollama_embedder import OllamaEmbedder
from .providers.ollama.ollama_llm import OllamaLLM
from .vectorstores.in_memory_store import InMemoryVectorStore
from .vectorstores.index_manager import IndexManager


def get_llm(config: GenAIConfig | None = None, tracer=None, check: bool = True, **options) -> OllamaLLM:
    """Local Ollama chat model; fails fast with a helpful message if Ollama is not running."""
    llm = OllamaLLM(config, tracer=tracer, **options)
    if check:
        llm.ensure_available()
    return llm


def get_embedder(config: GenAIConfig | None = None, cached: bool = True) -> BaseEmbedder:
    config = config or GenAIConfig.from_env()
    embedder = OllamaEmbedder(config)
    return CachedEmbedder(embedder, config.embedding_cache_dir) if cached else embedder


def build_hr_vectorstore(config: GenAIConfig | None = None, embedder: BaseEmbedder | None = None,
                         store_cls=InMemoryVectorStore, force_rebuild: bool = False):
    """Chunk + embed + persist the HR policy once; later runs reload from saved_models/genai."""
    config = config or GenAIConfig.from_env()
    embedder = embedder or get_embedder(config)
    chunks = hr_policy_chunks(config.chunk_size, config.chunk_overlap)
    name = f"hr_policy_{store_cls.__name__.lower()}_{config.chunk_size}_{config.chunk_overlap}"
    store = IndexManager(config.vectorstore_dir, store_cls).build_or_load(name, chunks, embedder, force_rebuild)
    return store, chunks
