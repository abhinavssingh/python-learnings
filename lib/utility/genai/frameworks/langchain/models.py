from langchain_ollama import ChatOllama, OllamaEmbeddings

from ...config.genai_config import GenAIConfig


def get_chat_model(config: GenAIConfig | None = None, **overrides) -> ChatOllama:
    """LangChain chat model for the local Ollama model (qwen3:8b by default)."""
    config = config or GenAIConfig.from_env()
    params = {
        "model": config.chat_model,
        "base_url": config.ollama_host,
        "temperature": config.temperature,
        "num_predict": config.max_tokens,
        "num_ctx": config.context_window,
        "reasoning": config.think,
    }
    params.update(overrides)
    return ChatOllama(**params)


def get_embeddings(config: GenAIConfig | None = None) -> OllamaEmbeddings:
    config = config or GenAIConfig.from_env()
    return OllamaEmbeddings(model=config.embedding_model, base_url=config.ollama_host)
