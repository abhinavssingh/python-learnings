# Configuration

`GenAIConfig` centralizes model, generation, chunking, retrieval, dataset, and
artifact settings. Defaults target a local Ollama server using `qwen3:8b` for
chat and `nomic-embed-text` for embeddings.

```python
from lib.utility.genai.config import GenAIConfig

config = GenAIConfig.from_env()
```

`from_env()` reads `OLLAMA_HOST`, `GENAI_CHAT_MODEL`, and
`GENAI_EMBEDDING_MODEL`; keyword overrides can be passed to `from_env()` or the
dataclass constructor. Prompt templates are stored in `prompts/`.

Convenience factory functions accept a config instance; see `factory.py` in the
parent package.
