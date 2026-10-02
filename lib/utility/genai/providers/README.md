# Providers

Concrete implementations of the framework's model and embedding abstractions.
The current provider is Ollama, located in `ollama/`; `ProviderRegistry` supports
provider lookup and registration.

For common local setup, prefer the convenience functions in the parent
package's `factory.py`:

```python
from lib.utility.genai.factory import get_embedder, get_llm

llm = get_llm()
embedder = get_embedder()
```

These use `GenAIConfig` defaults and environment overrides. `get_llm()` checks
that Ollama is available by default; start the Ollama service and pull the
configured models before making model calls.
