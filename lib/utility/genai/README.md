# GenAI Utility Framework

Reusable, local-first utilities for learning and building Generative AI
applications. The framework-agnostic core covers classic NLP, embeddings,
chunking, vector search, retrieval, prompting, RAG, memory, tools, and evaluation.
Optional integrations provide LangChain and LangGraph APIs.

The hands-on learning path is in
[`Module-5/GENAI`](../../../Module-5/GENAI/README.md).

## Setup

From the repository root, install the GenAI dependencies and ensure Ollama is
running with the configured chat and embedding models:

```powershell
.\.venv\Scripts\python.exe -m pip install -r Module-5\GENAI\requirements-genai.txt
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Defaults are `qwen3:8b` for chat, `nomic-embed-text` for embeddings, and
`http://localhost:11434` for Ollama. Override them with `GENAI_CHAT_MODEL`,
`GENAI_EMBEDDING_MODEL`, and `OLLAMA_HOST`.

## Quick start: HR policy RAG

Run this from the repository root. It builds or loads the local HR policy index,
retrieves relevant passages, and asks Ollama to answer from those passages:

```python
import init  # Set up the repository's import paths.

from lib.utility.genai.factory import build_hr_vectorstore, get_llm
from lib.utility.genai.rag import NaiveRAG
from lib.utility.genai.retrieval import DenseRetriever

llm = get_llm()
store, chunks = build_hr_vectorstore()
rag = NaiveRAG(DenseRetriever(store), llm, k=3)

result = rag.ask("What are the key elements of Total Rewards?")
print(result.answer)
print(f"Retrieved {len(result.sources)} sources from {len(chunks)} indexed chunks.")
```

`build_hr_vectorstore()` uses the HR policy PDF under `datasets/GEN AI/` and
persists its index under `saved_models/genai/vectorstores/`. Its default
chunking and storage can be configured or replaced; see `factory.py` and the
`chunking/` and `vectorstores/` packages.

## Package map

| Package | Purpose |
|---|---|
| `abstractions/` | Shared document, LLM, embedder, chunker, retriever, and vector-store interfaces |
| `foundations/` | Text cleaning, tokenization, n-grams, vectorization, similarity, and attention |
| `providers/` | Ollama chat and embedding implementations |
| `embeddings/` | Hashing, TF-IDF, Word2Vec, and cached embedding utilities |
| `loaders/`, `chunking/` | Load PDF, text, and tabular data; split content into chunks |
| `vectorstores/`, `retrieval/` | NumPy and FAISS stores; dense, BM25, hybrid, MMR, query transform, and reranking |
| `prompting/`, `rag/`, `memory/`, `tools/` | Prompt templates and guardrails, RAG pipelines, conversation memory, and tools |
| `frameworks/langchain/` | LangChain model adapters, chains, and tool-calling helpers |
| `frameworks/langgraph/` | Graph state, nodes, checkpointing, and agentic RAG, tool-agent, supervisor, and review graphs |
| `evaluation/`, `observability/`, `visualization/`, `reports/` | Metrics and LLM-as-judge, tracing, plots, and HTML report support |

`factory.py` provides convenience constructors (`get_llm`, `get_embedder`, and
`build_hr_vectorstore`). Individual package `__init__.py` files expose the
supported public imports.

## Configuration and generated files

`GenAIConfig.from_env()` reads the Ollama host and model environment variables.
It also provides defaults for generation, chunking, and retrieval; pass a
`GenAIConfig` instance to factory functions to override them in Python.

Generated embedding caches, vector indexes, and graph checkpoints are stored
under `saved_models/genai/` by default. Prompt templates are in
[`config/prompts/`](config/prompts/).

For complete examples—including n-grams, embedding comparisons, advanced RAG,
LangChain, LangGraph, and evaluation—see the numbered scripts in
[`Module-5/GENAI`](../../../Module-5/GENAI/).
