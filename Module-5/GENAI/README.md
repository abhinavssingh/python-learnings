# Module-5 · Generative AI — learning path

Hands-on scripts that go from classic NLP to agentic RAG, all running **locally** on
[Ollama](https://ollama.com) (`qwen3:8b` for chat, `nomic-embed-text` for embeddings).
Each script has a `main()` and writes an HTML report to its own `reports/` folder.

The reusable code lives in [`lib/utility/genai`](../../lib/utility/genai). Scripts only
orchestrate it, so you can reuse the framework in your own LangChain / LangGraph projects.

## Setup

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
pip install -r Module-5/GENAI/requirements-genai.txt
```

Data: `datasets/GEN AI/the_nestle_hr_policy_pdf_2012.pdf` (the RAG corpus) and
`datasets/GEN AI/eval/hr_policy_qa.json` (12 golden questions with reference answers,
keywords and relevant pages).

Useful environment variables:

| Variable | Effect |
|---|---|
| `GENAI_OPEN_REPORTS=0` | don't open each report in the browser |
| `GENAI_CHAT_MODEL` / `GENAI_EMBEDDING_MODEL` | switch Ollama models |
| `GENAI_EVAL_LIMIT` | number of golden questions in `11_evaluation` (default 4) |
| `GENAI_INTERACTIVE=1` | type approve/edit/reject yourself in the human-in-the-loop demo |

## Learning path

🟢 = no LLM (fast) · 🔵 = embeddings only · 🟣 = uses qwen3:8b (about 5 tokens/s on a laptop, so allow 1–4 minutes)

| # | Folder | Script | Topics |
|---|---|---|---|
| 1 | `01_foundations` | 🟢 `01_text_cleaning` | normalisation, stopwords, stemming |
| | | 🟢 `02_tokenization` | word / char / sentence tokenizers, BPE from scratch |
| | | 🟢 `03_ngrams_language_model` | n-grams, smoothing, perplexity, generation |
| | | 🟢 `04_vectorization_tfidf` | BoW, TF-IDF from scratch vs scikit-learn |
| | | 🟢 `05_similarity_bm25` | cosine / dot / euclidean, Jaccard, BM25 |
| 2 | `02_embeddings` | 🟢 `01_word2vec_scratch` | skip-gram + negative sampling |
| | | 🔵 `02_ollama_embeddings` | sentence embeddings, heatmap, PCA/t-SNE, cache |
| | | 🔵 `03_embedding_comparison` | hashing vs TF-IDF vs neural on the golden set |
| 3 | `03_llm_basics` | 🟣 `01_ollama_chat_basics` | messages, streaming, thinking mode, usage tracing |
| | | 🟣 `02_generation_parameters` | temperature, seed, top-k/p, max tokens |
| | | 🟣 `03_tool_calling_native` | function calling + agent loop |
| 4 | `04_prompt_engineering` | 🟣 `01_prompting_techniques` | zero/few-shot, chain-of-thought, roles, prompt library |
| | | 🟣 `02_structured_output_guardrails` | Pydantic JSON, schema-constrained output, PII, injection |
| 5 | `05_chunking` | 🟢 `01_chunking_strategies` | fixed, recursive, sentence, paragraph, Markdown, code |
| | | 🔵 `02_semantic_chunking` | embedding breakpoints |
| 6 | `06_vectorstores` | 🔵 `01_vectorstore_basics` | NumPy vs FAISS (flat/HNSW), filters, persistence |
| 7 | `07_retrieval` | 🔵 `01_retrieval_strategies` | dense, BM25, hybrid RRF, MMR + metrics |
| | | 🟣 `02_query_transformations_rerank` | rewrite, multi-query, HyDE, step-back, rerank, compression |
| 8 | `08_rag` | 🟣 `01_naive_rag` | retrieve → augment → generate, citations |
| | | 🟣 `02_advanced_rag` | hybrid + rewrite + rerank + compression |
| | | 🟣 `03_conversational_rag_memory` | memory types, condensing follow-ups |
| 9 | `09_langchain` | 🟣 `01_lcel_basics` | Runnables, batch/stream, parallel, structured output |
| | | 🟣 `02_langchain_rag_chain` | LangChain-native RAG + adapter to our retrievers |
| | | 🟣 `03_langchain_tool_calling` | `@tool`, `bind_tools`, manual loop |
| 10 | `10_langgraph` | 🟢 `01_langgraph_basics` | state, reducers, conditional edges, checkpoints |
| | | 🟣 `02_agentic_rag_graph` | adaptive + corrective + self-RAG |
| | | 🟣 `03_tool_agent_memory_graph` | ToolNode, thread memory, SQLite persistence |
| | | 🟣 `04_supervisor_multi_agent` | supervisor → researcher / calculator / writer |
| | | 🟣 `05_human_in_the_loop` | `interrupt()` + `Command(resume=...)` |
| 11 | `11_evaluation` | 🟣 `01_rag_evaluation` | retrieval + generation metrics, LLM-as-judge |
| 12 | `12_transformers` | 🟢 `01_attention_from_scratch` | attention, causal mask, positional encoding, multi-head |

## Running

```bash
# one script
python Module-5/GENAI/05_chunking/01_chunking_strategies.py

# a whole folder (via the repo runner)
python run.py --only "Module-5.GENAI.01_foundations.*"
```

## Building your own projects

```python
from lib.utility.genai.factory import get_llm, get_embedder, build_hr_vectorstore
from lib.utility.genai.loaders import PDFLoader
from lib.utility.genai.chunking import RecursiveCharacterChunker
from lib.utility.genai.vectorstores import IndexManager
from lib.utility.genai.retrieval import DenseRetriever, BM25Retriever, HybridRetriever
from lib.utility.genai.rag import AdvancedRAG
from lib.utility.genai.frameworks.langchain import get_chat_model, as_langchain_retriever, build_rag_chain
from lib.utility.genai.frameworks.langgraph import build_agentic_rag_graph, get_checkpointer
```

Capstone requirement documents stay in [`capstone/`](capstone) and [`projects/`](projects).
