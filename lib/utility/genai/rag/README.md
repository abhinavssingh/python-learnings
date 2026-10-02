# Retrieval-Augmented Generation

RAG pipelines connect a retriever to an LLM and return a `RAGResult` containing
the question, answer, retrieved sources, context, latency, and pipeline steps.

- `NaiveRAG`: retrieve, format context, and generate a grounded answer.
- `AdvancedRAG`: optionally rewrite or expand the query, rerank results, and
  compress context before generation.
- `ConversationalRAG`: use conversation memory to resolve follow-up questions.
- `citation.py`: format contexts and extract or tabulate citations.

```python
from lib.utility.genai.factory import build_hr_vectorstore, get_llm
from lib.utility.genai.rag import NaiveRAG
from lib.utility.genai.retrieval import DenseRetriever

store, _ = build_hr_vectorstore()
answer = NaiveRAG(DenseRetriever(store), get_llm()).ask(
    "What are the key elements of Total Rewards?"
)
print(answer.answer)
```

See `Module-5/GENAI/08_rag/` for complete examples.
