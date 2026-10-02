# Retrieval

Find and prepare relevant documents for a user query. Basic retrievers include
dense vector search, BM25 lexical search, hybrid reciprocal-rank fusion, and
maximal marginal relevance (MMR). Query transformations include rewriting,
multi-query, HyDE, decomposition, and step-back; post-retrieval utilities
include reranking and context compression.

```python
from lib.utility.genai.retrieval import DenseRetriever, HybridRetriever
```

Retrievers consume document/vector-store implementations and return
`SearchResult` values. Combine them with `NaiveRAG` or `AdvancedRAG` from
`rag/`. See `Module-5/GENAI/07_retrieval/`.
