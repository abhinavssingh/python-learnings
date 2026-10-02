# Retrieval Strategies

Retrieval is the step that finds candidate evidence for a question. This folder progresses from single-method search to query transformations and reranking, highlighting a central RAG lesson: a fluent answer cannot make up for evidence that was never retrieved.

## Lessons

1. `01_retrieval_strategies.py` compares dense embedding search, BM25 keyword search, hybrid retrieval with Reciprocal Rank Fusion (RRF), and Maximal Marginal Relevance (MMR). For `"Which employees qualify for parental leave?"`, dense search may match paraphrases, BM25 may favor exact policy terms, and hybrid search combines ranked lists. MMR can reduce near-duplicate passages in the final set.
2. `02_query_transformations_rerank.py` explores query rewriting, multi-query retrieval, HyDE, step-back questions, reranking, and context compression. For a vague follow-up such as `"What about contractors?"`, rewriting can make the search query more explicit before retrieval.

## Metrics and judgment

The first lesson reports hit@k, precision@k, recall@k, mean reciprocal rank (MRR), and normalized discounted cumulative gain (nDCG). These measure different aspects of ranking; a single score cannot tell the whole story. Inspect returned passages and use a known question set to understand tradeoffs.

Reranking and query-generation features may call the local chat model, while dense retrieval needs embeddings. Setup is described in the [GenAI setup guide](../README.md).

## Run

From the repository root:

```powershell
python Module-5/GENAI/07_retrieval/01_retrieval_strategies.py
python Module-5/GENAI/07_retrieval/02_query_transformations_rerank.py
```

Reports are written to this folder's `reports/` directory. Allow additional time for model-backed operations.

## Practice

Choose a few questions with known relevant passages. Compare dense, BM25, hybrid, and reranked results at the same `k`. For each strategy, record whether it found relevant evidence, whether results were redundant, and what the ranking metric missed.
