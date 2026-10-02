# Embeddings

Embedding algorithms and utilities that operate independently of a specific
vector database.

- `HashingEmbedder` and `TfidfEmbedder` provide local, non-neural baselines.
- `Word2VecScratch` is a learning implementation of Word2Vec.
- `CachedEmbedder` persists calls to an underlying embedder.
- `analysis` contains similarity-matrix, nearest-neighbour, vector-arithmetic,
  and 2D-projection helpers.

The shared embedder contract lives in `abstractions/`; the Ollama-backed
production embedder is in `providers/ollama/`.
See `Module-5/GENAI/02_embeddings/` for runnable examples.
