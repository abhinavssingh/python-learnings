# Vector Stores

Store embedded documents and retrieve nearest matches. The package provides
`InMemoryVectorStore` for a NumPy-backed local index and `FaissVectorStore` for
FAISS, along with metadata filter support and `IndexManager` for persisting and
reloading indexes.

Vector stores use the interfaces and document types in `abstractions/`. Use
`DenseRetriever` or `MMRRetriever` from `retrieval/` to expose search strategies
to a RAG pipeline.

See `Module-5/GENAI/06_vectorstores/` for the storage and persistence lesson.
Generated indexes from the shared factory are stored under
`saved_models/genai/vectorstores/`.
