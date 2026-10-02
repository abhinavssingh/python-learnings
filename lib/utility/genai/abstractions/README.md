# Abstractions

Shared contracts and data types used across the GenAI framework. Implementations
in providers, chunking, vector stores, and retrieval build on these interfaces.

## Main types

- `Document` and `SearchResult`: source content, metadata, and retrieval results.
- `BaseLLM`, `Message`, and `LLMResponse`: model inputs and generated responses.
- `BaseEmbedder`, `BaseChunker`, `BaseVectorStore`, and `BaseRetriever`: extension
  points for custom implementations.
- `MetadataFilter`: type used when applying metadata constraints.

Import the public types from `lib.utility.genai.abstractions`. See `providers/`
for LLM and embedder implementations and `Module-5/GENAI` for working examples.
