# Vector Stores and Nearest-Neighbor Search

A vector store keeps vectors alongside the text and metadata they represent, then finds nearby vectors for a query. This folder compares simple NumPy search with FAISS indexes and demonstrates filtering and persistence.

## Lesson

`01_vectorstore_basics.py` covers NumPy versus FAISS, flat and HNSW search, metadata filters, and saving/loading an index. For instance, encode three policy passages and a question about annual leave; nearest-neighbor search should return passages whose vectors are close to the query. A metadata filter can then restrict results to a selected document section.

## What you should learn

- A flat index is a useful exact-search baseline; approximate indexes such as HNSW trade some search behavior for speed at scale.
- The vector index is not the whole knowledge base: text, metadata, model identity, and persistence all matter.
- Vectors from different embedding models should not be mixed as if they shared the same coordinate space.
- Filtering metadata can improve relevance, but overly restrictive filters can hide the right passage.
- Measure retrieval quality on representative questions instead of judging an index by speed alone.

## Run

From the repository root:

```powershell
python Module-5/GENAI/06_vectorstores/01_vectorstore_basics.py
```

The lesson uses the configured local embedding model. Install dependencies and pull the model by following the [GenAI setup guide](../README.md). The script writes an HTML report under this folder's `reports/` directory and demonstrates index persistence.

## Practice

Index a few short passages with `source` and `topic` metadata. Search with and without a topic filter, compare NumPy and FAISS results, and check that a persisted index can be loaded and queried again. Note the model and source data used to create the vectors.
