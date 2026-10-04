# Vector Stores and Nearest-Neighbor Search

A vector store keeps vectors alongside the text and metadata they represent, then finds nearby vectors for a query. This folder compares simple NumPy search with FAISS indexes and demonstrates filtering and persistence.

## Why vector stores matter

Once text has been converted into embeddings, we need a way to search through many vectors efficiently.

Example:

```text
Query: "How do I apply for annual leave?"
```

Instead of scanning every document manually, a vector store can search for the embedding closest to the query embedding.

If we index passages such as:

- "Employees may submit annual leave requests online."
- "The company provides paid sick leave after 30 days."
- "Travel expenses are reimbursable with prior approval."

then the query vector should be closest to the annual leave passage, not to the travel-expense passage.

This is the core idea behind semantic search.

## What is a vector store?

A vector store typically holds:
- the original text chunk
- its embedding vector
- metadata such as source, page, document ID, author, topic, date, or section name

Example record:

```json
{
  "text": "Employees may request annual leave after 90 days of service.",
  "embedding": [0.12, -0.54, 0.92, ...],
  "metadata": {
    "source": "hr_policy.pdf",
    "section": "leave",
    "page": 12
  }
}
```

When a new query embedding arrives, the vector store finds the nearest neighbors by vector distance or similarity.

## Lessons

### 1. Vector store basics (`01_vectorstore_basics.py`)

This lesson covers:
- NumPy-based nearest-neighbor search
- FAISS indexes
- flat search vs approximate search
- metadata filters
- saving and loading indexes

#### NumPy baseline
A simple baseline is to compute pairwise similarity between the query vector and every stored vector.

Example:

```python
similarities = query_vector @ all_vectors.T
```

This is easy to understand and useful for small datasets, but it does not scale well to millions of embeddings.

#### FAISS
FAISS is designed for efficient similarity search over large vector collections.

There are two broad categories:
- flat index: exact search, straightforward and accurate
- HNSW or approximate index: faster search at scale, but not always exact

Example:

```text
Flat index: best for correctness and small datasets
HNSW: faster for large databases with approximate nearest neighbors
```

#### Filtering by metadata
A metadata filter restricts search to a subset of records.

Example:

```text
Search for annual leave, but only within the "leave" section.
```

This reduces irrelevant matches.

Example:

```json
{
  "topic": "leave"
}
```

This is very useful in enterprise documents, where multiple departments or policies may share related vocabulary.

#### Persistence
The index can be saved to disk and loaded later.

Example:

```text
index.faiss
metadata.json
```

This is important because re-embedding an entire corpus every time is expensive. Persistent indexes are standard in real production systems.

## What you should learn

- A flat index is a useful exact-search baseline; approximate indexes such as HNSW trade some search behavior for speed at scale.
- The vector index is not the whole knowledge base: text, metadata, model identity, and persistence all matter.
- Vectors from different embedding models should not be mixed as if they shared the same coordinate space.
- Filtering metadata can improve relevance, but overly restrictive filters can hide the right passage.
- Measure retrieval quality on representative questions instead of judging an index by speed alone.

## Example scenario

Suppose we have three HR policy passages:

```text
1. "Employees may request annual leave after 90 days of service."
2. "Employees may request sick leave up to 10 days per year."
3. "Travel reimbursement is available for approved business trips."
```

Query:

```text
"How long before I can take annual leave?"
```

The nearest-neighbor search should prioritize passage 1.

If metadata filtering is set to `topic = leave`, then the search must ignore passage 3 and likely prefer passage 1 over passage 2.

## Run

From the repository root:

```powershell
python Module-5/GENAI/06_vectorstores/01_vectorstore_basics.py
```

The lesson uses the configured local embedding model. Install dependencies and pull the model by following the [GenAI setup guide](../README.md). The script writes an HTML report under this folder's `reports/` directory and demonstrates index persistence.

## Practice

Index a few short passages with `source` and `topic` metadata. Search with and without a topic filter, compare NumPy and FAISS results, and check that a persisted index can be loaded and queried again. Note the model and source data used to create the vectors.

### Example exercise

Create passages:

```text
A: "Employees can request annual leave after 90 days of service."
B: "Sick leave can be used for illness-related absences."
C: "Travel expenses require prior manager approval."
```

Assign metadata:

```json
{"source": "policy-v1", "topic": "leave"}
{"source": "policy-v1", "topic": "health"}
{"source": "policy-v1", "topic": "travel"}
```

Then query:

```text
"When can I request annual leave?"
```

Expect the top result to be A, and the metadata filter to keep results within the `leave` topic if requested.

## Big picture

Vector stores are a retrieval layer, not the entire knowledge system.

```text
embedding model -> vector store -> nearest-neighbor lookup -> retrieved evidence -> answer generation
```

Without a good vector store, retrieval can become slow, inaccurate, or poorly scoped. With a good one, the model gets the right evidence at the right time.
