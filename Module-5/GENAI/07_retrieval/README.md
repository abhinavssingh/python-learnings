# Retrieval Strategies

Retrieval is the step that finds candidate evidence for a question. This folder progresses from single-method search to query transformations and reranking, highlighting a central RAG lesson: a fluent answer cannot make up for evidence that was never retrieved.

## Why good retrieval matters

A model can generate a convincing answer, but if it never sees the right evidence, it will still be wrong.

Example:

```text
Question: "Which employees qualify for parental leave?"
```

If the policy says:

```text
"Employees who have completed 12 months of continuous service are eligible for paid parental leave."
```

then retrieval should return that passage, not a nearby but irrelevant policy on travel reimbursement.

This is the difference between a lucky answer and a grounded answer.

## Retrieval methods

### 1. Dense retrieval
Dense retrieval uses embeddings to find semantically similar passages.

Example:

```text
Question: "How do I apply for time off?"
Passage: "Employees can request annual leave through the employee portal."
```

The words differ, but the meaning is similar. Embeddings can often connect them.

This works well for paraphrases and natural-language questions.

### 2. BM25 retrieval
BM25 is a classic keyword-based retrieval method.

It rewards exact term matches and penalizes overly common words.

Example:

```text
Query: "annual leave policy"
Passage: "The annual leave policy explains the rules for paid vacation time."
```

BM25 usually ranks this highly because it contains many of the exact search terms.

This is strong when queries are keyword-heavy or when exact wording matters.

### 3. Hybrid retrieval
Hybrid retrieval combines dense and keyword search.

Example:

```text
Query: "Who qualifies for maternity leave?"
```

Dense retrieval may find semantically similar passages, while BM25 may find exact phrase matches from policy text. Combining them often improves recall and precision.

### 4. MMR (Maximal Marginal Relevance)
MMR reduces redundancy in the top results.

Example:

```text
Result 1: "Employees with 12 months of service are eligible for parental leave."
Result 2: "Employees with 12 months of service are eligible for parental leave."
```

This duplicate is not useful. MMR tries to keep the top results diverse so you do not retrieve five near-identical passages.

## Lessons

### 1. Retrieval strategies (`01_retrieval_strategies.py`)

This lesson compares:
- dense embedding search
- BM25 keyword search
- hybrid retrieval with Reciprocal Rank Fusion (RRF)
- Maximal Marginal Relevance (MMR)

#### Example query

```text
"Which employees qualify for parental leave?"
```

Possible results:
- dense search may match paraphrases like "employees eligible for maternity leave"
- BM25 may favor exact matches like "parental leave eligibility"
- hybrid search may combine both result lists
- MMR may remove duplicates and keep diverse passages

This makes the trade-offs clear: one method dominates by semantics, another by exact matches, and hybrid search often balances them.

### 2. Query transformations and reranking (`02_query_transformations_rerank.py`)

This lesson explores:
- query rewriting
- multi-query retrieval
- HyDE (Hypothetical Document Embeddings)
- step-back questions
- reranking
- context compression

#### Query rewriting example

```text
Follow-up: "What about contractors?"
```

This vague question may be incomplete without context. A rewriting step may convert it into:

```text
"Are contractors eligible for the same leave policies as full-time employees?"
```

This much clearer query will produce a better candidate set.

#### Multi-query retrieval
Instead of one query, the system generates multiple related queries:

```text
"annual leave policy"
"vacation leave policy"
"employee time off rules"
```

Then it retrieves evidence for all of them and combines the results.

#### Reranking
After initial retrieval, a reranker reorders candidates based on relevance to the actual question.

This can fix cases where a naive retrieval stack returns several somewhat related results before the most relevant one.

## Metrics and judgment

The first lesson reports:
- hit@k
- precision@k
- recall@k
- mean reciprocal rank (MRR)
- normalized discounted cumulative gain (nDCG)

These measure different aspects of ranking; a single score cannot tell the whole story.

Example:

```text
Questions: 10
For each question, top 5 results are inspected.
```

A system may have high recall but poor ranking if the relevant result is buried down the list.

This is why you should inspect returned passages and use a known question set to understand tradeoffs.

## Run

From the repository root:

```powershell
python Module-5/GENAI/07_retrieval/01_retrieval_strategies.py
python Module-5/GENAI/07_retrieval/02_query_transformations_rerank.py
```

Reports are written to this folder's `reports/` directory. Allow additional time for model-backed operations.

## Practice

Choose a few questions with known relevant passages. Compare dense, BM25, hybrid, and reranked results at the same `k`. For each strategy, record whether it found relevant evidence, whether results were redundant, and what the ranking metric missed.

### Example exercise

Question:

```text
"When can employees request parental leave?"
```

Candidate passages:
- "Employees may request parental leave after 12 months of service."
- "Travel reimbursement is processed monthly."
- "Employees may request parental leave after 12 months of service."

If the system returns the second or third before the first, it is likely ranking poorly.

Ask:
- Did dense retrieval find the paraphrase?
- Did BM25 favor the exact wording?
- Did hybrid retrieval fix the ranking?
- Did MMR reduce duplicates?

## Big picture

Retrieval is not only a search problem; it is a relevance problem.

The flow is:

```text
question -> retrieval strategy -> topk candidates -> reranking -> evidence for generation
```

A good answer is only as good as the evidence that reaches the model. Retrieval is where many real-world RAG failures begin.
