# Embeddings

This folder moves from count-based text features to dense vectors learned from language. Embeddings are useful when related ideas use different words, but they are not magic: their quality depends on the model, and similarity is a signal rather than proof that two passages mean the same thing.

## Why embeddings matter

In the previous module, we used representations such as:
- bag-of-words
- TF-IDF
- n-grams

These are useful, but they are sparse and rely heavily on exact words. This creates a problem:

Example:
- `"I need time off"`
- `"How do I apply for leave?"`

These sentences have similar meaning but different words. A bag-of-words model may not match them well because the words do not overlap much.

Embeddings solve this by transforming words or sentences into dense vectors in a learned space where related meanings are placed near each other.

Example:

```text
"leave"          -> vector [0.21, -0.67, 0.89, ...]
"time off"       -> vector [0.18, -0.63, 0.91, ...]
"vacation"       -> vector [0.17, -0.58, 0.84, ...]
```

Even though the words differ, their vectors may be close because the embedding model learned that they often appear in similar contexts.

## Lessons

### 1. Word2Vec from scratch (`01_word2vec_scratch.py`)

This lesson implements a word embedding model using skip-gram with negative sampling.

#### What is Word2Vec?
Word2Vec learns word vectors by looking at surrounding words. If two words often appear in similar contexts, their vectors become similar.

Example sentence:

```text
"birds build nests"
```

The model may consider training pairs like:
- center word = `"build"`, context = `"birds"`
- center word = `"build"`, context = `"nests"`
- center word = `"birds"`, context = `"build"`

This teaches the model that `"build"` is related to `"birds"` and `"nests"`.

#### Skip-gram intuition
Skip-gram tries to predict the context words around a target word.

Example:

```text
Input word: "build"
Expected neighbors: "birds", "nests"
```

The model learns vector weights so that:
- `vector("build")` is useful for predicting `"birds"`
- `vector("build")` is also useful for predicting `"nests"`

#### Negative sampling
Negative sampling speeds up learning by comparing the correct neighbors against a few random negative words.

Example:

```text
Positive pair: ("build", "birds")
Negative sample: ("build", "car")
```

The model learns to make positive pairs closer together and negative pairs farther apart.

#### Why this is important
This lesson is a direct demonstration of how embedding models learn meaning from context rather than from manual rules.

In other words:
- words are not given a prepared semantic label
- the model learns them from repeated patterns in text

This is the core idea behind modern embedding-based search and retrieval.

### 2. Local Ollama embeddings (`02_ollama_embeddings.py`)

This lesson creates sentence embeddings with a local Ollama embedding model and then visualizes how those embeddings relate.

#### What is a sentence embedding?
A sentence embedding converts a whole sentence into one dense vector.

Examples:
- `"An employee can request leave"`
- `"How do I apply for time off?"`
- `"The server is down"`

The model should place the first two vectors near each other because they are semantically similar, even though they use different wording.

#### Similarity heatmap
A heatmap shows how similar different sentences are to one another.

Example matrix:

```text
                leave request   time off   server down
leave request        1.00           0.82        0.10
time off             0.82           1.00        0.08
server down          0.10           0.08        1.00
```

This makes it easy to see that:
- `"leave request"` and `"time off"` are related
- `"server down"` is unrelated to both

#### Dimensionality reduction (PCA / t-SNE)
Embedding vectors are high-dimensional, so they are hard to plot directly. PCA and t-SNE reduce them to 2D or 3D for visualization.

This helps explain the geometry of the embedding space:
- semantically similar sentences cluster together
- unrelated sentences separate out

Example:

```text
"How do I apply for leave?"     --> cluster near "An employee can request leave"
"The printer is offline"        --> appears in another part of the plot
```

#### Caching
The lesson also demonstrates caching, which saves time when repeated runs use the same text.

Example:
- First run generates embeddings for the sentences and stores them.
- Second run reuses the cached values instead of hitting the model again.

This is useful, but it also means:
- if the text changes, the cache may be stale
- if the embedding model changes, old cache entries may no longer match the new model

#### Important caution
Similarity is a useful signal, but not proof of equivalent meaning.

Example:
- `"I am taking annual leave"`
- `"I am taking a leave of absence"`

These may be close in a semantic model, but they are not identical in legal or HR context. The model may match them but the task still needs human interpretation.

### 3. Embedding comparison (`03_embedding_comparison.py`)

This lesson compares different text representations on the same HR-policy question set:
- hashing
- TF-IDF
- neural embeddings

#### Why compare methods?
A vector can look intuitive, but the real question is not: "Does it look good?"
The real question is: "Does it retrieve the right evidence for the task?"

#### Hashing
Hashing creates a fixed-size vector by hashing token values.

Example:

```text
"leave policy"
```

The tokens are hashed into a number of buckets, producing a vector representation without learning any word semantics.

This is fast and memory-efficient, but weak for semantic matching.

#### TF-IDF
TF-IDF is more structured than simple bag-of-words because it weights important words based on rarity.

Example:

```text
Question: "How do I request sick leave?"
Relevant policy text: "Employees may request sick leave through the HR portal."
```

TF-IDF can retrieve the relevant policy text because it sees overlap on relevant terms such as `"request"`, `"sick"`, and `"leave"`.

#### Neural embeddings
Neural embeddings capture meaning, not just exact word overlap.

Example:

```text
Question: "How do I take time off?"
Relevant policy text: "Annual leave may be requested through the employee portal."
```

A pure word-matching model may miss this because `"time off"` and `"annual leave"` differ lexically. An embedding model may still retrieve the right passage because the concepts are related.

#### Golden question set
This lesson uses a small set of representative HR questions to compare retrieval quality.

Example golden questions:
- `"How do I apply for maternity leave?"`
- `"What is the process for annual leave?"`
- `"Can I work remotely during sick leave?"`

The goal is to test whether the retrieval method finds the correct policy evidence for each question.

This is more useful than asking "which representation is mathematically elegant" because real-world systems need useful results, not just pretty vectors.

## What to pay attention to

- Similarity depends on both the representation and the metric; inspect the actual neighbors rather than trusting a single number.
- PCA and t-SNE are visualization tools. A 2D scatter plot is useful for intuition, but it is not a perfect representation of the original high-dimensional space.
- Cached vectors speed up repeat runs, but stale caches can produce misleading results after text or model changes.
- The first lesson is a learning exercise from scratch. Lessons 2 and 3 depend on Ollama and the configured embedding model.
- Embeddings are strong retrieval tools, but they are not magic; they can still make wrong matches when meanings are context-sensitive or domain-specific.

## Run

From the repository root:

```powershell
python Module-5/GENAI/02_embeddings/01_word2vec_scratch.py
python Module-5/GENAI/02_embeddings/02_ollama_embeddings.py
```

Install the dependencies and pull the embedding model as described in the [GenAI setup guide](../README.md). Reports are written to `reports/` inside this folder.

## Practice

Create a small set of paraphrases and unrelated sentences. Compare the nearest neighbors returned by TF-IDF and the neural embedder.

### Example practice set

1. `"An employee can request leave"`
2. `"How do I apply for time off?"`
3. `"The office printer is broken"`
4. `"Employees may take annual vacation days"`
5. `"The system requires a password reset"`

Now compare results:
- Sentence 1 and 2 should be closest semantically.
- Sentence 4 may also be close to 1 and 2 because it is related to leave/vacation.
- Sentence 3 and 5 should be far away because they are unrelated.

Ask yourself:
- Did the model match by shared meaning or by vocabulary overlap?
- Did it retrieve a relevant result or a misleading association?
- Was the similarity score useful in context, or did it need domain knowledge to interpret it?

## Big picture

Embeddings move us from exact lexical matching to representation learning.

The core progression is:

```text
raw text -> tokens -> dense vectors -> semantic similarity -> retrieval and reasoning
```

This is one of the most important ideas in modern NLP and GenAI:
- if we can place related concepts near each other in vector space,
- then we can retrieve relevant information, search semantically, and build more intelligent systems.

The important caveat is that embeddings are only a representation. They do not replace careful evaluation, domain knowledge, or task-specific validation.
