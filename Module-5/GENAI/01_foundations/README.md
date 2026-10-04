# Foundations of Language Processing

This folder builds the vocabulary and intuition needed for modern language models. The lessons begin with ordinary text and progressively turn it into tokens, probability estimates, and vectors that computers can compare. Start here if tokenization, embeddings, or retrieval still feel like black boxes.

## Why this matters

A language model does not receive words in the same way a person reads them. It receives token IDs, and many downstream methods rely on counts, co-occurrence, or vector representations. Understanding this path makes model behavior easier to debug: an odd answer may start with normalization, token boundaries, or a sparse representation rather than with the model itself.

In practice, every modern NLP and GenAI workflow goes through a similar pipeline:

1. Raw text enters the system.
2. It is normalized and cleaned.
3. It is split into tokens.
4. Words are turned into numbers or vectors.
5. Models estimate meaning or relevance from those representations.

Without understanding this pipeline, it is easy to misread why a model performs well or poorly on a particular task.

## Lesson-by-lesson detail

### 1. Text cleaning (`01_text_cleaning.py`)

This lesson introduces the idea that text needs to be standardized before we can compare or compute over it.

Key concepts:
- Lowercasing: `"Cats"` and `"cats"` should be treated as the same word.
- Punctuation removal: `"RUNNING!"` should become `"RUNNING"` for many tasks.
- Stop-word removal: common words such as `"the"`, `"is"`, and `"a"` are often removed because they add little meaning.
- Stemming: reducing a word to its root form such as `"running"` to `"run"`.

Example:

```text
Original: "Cats are RUNNING! They are happy."
Lowercase: "cats are running! they are happy."
Remove punctuation: "cats are running they are happy"
Remove stop words: "cats running happy"
Stem words: "cat run happy"
```

Why this matters:
- Cleaning reduces noise and makes text more comparable.
- It also changes meaning. For example, removing stop words may remove important context in tasks like sentiment analysis or legal document review.
- The lesson intentionally shows that cleaning is not always free: it can improve consistency but can also erase useful distinctions.

### 2. Tokenization (`02_tokenization.py`)

Tokenization decides what counts as a unit of text. A model cannot work directly with raw text; it needs discrete units.

Types of tokenization:
- Word tokenization: split by spaces and punctuation
- Character tokenization: split by individual letters or characters
- Sentence tokenization: split by sentences
- Subword tokenization: break rare or complex words into reusable parts

Example:

```text
Sentence: "The unhappy employee is unhappy."
Word tokens: ["The", "unhappy", "employee", "is", "unhappy"]
Character tokens: ["T", "h", "e", " ", "u", "n", "h", "a", "p", "p", "y", ...]
Sentence tokens: ["The unhappy employee is unhappy."]
```

Subword example:

```text
Word: "unhappiness"
Word-level tokenization: ["unhappiness"]
Subword tokenization (BPE-like): ["un", "happi", "ness"]
```

Why this matters:
- Some words are rare, and a model may not know them as whole tokens.
- Subword units let the model reuse familiar pieces.
- This is one of the reasons modern LLMs can handle unseen or uncommon words better than classic word-based systems.

### 3. N-grams and language modeling (`03_ngrams_language_model.py`)

An n-gram model predicts the next token based on the previous `n - 1` tokens.

Example:

```text
Sentence: "the cat sat on the mat"
2-grams: ["the cat", "cat sat", "sat on", "on the", "the mat"]
3-grams: ["the cat sat", "cat sat on", "sat on the", "on the mat"]
```

Given the sequence:

```text
"the cat sat"
```

A trigram model looks at the previous two tokens (`"the cat"` and then `"sat"` in a sliding context). It estimates what token is likely next, such as `"on"`.

Important concepts:
- Probability model: the model estimates the likelihood of each next token.
- Smoothing: if a phrase has never appeared before, the model still assigns some probability instead of zero.
- Perplexity: a measure of how surprised the model is by the next token. Lower perplexity usually means better predictions.
- Generation: the model repeatedly predicts the next token to continue a sequence.

Example:

```text
Prompt: "the cat sat"
Possible next tokens: "on" (high probability), "near" (medium), "down" (low)
```

This is the same intuition behind modern generative models, just with much larger context windows and more powerful parameterizations.

### 4. Vectorization and TF-IDF (`04_vectorization_tfidf.py`)

Text must be converted into numbers before many machine learning methods can compare it. This is the core of vector space models.

#### Bag-of-words
This counts words but ignores their order.

Example:

```text
Doc 1: "the policy covers leave"
Doc 2: "leave policy details"
```

Vector representation (simplified):

```text
Words: [the, policy, covers, leave, details]
Doc 1: [1, 1, 1, 1, 0]
Doc 2: [0, 1, 0, 1, 1]
```

This captures word presence or frequency, but it loses word order and sentence structure.

#### TF-IDF
TF-IDF stands for:
- TF = Term frequency: how often a word appears in a document
- IDF = Inverse document frequency: how rare a word is across all documents

Example:

```text
Query: "annual leave policy"
```

A word like `"annual"` may be important in a specific type of document, while `"the"` is likely unimportant because it appears everywhere.

So the representation becomes:
- words that appear often in one document but less often overall receive higher weights
- common fillers like `"the"` receive lower weight

This is the basis for many search and retrieval systems.

### 5. Similarity and BM25 (`05_similarity_bm25.py`)

Once text is represented as vectors, we can compare documents and rank them by relevance.

#### Cosine similarity
Measures how similar two vectors are by looking at their angle rather than their raw length.

Example:

```text
Doc A: [1, 0, 1]
Doc B: [1, 1, 0]
```

They share some information but are not identical. Cosine similarity tells us how aligned they are in vector space.

#### Dot product
Multiplies corresponding elements of two vectors.

```text
Query: [1, 0, 1]
Doc:   [2, 0, 3]
Dot product = 1*2 + 0*0 + 1*3 = 5
```

This is useful but can be affected by vector magnitude.

#### Euclidean distance
Measures the straight-line distance between points in vector space.

```text
Point A = (1, 1)
Point B = (4, 5)
Distance = sqrt((4 - 1)^2 + (5 - 1)^2) = 5
```

#### Jaccard overlap
Measures set overlap.

```text
A = {policy, leave, salary}
B = {leave, policy, vacation}
intersection = {policy, leave}
union = {policy, leave, salary, vacation}
Jaccard = 2 / 4 = 0.5
```

This works well when you care about shared terms, but it ignores term frequency and relative importance.

#### BM25
BM25 is a classic retrieval score used in search systems. It adjusts for:
- how often query terms appear in a document
- document length
- how rare a term is across the corpus

Example:

```text
Query: "annual leave policy"
Documents:
1. "annual leave policy details"
2. "policy and procedures"
3. "leave is allowed for employees"
```

BM25 usually ranks document 1 highest because it contains the most relevant terms and matches the user query closely.

This is a key idea in retrieval-augmented generation (RAG): we search for the best relevant passages, then pass those passages to a model.

## Suggested study sequence

Run the lessons in numerical order. After each one, write down:
- what information the representation preserves
- what information it loses

Examples:
- Bag-of-words preserves term counts but loses word order.
- N-grams preserve local context but only within a small window.
- TF-IDF preserves distinguishing terms but discards much of the original structure.
- Similarity metrics compare representations but do not explain the full semantics of a passage.

## Run

From the repository root:

```powershell
python Module-5/GENAI/01_foundations/01_text_cleaning.py
python Module-5/GENAI/01_foundations/05_similarity_bm25.py
```

These foundations are designed to run without an Ollama model. Each script prints examples and saves an HTML report under this folder's `reports/` directory. See the [GenAI setup and learning path](../README.md) for the repository-wide overview.

## Practice

Create three short passages about the same topic using different vocabulary. Try to predict which method will rank each passage highest for a chosen query, then run the similarity lesson and explain any difference. Change one word at a time to see how stop-word removal and TF-IDF affect the result.

### Example practice prompt

Topic: employee leave policy

Passage A:
- "The annual leave policy explains paid time off for employees."

Passage B:
- "Employees may take vacation days according to the holiday rules."

Passage C:
- "This document explains company policy for time away from work."

Query:
- "annual leave policy"

Prediction:
- Passage A should likely rank highest because it contains the exact terms `annual`, `leave`, and `policy`.
- TF-IDF may give more weight to less common terms such as `annual` and `leave` than to common filler words.

This exercise helps you see how representation choices change retrieval outcomes.

## Big picture

These lessons show the path from raw text to searchable and learnable representations:

```text
raw text -> cleaning -> tokenization -> vectors -> similarity/ranking -> retrieval/LLM context
```

That is the foundation behind many modern AI systems, including semantic search, recommendation, classification, and retrieval-augmented generation.
