# Foundations of Language Processing

This folder builds the vocabulary and intuition needed for modern language models. The lessons begin with ordinary text and progressively turn it into tokens, probability estimates, and vectors that computers can compare. Start here if tokenization, embeddings, or retrieval still feel like black boxes.

## Why this matters

A language model does not receive words in the same way a person reads them. It receives token IDs, and many downstream methods rely on counts, co-occurrence, or vector representations. Understanding this path makes model behavior easier to debug: an odd answer may start with normalization, token boundaries, or a sparse representation rather than with the model itself.

## Lessons

1. `01_text_cleaning.py` explores normalization, stop words, and stemming. Compare `"Cats are RUNNING!"` before and after lowercasing, punctuation cleanup, stop-word removal, and stemming. Notice that cleaning can remove useful meaning as well as noise.
2. `02_tokenization.py` compares word, character, and sentence tokenization and demonstrates byte-pair encoding (BPE). For `"unhappiness"`, a word tokenizer may keep one unit while a subword tokenizer can represent reusable pieces.
3. `03_ngrams_language_model.py` uses neighboring tokens to estimate likely continuations. Given `"the cat sat"`, a trigram model estimates the next token from the preceding two, then illustrates smoothing, perplexity, and generation.
4. `04_vectorization_tfidf.py` builds bag-of-words and TF-IDF representations and compares a from-scratch version with scikit-learn. In `"the policy covers leave"`, a common word such as `"the"` should carry less distinguishing weight than a rarer term such as `"leave"`.
5. `05_similarity_bm25.py` compares cosine, dot-product, and Euclidean similarity, Jaccard overlap, and BM25. Search for `"annual leave policy"` among short passages and compare how exact term overlap differs from vector geometry.

## Suggested study sequence

Run the lessons in numerical order. After each one, write down what information its representation preserves and what it loses. For example, bag-of-words can notice which words occur but normally loses word order; n-grams preserve a limited amount of local order.

## Run

From the repository root:

```powershell
python Module-5/GENAI/01_foundations/01_text_cleaning.py
python Module-5/GENAI/01_foundations/05_similarity_bm25.py
```

These foundations are designed to run without an Ollama model. Each script prints examples and saves an HTML report under this folder's `reports/` directory. See the [GenAI setup and learning path](../README.md) for the repository-wide overview.

## Practice

Create three short passages about the same topic using different vocabulary. Try to predict which method will rank each passage highest for a chosen query, then run the similarity lesson and explain any difference. Change one word at a time to see how stop-word removal and TF-IDF affect the result.
