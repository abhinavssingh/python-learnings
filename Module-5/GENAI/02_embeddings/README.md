# Embeddings

This folder moves from count-based text features to dense vectors learned from language. Embeddings are useful when related ideas use different words, but they are not magic: their quality depends on the model, and similarity is a signal rather than proof that two passages mean the same thing.

## Lessons

1. `01_word2vec_scratch.py` implements skip-gram with negative sampling. The learning goal is to see how nearby words become training examples and how the model learns vectors that reflect co-occurrence. Consider the context pairs from `"birds build nests"`: the center word `"build"` is trained to predict nearby words such as `"birds"` and `"nests"`.
2. `02_ollama_embeddings.py` creates sentence embeddings with a local Ollama embedding model and visualizes relationships with a similarity heatmap and dimensionality reduction (PCA/t-SNE). It also demonstrates caching. Compare the sentences `"An employee can request leave"` and `"How do I apply for time off?"` with an unrelated sentence.
3. `03_embedding_comparison.py` compares hashing, TF-IDF, and neural embeddings against the HR-policy golden question set. Use this to ask not merely which vector looks intuitive, but which method retrieves useful evidence for the task.

## What to pay attention to

- Similarity depends on the representation and metric; inspect the actual neighbors instead of trusting a single score.
- PCA and t-SNE compress high-dimensional vectors for plotting. A 2D plot is an illustration, not a faithful map of every distance in the original space.
- Cached vectors save time, but changing the source text or embedding model can make an old cache inappropriate.
- The first lesson is a from-scratch learning exercise. Lessons 2 and 3 require Ollama and the configured embedding model.

## Run

From the repository root:

```powershell
python Module-5/GENAI/02_embeddings/01_word2vec_scratch.py
python Module-5/GENAI/02_embeddings/02_ollama_embeddings.py
```

Install the dependencies and pull the embedding model as described in the [GenAI setup guide](../README.md). Reports are written to `reports/` inside this folder.

## Practice

Make a small set of paraphrases and unrelated sentences. Compare the nearest neighbors returned by TF-IDF and the neural embedder. For every surprising match, inspect the text and ask whether the model found shared meaning, shared vocabulary, or a misleading association.
