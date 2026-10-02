"""
Sentence embeddings with a local Ollama model (nomic-embed-text).

Learn:
  * embedding sentences into dense vectors (dimension, normalisation)
  * task prefixes (search_document / search_query) used by nomic-embed-text
  * semantic similarity heatmap — paraphrases score high even with different words
  * nearest-neighbour search, PCA / t-SNE projections, embedding cache speed-up

Requires:  ollama pull nomic-embed-text
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

import numpy as np  # noqa: E402

from lib.utility.genai.config import GenAIConfig  # noqa: E402
from lib.utility.genai.datasets import SAMPLE_SENTENCES  # noqa: E402
from lib.utility.genai.embeddings import CachedEmbedder, nearest_neighbours, project_2d, similarity_matrix  # noqa: E402
from lib.utility.genai.providers.ollama import OllamaEmbedder  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.visualization import embedding_scatter, similarity_heatmap  # noqa: E402

QUERIES = ["How is pay increased?", "Are there courses to learn new skills?", "What behaviour is not allowed at work?"]


def main():
    config = GenAIConfig.from_env()
    embedder = OllamaEmbedder(config)

    texts = [s for s, _ in SAMPLE_SENTENCES]
    groups = [lab for _, lab in SAMPLE_SENTENCES]
    vectors = embedder.embed_documents(texts)

    sim = similarity_matrix(vectors, [f"{g}: {t[:28]}" for t, g in SAMPLE_SENTENCES])
    values = sim.values
    same = np.array([[a == b for b in groups] for a in groups])
    np.fill_diagonal(same, False)
    off_diag = ~np.eye(len(groups), dtype=bool)
    intra, inter = values[same].mean(), values[off_diag & ~same].mean()

    search_rows = []
    for q in QUERIES:
        for rank, (label, score) in enumerate(nearest_neighbours(embedder.embed_query(q), vectors, texts, k=3), start=1):
            search_rows.append({"query": q, "rank": rank, "score": round(score, 3), "sentence": label})

    cache = CachedEmbedder(embedder, config.embedding_cache_dir)
    t0 = time.perf_counter()
    cache.embed_documents(texts)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    cache.embed_documents(texts)
    second = time.perf_counter() - t0

    report = GenAIReport(f"Ollama Embeddings ({config.embedding_model})")
    report.grid([
        report.kv("Embedding facts", {"model": config.embedding_model, "dimension": vectors.shape[1],
                                      "vector norm (normalised)": round(float(np.linalg.norm(vectors[0])), 4),
                                      "mean similarity same topic": round(float(intra), 3),
                                      "mean similarity different topic": round(float(inter), 3)}),
        report.kv("Embedding cache", {"first call (s)": round(first, 3), "second call (s)": round(second, 4),
                                      **cache.stats(), "cache dir": str(config.embedding_cache_dir)}),
    ])
    report.full_plot(similarity_heatmap(sim, title="Semantic cosine similarity"),
                     "Compare with the TF-IDF heatmap in 01_foundations: paraphrases now match")
    report.full_table("Semantic search (query -> nearest sentences)", search_rows, rows=12)
    report.plots([
        (embedding_scatter(project_2d(vectors, "pca"), [t[:25] for t in texts], groups, "PCA projection"), "PCA"),
        (embedding_scatter(project_2d(vectors, "tsne"), [t[:25] for t in texts], groups, "t-SNE projection"), "t-SNE"),
    ])
    report.full_text("Key takeaways", "\n".join([
        "* Transformer embedding models map whole sentences to one vector capturing MEANING.",
        "* nomic-embed-text expects prefixes: 'search_document: ' for passages, 'search_query: ' for questions.",
        "* Normalised vectors -> cosine == dot product, which is what vector stores compute.",
        "* Embeddings are deterministic: cache them on disk so documents are embedded only once.",
    ]))
    report.save(__file__, "ollama_embeddings_report.html")


if __name__ == "__main__":
    main()
