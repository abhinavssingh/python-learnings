import numpy as np
import pandas as pd

from ..foundations.similarity import cosine_similarity_matrix, l2_normalize


def similarity_matrix(embeddings: np.ndarray, labels: list[str] | None = None) -> pd.DataFrame:
    matrix = cosine_similarity_matrix(embeddings)
    labels = labels or [str(i) for i in range(len(matrix))]
    return pd.DataFrame(np.round(matrix, 3), index=labels, columns=labels)


def nearest_neighbours(query_vector: np.ndarray, matrix: np.ndarray, labels: list[str], k: int = 5) -> list[tuple[str, float]]:
    sims = (l2_normalize(matrix) @ l2_normalize(query_vector).ravel())
    order = np.argsort(sims)[::-1][:k]
    return [(labels[i], float(sims[i])) for i in order]


def project_2d(embeddings: np.ndarray, method: str = "pca", seed: int = 42) -> np.ndarray:
    """Reduce embeddings to 2-D for plotting (PCA or t-SNE)."""
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = max(2, min(30, len(embeddings) - 1) // 3)
        return TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca").fit_transform(embeddings)
    raise ValueError("method must be 'pca' or 'tsne'")


def vector_arithmetic(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Classic analogy: a - b + c  (e.g. king - man + woman)."""
    return a - b + c
