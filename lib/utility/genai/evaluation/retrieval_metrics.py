import math
from typing import Callable

import numpy as np
import pandas as pd

from ..abstractions.base_retriever import BaseRetriever
from ..abstractions.document import SearchResult


def hit_rate_at_k(relevance: list[bool], k: int) -> float:
    return float(any(relevance[:k]))


def precision_at_k(relevance: list[bool], k: int) -> float:
    return sum(relevance[:k]) / k if k else 0.0


def recall_at_k(relevance: list[bool], k: int, total_relevant: int) -> float:
    return min(1.0, sum(relevance[:k]) / total_relevant) if total_relevant else 0.0


def reciprocal_rank(relevance: list[bool]) -> float:
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            return 1 / rank
    return 0.0


def ndcg_at_k(relevance: list[bool], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, rel in enumerate(relevance[:k]) if rel)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, sum(relevance) or 0)))
    return dcg / ideal if ideal else 0.0


def page_relevance(item: dict) -> Callable[[SearchResult], bool]:
    """A retrieved chunk is relevant if it comes from one of the item's `relevant_pages`."""
    pages = set(item.get("relevant_pages", []))
    return lambda r: r.metadata.get("page") in pages


def keyword_relevance(item: dict, min_hits: int = 1) -> Callable[[SearchResult], bool]:
    keywords = [k.lower() for k in item.get("expected_keywords", [])]
    return lambda r: sum(k in r.text.lower() for k in keywords) >= min_hits


class RetrievalEvaluator:
    """Run a golden question set through retrievers and compute IR metrics."""

    def __init__(self, dataset: list[dict], k: int = 4, relevance_fn=page_relevance):
        self.dataset = dataset
        self.k = k
        self.relevance_fn = relevance_fn

    def evaluate(self, retriever: BaseRetriever, name: str | None = None) -> pd.DataFrame:
        rows = []
        for item in self.dataset:
            results = retriever.retrieve(item["question"], self.k)
            is_rel = self.relevance_fn(item)
            relevance = [is_rel(r) for r in results] + [False] * (self.k - len(results))
            total_relevant = max(1, len(item.get("relevant_pages", [])))
            rows.append({
                "retriever": name or retriever.name,
                "question": item["question"],
                f"hit@{self.k}": hit_rate_at_k(relevance, self.k),
                f"precision@{self.k}": precision_at_k(relevance, self.k),
                f"recall@{self.k}": recall_at_k(relevance, self.k, total_relevant),
                "mrr": reciprocal_rank(relevance),
                f"ndcg@{self.k}": ndcg_at_k(relevance, self.k),
            })
        return pd.DataFrame(rows)

    def compare(self, retrievers: dict[str, BaseRetriever]) -> tuple[pd.DataFrame, pd.DataFrame]:
        detail = pd.concat([self.evaluate(r, name) for name, r in retrievers.items()], ignore_index=True)
        summary = detail.drop(columns=["question"]).groupby("retriever").mean().round(3)
        summary = summary.sort_values("mrr", ascending=False).reset_index()
        return summary, detail


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}
