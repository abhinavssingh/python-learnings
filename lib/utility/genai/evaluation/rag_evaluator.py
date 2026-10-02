import pandas as pd

from ..rag.pipelines import RAGResult
from .generation_metrics import generation_scores
from .llm_judge import LLMJudge
from .retrieval_metrics import page_relevance, reciprocal_rank


class RAGEvaluator:
    """
    End-to-end RAG evaluation on a golden QA set.

    For every question: run the pipeline, then compute
      - retrieval: hit / MRR against `relevant_pages`
      - generation: token F1, ROUGE-L, keyword recall vs the reference
      - (optional) LLM judge: faithfulness, answer & context relevance
    """

    def __init__(self, dataset: list[dict], judge: LLMJudge | None = None):
        self.dataset = dataset
        self.judge = judge

    def evaluate(self, pipeline, name: str | None = None, progress: bool = True) -> tuple[pd.DataFrame, list[RAGResult]]:
        name = name or getattr(pipeline, "name", type(pipeline).__name__)
        rows, results = [], []
        for i, item in enumerate(self.dataset, start=1):
            if progress:
                print(f"  [{name}] {i}/{len(self.dataset)}: {item['question']}")
            result: RAGResult = pipeline.ask(item["question"])
            results.append(result)
            is_rel = page_relevance(item)
            relevance = [is_rel(r) for r in result.sources]
            row = {
                "pipeline": name,
                "question": item["question"],
                "answer": result.answer,
                "hit": float(any(relevance)),
                "mrr": reciprocal_rank(relevance),
                "latency_s": round(result.latency_s, 2),
                **generation_scores(result.answer, item.get("reference_answer", ""), item.get("expected_keywords")),
            }
            if self.judge:
                row.update(self.judge.evaluate(item["question"], result.answer, result.context, item.get("reference_answer")))
            rows.append(row)
        return pd.DataFrame(rows), results

    @staticmethod
    def summary(frames: list[pd.DataFrame]) -> pd.DataFrame:
        detail = pd.concat(frames, ignore_index=True)
        numeric = detail.select_dtypes("number").columns.tolist()
        return detail.groupby("pipeline")[numeric].mean().round(3).reset_index()
