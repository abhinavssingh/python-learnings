# Evaluation

Evaluate retrieval and generated answers with reference-based metrics and
optional LLM judging.

- `RetrievalEvaluator` and standalone metrics cover hit rate, precision,
  recall, reciprocal rank, nDCG, keyword relevance, and page relevance.
- Generation metrics include exact match, token F1, BLEU, ROUGE-L, and keyword
  recall.
- `RAGEvaluator` combines RAG evaluation; `LLMJudge` provides model-based
  judgments.

The learning evaluation set is `datasets/GEN AI/eval/hr_policy_qa.json`.
`GENAI_EVAL_LIMIT` controls how many questions the evaluation script runs.
See `Module-5/GENAI/11_evaluation/`.
