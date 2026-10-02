# Evaluating RAG Systems

A RAG system has at least two quality questions: did retrieval find the right evidence, and did generation use that evidence correctly? This folder evaluates both instead of treating a plausible-sounding answer as success.

## Lesson

`01_rag_evaluation.py` runs the HR-policy golden question set and reports retrieval and generation measures, including an LLM-as-judge assessment. The set is in `datasets/GEN AI/eval/hr_policy_qa.json`; each entry includes a question and reference information such as expected answer details, keywords, or relevant pages.

For a question such as `"How does the company support gender balance?"`, examine separately:

- Whether the retrieved passages include the relevant policy pages.
- Whether the generated response answers the question using those passages.
- Whether citations or supporting details point back to the source.
- Whether the judge's assessment agrees with manual review.

## How to interpret results

Retrieval metrics such as hit@k, recall, and ranking measures evaluate candidate evidence; answer similarity or an LLM judge evaluates a different layer. The judge is another model-based signal and can be inconsistent, so read examples manually. A small question set helps development but does not prove general quality or fairness.

The top-level guide documents `GENAI_EVAL_LIMIT` (default 4) to limit the number of questions during a quick run. Use the full set for a more complete check.

## Run

From the repository root:

```powershell
python Module-5/GENAI/11_evaluation/01_rag_evaluation.py
```

This demo requires the local chat and embedding models and the source corpus; setup steps are in the [GenAI guide](../README.md). Its HTML report is written under this folder's `reports/` directory.

## Practice

Choose one answer that scores well and one that scores poorly. Inspect the retrieved passages and compare the response to the reference. Decide whether each issue is retrieval, context construction, generation, or evaluation disagreement; then change one component and rerun the same questions.
