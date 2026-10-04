# Evaluating RAG Systems

A RAG system has at least two quality questions: did retrieval find the right evidence, and did generation use that evidence correctly? This folder evaluates both instead of treating a plausible-sounding answer as success.

## Why evaluation matters

A model can sound confident and still be wrong. In enterprise or policy use cases, this matters a lot because a confident hallucination is still a failure.

Example:

```text
Question: "What is the company policy for extended sick leave?"
Model answer: "Employees can take 30 days of extended sick leave per year."
```

This may sound reasonable, but if the policy actually says something else or does not mention it, the answer is wrong even though it is fluent.

This is why evaluation is essential.

## What is being evaluated?

RAG evaluation usually looks at multiple layers.

### 1. Retrieval quality
Did the system fetch the right evidence?

Example:

```text
Question: "How do employees request annual leave?"
Relevant passage: "Employees may request annual leave through the HR portal."
```

If the retriever brings back irrelevant passages about payroll or travel, retrieval quality is poor.

### 2. Generation quality
Did the model answer the question correctly using the retrieved evidence?

Example:

```text
Question: "How do employees request annual leave?"
Generated answer: "Employees submit their request through the HR portal."
```

This is good if it matches the policy.

### 3. Citation or grounding quality
Did the answer point to the right source? 

This matters because a response can be factual but still untraceable unless the supporting evidence is clear.

## Lesson

`01_rag_evaluation.py` runs the HR-policy golden question set and reports retrieval and generation measures, including an LLM-as-judge assessment. The set is in `datasets/GEN AI/eval/hr_policy_qa.json`; each entry includes a question and reference information such as expected answer details, keywords, or relevant pages.

For a question such as `"How does the company support gender balance?"`, examine separately:

- Whether the retrieved passages include the relevant policy pages.
- Whether the generated response answers the question using those passages.
- Whether citations or supporting details point back to the source.
- Whether the judge's assessment agrees with manual review.

## Example metrics

### Hit@k
Did the correct passage appear in the top `k` retrieved results?

```text
Top 5 results: [travel policy, leave policy, payroll policy]
Relevant result: leave policy
```

If `leave policy` is in the top 3, then hit@3 is 1.

### Precision@k
How many of the top `k` results are actually relevant?

### Recall@k
How much of the relevant evidence was retrieved within the top `k`?

### MRR (Mean Reciprocal Rank)
Does the correct result appear early in the ranking?

### nDCG
Does the ranking place more relevant results higher than less relevant ones?

These measures help answer different questions. A system can have good recall but poor ranking if the relevant evidence is buried.

## LLM-as-judge

This evaluation method asks a model to rate or compare answers.

Example:

```text
Score whether this answer is grounded in the retrieved evidence and directly answers the question.
```

This is useful, but it is also a model-based judgment and can be inconsistent. It should be treated as one signal, not as the sole truth.

The top-level guide documents `GENAI_EVAL_LIMIT` (default 4) to limit the number of questions during a quick run. Use the full set for a more complete check.

## Run

From the repository root:

```powershell
python Module-5/GENAI/11_evaluation/01_rag_evaluation.py
```

This demo requires the local chat and embedding models and the source corpus; setup steps are in the [GenAI guide](../README.md). Its HTML report is written under this folder's `reports/` directory.

## Practice

Choose one answer that scores well and one that scores poorly. Inspect the retrieved passages and compare the response to the reference. Decide whether each issue is retrieval, context construction, generation, or evaluation disagreement; then change one component and rerun the same questions.

### Example debugging workflow

1. Take a poor answer.
2. Inspect top retrieved passages.
3. See whether the relevant evidence was absent.
4. Check if the model ignored the source context.
5. Adjust retrieval or prompt construction.
6. Rerun and compare scores.

This is a practical way to diagnose the root cause of a RAG error.

## Big picture

Evaluation is how we know whether a system is actually useful.

```text
retrieval score + generation score + manual review + judge check
```

A system with a fluent answer but weak evidence is not reliable. Good AI engineering requires looking at the whole pipeline, not just the final sentence.
