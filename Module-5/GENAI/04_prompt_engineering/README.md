# Prompt Engineering and Output Guardrails

This folder studies how to give a model a clear task and how to constrain or validate what comes back. Prompting improves the odds of a useful response; validation and safety checks are still needed because instructions alone do not guarantee correctness.

## Lessons

1. `01_prompting_techniques.py` compares zero-shot and few-shot examples, role instructions, chain-of-thought-style prompting, and a reusable prompt library. For example, ask the model to classify a policy question as `leave`, `benefits`, or `other`; then provide two labeled examples and compare consistency. Avoid requesting or relying on hidden reasoning: focus the prompt on a concise answer and verifiable evidence.
2. `02_structured_output_guardrails.py` demonstrates Pydantic-backed JSON output, schema-constrained responses, personally identifiable information (PII) checks, and prompt-injection handling. A useful exercise is to request `{category, urgency, summary}` for a support message and reject a response that does not satisfy the schema.

## Learner takeaways

- State the task, relevant context, constraints, and expected output format explicitly.
- Examples can clarify a format or classification boundary, but can also bias the result if they are unrepresentative.
- Parse structured output and validate its fields in application code; do not assume JSON-looking text is valid JSON or semantically correct.
- Treat user content and retrieved documents as data, not trusted instructions. A prompt-injection check is one defense, not a complete security boundary.
- Minimize sensitive data and avoid logging it unnecessarily.

## Run

From the repository root:

```powershell
python Module-5/GENAI/04_prompt_engineering/01_prompting_techniques.py
python Module-5/GENAI/04_prompt_engineering/02_structured_output_guardrails.py
```

These demos use the local chat model. Follow the [GenAI setup guide](../README.md) for installation and Ollama model setup. Each script produces a report in this folder's `reports/` directory.

## Practice

Write a prompt that extracts a policy topic and a short answer from a paragraph. Test it on normal text, missing information, contradictory instructions, and an injection-like sentence. Confirm that your application validates the output and can say that the source does not contain enough information.
