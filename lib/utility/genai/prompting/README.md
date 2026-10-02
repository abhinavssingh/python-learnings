# Prompting

Prompt templates, prompt-library access, output parsers, and input/output
guardrails for LLM workflows.

- `PromptTemplate`, `ChatPromptTemplate`, and `FewShotPromptTemplate` build
  prompts; `PromptLibrary` loads the shared templates in `config/prompts/`.
- JSON, list, yes/no, and Pydantic parsers convert or validate model output.
- `OutputValidator`, `PIIMasker`, and `PromptInjectionDetector` provide
  guardrail helpers.

Import public helpers from `lib.utility.genai.prompting`. See
`Module-5/GENAI/04_prompt_engineering/` for examples.
