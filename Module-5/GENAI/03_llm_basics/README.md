# Large Language Model Basics

These demos introduce chat models as programs that consume structured messages and generate continuations. They use Ollama locally, so you can experiment with model behavior without sending prompts to a hosted API. The examples use the configured chat model (the repository guide currently documents `qwen3:8b`).

## Lessons

1. `01_ollama_chat_basics.py` demonstrates message roles, streaming, thinking mode, and usage tracing. A `system` instruction establishes behavior, a `user` message asks a question, and an `assistant` message is the response. Streaming lets you observe output as it arrives rather than waiting for the complete answer.
2. `02_generation_parameters.py` varies temperature, seed, top-k/top-p sampling, and maximum output tokens. Ask for three short descriptions of the same object, then compare a low-temperature run with a more varied run. These controls affect generation; they do not add knowledge or guarantee correctness.
3. `03_tool_calling_native.py` demonstrates native function/tool calling and the application-side agent loop. For a request such as `"What is 15% of 60,000?"`, the model can request a calculator tool; the Python program executes the function and returns its result to the model.

## A useful mental model

A tool call is a structured request, not a magical action performed by the model. Your application decides which tools are available, validates arguments, executes approved functions, and sends results back. Keep tools narrow and treat model-generated arguments as untrusted input.

## Run

From the repository root:

```powershell
python Module-5/GENAI/03_llm_basics/01_ollama_chat_basics.py
python Module-5/GENAI/03_llm_basics/03_tool_calling_native.py
```

Before running, install the GenAI dependencies, start Ollama, and pull the chat model as described in the [GenAI setup guide](../README.md). Model-backed demos can take longer than the no-model lessons. Reports are saved under this folder's `reports/` directory.

## Practice

Change one generation parameter at a time and keep the prompt fixed. Record which outputs change and which do not. Then add one narrowly scoped tool and test valid, invalid, and ambiguous arguments; the application should not blindly execute arbitrary model text.
