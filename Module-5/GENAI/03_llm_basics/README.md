# Large Language Model Basics

These demos introduce chat models as programs that consume structured messages and generate continuations. They use Ollama locally, so you can experiment with model behavior without sending prompts to a hosted API. The examples use the configured chat model (the repository guide currently documents `qwen3:8b`).

## Why this matters

A large language model is not a magical oracle. It is a probabilistic system that predicts the next token based on patterns learned from training data. In practical app development, we use it as a component inside a bigger system: the application supplies messages, optional tools, and constraints, and the model produces a continuation.

This module focuses on the basics of how to interact with models safely and effectively.

## Background mental model

A chat model takes a sequence of messages and predicts a likely next message.

Example message sequence:

```python
messages = [
    {"role": "system", "content": "You are a helpful HR assistant."},
    {"role": "user", "content": "How do I request annual leave?"}
]
```

The model uses this context to generate an answer such as:

```text
You can request annual leave through the employee portal. Go to Leave Requests and submit the form.
```

The model is not searching a database by itself; it is generating a likely answer based on patterns and retrieved context, depending on the design of the app.

## Lessons

### 1. Chat basics and message roles (`01_ollama_chat_basics.py`)

This lesson demonstrates the structure of chat requests.

#### Message roles
- `system`: sets the global behavior or persona of the model
- `user`: contains the actual user request
- `assistant`: the model's previous response, useful in multi-turn conversations

Example:

```python
messages = [
    {"role": "system", "content": "You are a concise technical assistant."},
    {"role": "user", "content": "Explain the difference between a list and a tuple in Python."}
]
```

The model may respond with:

```text
A list is mutable and uses square brackets; a tuple is immutable and uses parentheses.
```

#### Streaming
Streaming means the model output arrives incrementally instead of all at once.

Example:

```text
Token 1: "A"
Token 2: " list"
Token 3: " is"
Token 4: " mutable"
```

This helps users see partial progress and is often useful in interactive apps or UIs.

#### Thinking mode
Some models expose a separate reasoning/thinking mode. This can be helpful for debugging or for understanding what the model is doing internally, but it is not the same as guaranteed correctness.

#### Usage tracing
The lesson also shows tracing model usage, which helps answer:
- how many tokens were used?
- how long did the response take?
- was the model generating quickly or slowly?

This is useful for cost control and observability in real applications.

### 2. Generation parameters (`02_generation_parameters.py`)

This lesson varies generation controls such as:
- temperature
- seed
- top-k
- top-p
- max tokens

These parameters change how the model samples the next token.

#### Temperature
Temperature controls randomness.

Example prompts:

```text
Write a short description of a mountain.
```

Low temperature:

```text
A mountain is a large natural elevation of the earth's surface.
```

High temperature:

```text
A mountain is a giant stone dream that wears clouds like a crown and keeps the sky company.
```

Low temperature gives more stable, conservative answers.
High temperature gives more creative and varied outputs.

#### Seed
A seed makes sampling more reproducible when the same settings are used again.

Example:

```python
seed = 42
```

With the same model, same prompt, and same seed, you often get the same output.

This is extremely useful for testing, regression checks, and debugging.

#### Top-k and top-p
These constrain token selection.

- Top-k chooses from the top `k` likely next tokens.
- Top-p chooses from a cumulative probability threshold.

Example:

```text
Prompt: "The capital of France is"
```

The model may consider:
- `"Paris"` (very likely)
- `"France"` (possible but less likely)
- `"beautiful"` (unlikely)

Top-k/top-p restrict this choice in different ways.

#### Max tokens
This limits how long the generated output can be.

Example:

```text
Generate a 2-sentence answer.
```

If `max_tokens` is too low, the model may cut off mid-answer.

#### Important caveat
These controls do not add factual knowledge. They only affect sampling behavior. A model can still produce incorrect or hallucinated results, even with low temperature.

### 3. Tool calling and agent loops (`03_tool_calling_native.py`)

This lesson shows a core design pattern: the LLM requests a tool call, the application runs the function, and the result is returned to the model.

#### Why this matters
A model cannot reliably do precise arithmetic, database queries, or system actions on its own unless the app provides a controlled execution layer.

Example request:

```text
What is 15% of 60,000?
```

The model may decide to call a calculator tool:

```json
{
  "name": "calculate_percentage",
  "arguments": {
    "percentage": 15,
    "value": 60000
  }
}
```

The Python application validates this and executes:

```python
def calculate_percentage(percentage, value):
    return value * percentage / 100
```

The result is then returned:

```text
15% of 60,000 is 9,000.
```

#### Agent loop pattern
The application usually follows this loop:

1. Send user request and available tools to the model
2. If model requests a tool, validate the request
3. Execute the tool safely
4. Feed the tool output back to the model
5. Continue until the model produces a final answer

This is the foundation of tool-using assistants.

#### Security mindset
A model-generated argument is not trusted input.

Example:

```text
"Delete all records in the database"
```

Even if the model emits a tool call, the application should only allow specific approved tools and validate the arguments before acting.

That is why tool definitions must be narrow and explicit.

## A useful mental model

A tool call is a structured request, not a magical action performed by the model. Your application decides which tools are available, validates arguments, executes approved functions, and sends results back. Keep tools narrow and treat model-generated arguments as untrusted input.

Example of a dangerous mindset:

```text
The model said to delete files, so I will do it.
```

Safe mindset:

```text
The model requested a specific operation; the app checked the requested parameters and allowed only the approved tool call path.
```

## Run

From the repository root:

```powershell
python Module-5/GENAI/03_llm_basics/01_ollama_chat_basics.py
python Module-5/GENAI/03_llm_basics/03_tool_calling_native.py
```

Before running, install the GenAI dependencies, start Ollama, and pull the chat model as described in the [GenAI setup guide](../README.md). Model-backed demos can take longer than the no-model lessons. Reports are saved under this folder's `reports/` directory.

## Practice

### Parameter experiment
Change one generation parameter at a time and keep the prompt fixed.

Example prompt:

```text
Write a 2-sentence description of a robot.
```

Try with:
- temperature = 0.2
- temperature = 0.8
- max_tokens = 20
- max_tokens = 100

Observe:
- which outputs remain stable
- which outputs become more creative
- which outputs become repetitive or cut off

### Tool safety exercise
Add one narrowly scoped tool and test:
- valid arguments
- invalid arguments
- ambiguous arguments

Example tool:

```python
def lookup_policy(policy_name: str) -> str:
    ...
```

Test inputs such as:
- `"annual_leave"` -> valid
- `""` -> invalid
- `"annual leave policy and employee data"` -> ambiguous / suspicious

The application should not blindly execute arbitrary model text.

## Big picture

This module highlights the difference between a model and an application:

```text
Model: predicts the next token
Application: decides which tools exist, validates inputs, runs functions, and controls the workflow
```

That design is the heart of real AI applications.

Without this layer, a model is just a text generator. With it, the model becomes a reasoning component inside a larger system that can manipulate tools, query data, and respond safely.
