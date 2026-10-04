# LangChain Integration

This folder shows how to use LangChain's runnable and tool abstractions while reusing the repository's local models and retrieval utilities. Learn the framework vocabulary here, then compare the same ideas with the graph-based control flow in `10_langgraph`.

## Why LangChain matters

LangChain is a framework for composing AI components into a pipeline. Instead of writing one big monolithic prompt script, it provides building blocks such as:
- prompts
- models
- retrievers
- tools
- output parsers
- runnables

This makes it easier to build reusable AI workflows.

Example workflow:

```python
prompt = PromptTemplate.from_template("Answer based on context: {context} Question: {question}")
llm = ChatOllama(model="qwen3:8b")
retriever = vectorstore.as_retriever()
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
```

This defines a retrieval-then-answer workflow as a readable composition.

## Lessons

### 1. LCEL basics (`01_lcel_basics.py`)

This lesson introduces LangChain Expression Language (LCEL), which is a way to compose runnables.

A runnable is any object that can accept input and produce output.

Examples of runnables:
- prompt templates
- model calls
- output parsers
- retrievers
- tool calls

#### Runnable composition
The idea is simple:

```python
chain = step1 | step2 | step3
```

This means the output of step1 becomes the input to step2, and so on.

Example:

```python
prompt = PromptTemplate.from_template("Answer: {question}")
llm = ChatOllama(model="qwen3:8b")
chain = prompt | llm
```

Now you can call:

```python
chain.invoke({"question": "What is the leave policy?"})
```

#### Batch and streaming
LangChain also supports:
- `batch()` for multiple requests at once
- `stream()` for progressive output

Example:

```python
result = chain.batch([
    {"question": "What is annual leave?"},
    {"question": "How do I file a claim?"}
])
```

Streaming is useful in UIs because the user sees output as it arrives rather than waiting for the final complete answer.

#### Structured output
A chain can also parse results into a structured schema.

Example:

```python
class Answer(BaseModel):
    category: str
    summary: str
```

Now the chain can return a typed object instead of raw text.

### 2. LangChain RAG chain (`02_langchain_rag_chain.py`)

This lesson adapts the repository's retrievers to LangChain's retriever interface.

Example:

```text
Question: "How do I apply for annual leave?"
```

The retriever fetches relevant policy chunks, and the model uses them as context to answer.

This is the LangChain form of the same RAG pattern we studied earlier.

The important idea is that the framework helps manage the flow, but it does not replace understanding of retrieval and prompt design.

### 3. LangChain tool calling (`03_langchain_tool_calling.py`)

This lesson demonstrates:
- `@tool` decorator
- `bind_tools`
- model/tool loop

Example tool:

```python
@tool
def calculate_percentage(percentage: float, value: float) -> float:
    return (percentage / 100) * value
```

The model may decide to call it when asked:

```text
What is 15% of 60000?
```

The application then executes the tool and sends the result back to the model.

This is the same pattern introduced in the LLM basics module, but using LangChain abstractions.

## What to notice

- Composition makes inputs and outputs between steps important. Read the chain's shape instead of treating it as one opaque call.
- Batch and streaming are different execution modes; streaming can improve perceived responsiveness but does not change factual quality.
- Tool definitions should be narrow, argument descriptions clear, and tool outputs validated.
- An adapter connects interfaces; it does not remove the need to understand the retriever or data source behind it.

## Example of a LangChain pattern

```python
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough

prompt = PromptTemplate.from_template("Context: {context}\nQuestion: {question}\nAnswer briefly.")
llm = ChatOllama(model="qwen3:8b")
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

This is a compact representation of a retrieval-augmented answer workflow.

## Run

From the repository root:

```powershell
python Module-5/GENAI/09_langchain/01_lcel_basics.py
python Module-5/GENAI/09_langchain/02_langchain_rag_chain.py
```

Model-backed lessons require the local setup in the [GenAI guide](../README.md), including Ollama models. Reports are written to this folder's `reports/` directory.

## Practice

Trace one value from the initial input through each runnable to the final output. Then make a batch call with two questions and compare its outputs with streaming. Add a small calculator tool and test normal, malformed, and out-of-scope requests.

### Example practice prompt

```text
What is 20% of 2500?
```

The model should call the calculator tool, and the application should validate the arguments before execution.

For another test:

```text
What is the best movie of all time?
```

This request may not fit your tool set, so the app should gracefully refuse or answer without calling a tool.

## Big picture

LangChain makes AI application workflows easier to express, but its abstractions still rely on the same principles we have studied throughout this module:

```text
prompting -> retrieval -> tools -> structured output -> validation
```

The framework does not replace good engineering; it helps organize it.
