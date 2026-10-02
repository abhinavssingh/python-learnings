# LangChain Integration

This folder shows how to use LangChain's runnable and tool abstractions while reusing the repository's local models and retrieval utilities. Learn the framework vocabulary here, then compare the same ideas with the graph-based control flow in `10_langgraph`.

## Lessons

1. `01_lcel_basics.py` introduces LangChain Expression Language (LCEL) runnables, composition, batch and stream operations, parallel branches, and structured output. Think of a chain as a typed sequence of steps: format input, call a model, and parse its result.
2. `02_langchain_rag_chain.py` builds a LangChain-native RAG chain and adapts the project's retrievers to LangChain's retriever interface. Ask a question about HR policy and inspect how retrieval output becomes model context.
3. `03_langchain_tool_calling.py` demonstrates the `@tool` decorator, `bind_tools`, and a manual model/tool loop. The model proposes a tool call; application code invokes the function and returns the tool result to the model.

## What to notice

- Composition makes inputs and outputs between steps important. Read the chain's shape instead of treating it as one opaque call.
- Batch and streaming are different execution modes; streaming can improve perceived responsiveness but does not change factual quality.
- Tool definitions should be narrow, argument descriptions clear, and tool outputs validated.
- An adapter connects interfaces; it does not remove the need to understand the retriever or data source behind it.

## Run

From the repository root:

```powershell
python Module-5/GENAI/09_langchain/01_lcel_basics.py
python Module-5/GENAI/09_langchain/02_langchain_rag_chain.py
```

Model-backed lessons require the local setup in the [GenAI guide](../README.md), including Ollama models. Reports are written to this folder's `reports/` directory.

## Practice

Trace one value from the initial input through each runnable to the final output. Then make a batch call with two questions and compare its outputs with streaming. Add a small calculator tool and test normal, malformed, and out-of-scope requests.
