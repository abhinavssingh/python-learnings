# Retrieval-Augmented Generation (RAG)

RAG combines document retrieval with language-model generation. The system searches a knowledge base, places selected passages into a prompt, and asks a model to answer from that evidence. These lessons use the HR policy corpus and local Ollama models described in the [parent GenAI guide](../README.md).

## Lessons

1. `01_naive_rag.py` demonstrates the core retrieve-augment-generate flow and citations. Try `"How does the company support gender balance?"`; inspect which passages were retrieved, then verify that the response and citations are supported by those passages.
2. `02_advanced_rag.py` adds hybrid retrieval, query rewriting, reranking, and compression. Compare its evidence set with the naive version to see whether extra stages improve relevance or merely add complexity and latency.
3. `03_conversational_rag_memory.py` covers memory approaches and condensing follow-up questions. After asking about a policy, ask `"And how do I apply?"`; the system must resolve the follow-up into a standalone search question while using conversation history appropriately.

## The answer pipeline

1. Load and chunk source documents.
2. Embed and index chunks with useful metadata.
3. Retrieve candidate passages for a question.
4. Optionally rewrite, combine, rerank, or compress candidates.
5. Generate an answer grounded in the selected context.
6. Inspect answer quality, evidence, and citations together.

RAG can reduce unsupported answers, but it does not eliminate them. Bad extraction, poor chunk boundaries, weak retrieval, or ambiguous instructions can all lead to a confident but unsupported response.

## Run

From the repository root:

```powershell
python Module-5/GENAI/08_rag/01_naive_rag.py
python Module-5/GENAI/08_rag/03_conversational_rag_memory.py
```

Install the GenAI requirements, start Ollama, and pull the chat and embedding models per the parent guide. The golden questions live under `datasets/GEN AI/eval/`. HTML reports are saved in this folder's `reports/` directory.

## Practice

Use one golden question and trace its retrieved chunks, final prompt context, answer, and citation. Change only one stage at a time and note whether the evidence improves. Include an unanswerable question and check that the assistant does not invent policy details.
