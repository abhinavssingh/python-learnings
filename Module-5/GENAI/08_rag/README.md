# Retrieval-Augmented Generation (RAG)

RAG combines document retrieval with language-model generation. The system searches a knowledge base, places selected passages into a prompt, and asks a model to answer from that evidence. These lessons use the HR policy corpus and local Ollama models described in the [parent GenAI guide](../README.md).

## Why RAG matters

A plain language model answers from patterns learned during training. It does not automatically know the exact contents of your organization's policy documents unless you provide them.

Example:

```text
Question: "What is the policy for parental leave after 12 months of service?"
```

A general model may know broad concepts of parental leave, but it may not know your company policy wording or exceptions.

RAG solves this by retrieving the relevant policy passages and grounding the answer in them.

## The RAG pipeline

A typical RAG workflow is:

1. Split documents into chunks
2. Embed and store chunks in a vector store
3. Receive a user question
4. Retrieve candidate passages
5. Add the passages to the model prompt as context
6. Generate an answer grounded in that evidence
7. Return citations or source references

Example:

```text
User question: "How do I apply for annual leave?"
Retrieved passages:
- "Employees may request annual leave through the employee portal."
- "Annual leave requests require manager approval."

Model answer:
"Submit the request through the employee portal and include manager approval."
```

This is much more grounded and reliable than a model guessing from memory.

## Lessons

### 1. Naive RAG (`01_naive_rag.py`)

This is the core retrieve-augment-generate flow.

Example user query:

```text
"How does the company support gender balance?"
```

The system retrieves supporting passages, such as:

```text
"The company is committed to gender balance and supports fair hiring practices."
```

Then it passes those passages to the model together with the question.

The model answers from that evidence and cites the relevant passage.

This is a useful baseline to understand the entire architecture before adding optimizations.

### 2. Advanced RAG (`02_advanced_rag.py`)

This adds:
- hybrid retrieval
- query rewriting
- reranking
- compression

These steps aim to improve evidence quality.

Example:

```text
Question: "What policies apply to contractors?"
```

Naive retrieval might fetch mostly general HR text. An advanced retriever may rewrite the question to:

```text
"Which leave, benefits, or contractor-specific policies apply to non-employee staff?"
```

Then it reranks the results and keeps only the most relevant evidence.

This can improve recall and reduce noisy context.

### 3. Conversational RAG memory (`03_conversational_rag_memory.py`)

This handles follow-up questions in a chat setting.

Example:

```text
User: "What is the parental leave policy?"
Assistant: "Employees with 12 months of service are eligible."
User: "And how do I apply?"
```

The system must resolve the second question into a standalone search query such as:

```text
"How do employees apply for parental leave after 12 months of service?"
```

It should also keep the earlier context in mind while choosing context for generation.

This is a common and important real-world challenge in retrieval systems.

## The answer pipeline

1. Load and chunk source documents.
2. Embed and index chunks with useful metadata.
3. Retrieve candidate passages for a question.
4. Optionally rewrite, combine, rerank, or compress candidates.
5. Generate an answer grounded in the selected context.
6. Inspect answer quality, evidence, and citations together.

RAG can reduce unsupported answers, but it does not eliminate them. Bad extraction, poor chunk boundaries, weak retrieval, or ambiguous instructions can all lead to a confident but unsupported response.

## Example of bad RAG behavior

```text
Question: "Does the company allow remote work during sick leave?"
```

If the retrieved passages are unrelated and do not mention sick leave or remote work, the model may still answer something plausible but legally wrong.

This is why evaluation and evidence checking are essential.

## Run

From the repository root:

```powershell
python Module-5/GENAI/08_rag/01_naive_rag.py
python Module-5/GENAI/08_rag/03_conversational_rag_memory.py
```

Install the GenAI requirements, start Ollama, and pull the chat and embedding models per the parent guide. The golden questions live under `datasets/GEN AI/eval/`. HTML reports are saved in this folder's `reports/` directory.

## Practice

Use one golden question and trace its retrieved chunks, final prompt context, answer, and citation. Change only one stage at a time and note whether the evidence improves. Include an unanswerable question and check that the assistant does not invent policy details.

### Example practice case

Question:

```text
"What is the policy for emergency leave?"
```

If the source document does not mention emergency leave, a strong RAG system should answer:

```text
The policy does not specify emergency leave.
```

not:

```text
Employees get 10 days of emergency leave every year.
```

This is one of the clearest ways to test whether the system is grounded in retrieved evidence.

## Big picture

RAG is a practical pattern for grounding LLMs in real documents.

```text
documents -> chunking -> retrieval -> context -> model answer
```

It is especially powerful when the answer must be evidence-based, such as in HR policies, legal documents, internal knowledge bases, and customer support systems.
