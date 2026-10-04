# Transformer Building Blocks

This folder makes the attention mechanism visible by implementing its key pieces in a small learning example. The aim is to understand why attention mixes information across token positions, not to train a production language model from scratch.

## Why transformers matter

The transformer architecture is the foundation of modern LLMs. At the center of it is attention.

Attention lets a model answer questions like:

```text
When I read the word "it" in a sentence, which earlier word does it refer to?
```

Example sentence:

```text
"The model learns from data because it improves with examples."
```

The word `"it"` may attend strongly to `"model"`, not to the word `"data"` or `"examples"`.

This is what gives transformers their ability to model context over long sequences.

## Core idea: attention

Each token is converted into a vector, and each vector is used to compute:
- a query
- a key
- a value

Then attention weights are computed using the similarity between queries and keys.

Example:

```text
Sequence: "the model learns"
```

For each token, the model asks:

```text
How much should I pay attention to each other token when building this token's representation?
```

The output is a weighted sum of value vectors.

## Causal masking

In autoregressive generation, a token cannot attend to future tokens.

Example:

```text
Sequence: "the model learns"
```

When generating the token `"learns"`, the model may attend to `"the"` and `"model"`, but not to future tokens that do not exist yet.

This is enforced by a causal mask.

Example matrix:

```text
[[1, 0, 0],
 [1, 1, 0],
 [1, 1, 1]]
```

This ensures that each position can only attend to earlier positions and itself.

## Positional encoding

Without positional information, a transformer would treat the sequence as a bag of tokens.

Example:

```text
"Paris is the capital of France"
```

vs

```text
"France is the capital of Paris"
```

The words are the same, but the order matters.

Positional encodings add information about token position so the model can distinguish sequence order.

## Multi-head attention

A transformer does not use one attention pattern only. It uses multiple heads.

Each head can learn a different type of relation.

Example:

- Head 1: lexical or local relations
- Head 2: subject-verb connections
- Head 3: broader semantic similarity

This is why multi-head attention is so powerful.

## Lesson

`01_attention_from_scratch.py` demonstrates attention, a causal mask, positional encoding, and multi-head attention. For a sequence like `"the model learns"`, attention assigns weights to other positions when building a representation for each token. A causal mask prevents a position from attending to future tokens during next-token prediction.

## Concepts to take away

- Attention scores express how strongly positions interact; softmax converts scores into weights used to combine value vectors.
- Positional information lets the model distinguish token order; without it, attention alone does not encode sequence order.
- Multiple heads can learn different interaction patterns in parallel.
- Causal masking is essential for autoregressive generation so training cannot see the answer ahead of time.
- The implementation is intentionally educational and small; it is not a replacement for optimized transformer libraries.

## Example attention calculation

```text
Query vector for token 2: [0.7, 0.2]
Key vectors for all positions: [0.1, 0.2], [0.8, 0.4], [0.3, 0.9]
```

Dot products produce scores, which are then passed through softmax to produce normalized attention weights.

The strongest weight may point to a nearby or semantically related token, which is exactly how the model mixes information across positions.

## Run

From the repository root:

```powershell
python Module-5/GENAI/12_transformers/01_attention_from_scratch.py
```

The lesson is designed as a quick, no-Ollama experiment. It writes an HTML report under this folder's `reports/` directory. For the broader GenAI curriculum, see the [parent guide](../README.md).

## Practice

Inspect the attention weights and mask for a short sequence. Predict which entries should be blocked by the causal mask, then verify the result. Try changing sequence length or the number of heads and describe what changes in tensor shapes and what stays conceptually the same.

### Example practice task

Sequence:

```text
"the cat sat"
```

Before the model predicts the next word, it may look at positions for `"the"`, `"cat"`, and `"sat"` in order. The causal mask should block any attention to future tokens that do not yet exist.

This gives a direct view of how transformers build contextual token representations.

## Big picture

The transformer is a stack of attention layers, each allowing tokens to gather context from other tokens.

```text
tokens -> embeddings -> positional encoding -> attention -> feed-forward -> next-layer representation
```

This is the architectural core behind nearly all modern language models, including generative models, retrieval-enhanced systems, and multimodal models.
