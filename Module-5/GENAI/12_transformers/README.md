# Transformer Building Blocks

This folder makes the attention mechanism visible by implementing its key pieces in a small learning example. The aim is to understand why attention mixes information across token positions, not to train a production language model from scratch.

## Lesson

`01_attention_from_scratch.py` demonstrates attention, a causal mask, positional encoding, and multi-head attention. For a sequence like `"the model learns"`, attention assigns weights to other positions when building a representation for each token. A causal mask prevents a position from attending to future tokens during next-token prediction.

## Concepts to take away

- Attention scores express how strongly positions interact; softmax converts scores into weights used to combine value vectors.
- Positional information lets the model distinguish token order; without it, attention alone does not encode sequence order.
- Multiple heads can learn different interaction patterns in parallel.
- Causal masking is essential for autoregressive generation so training cannot see the answer ahead of time.
- The implementation is intentionally educational and small; it is not a replacement for optimized transformer libraries.

## Run

From the repository root:

```powershell
python Module-5/GENAI/12_transformers/01_attention_from_scratch.py
```

The lesson is designed as a quick, no-Ollama experiment. It writes an HTML report under this folder's `reports/` directory. For the broader GenAI curriculum, see the [parent guide](../README.md).

## Practice

Inspect the attention weights and mask for a short sequence. Predict which entries should be blocked by the causal mask, then verify the result. Try changing sequence length or the number of heads and describe what changes in tensor shapes and what stays conceptually the same.
