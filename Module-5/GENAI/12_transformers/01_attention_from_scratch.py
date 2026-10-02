"""
Transformer building blocks from scratch (NumPy).

Learn:
  * scaled dot-product attention: softmax(Q K^T / sqrt(d)) V
  * causal masking (decoder-only LLMs cannot look at future tokens)
  * sinusoidal positional encodings
  * multi-head attention: several attention patterns in parallel
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

import numpy as np  # noqa: E402

from lib.utility.genai.foundations import (  # noqa: E402
    MultiHeadAttention,
    causal_mask,
    positional_encoding,
    scaled_dot_product_attention,
    softmax,
)
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.visualization import attention_heatmap, positional_encoding_heatmap  # noqa: E402

TOKENS = ["the", "manager", "coached", "the", "new", "employee"]


def main():
    rng = np.random.default_rng(0)
    d_model = 16
    x = rng.normal(size=(len(TOKENS), d_model)) + positional_encoding(len(TOKENS), d_model)

    W_q, W_k, W_v = (rng.normal(scale=0.4, size=(d_model, d_model)) for _ in range(3))
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    _, weights = scaled_dot_product_attention(Q, K, V)
    _, causal_weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(len(TOKENS)))

    mha = MultiHeadAttention(d_model=d_model, num_heads=4)
    mha_out, head_weights = mha(x)

    logits = np.array([2.0, 1.0, 0.1])
    temperature_rows = [{"temperature": t, "probabilities": np.round(softmax(logits / t), 3).tolist()} for t in (0.5, 1.0, 2.0)]

    report = GenAIReport("Attention & Transformers from scratch")
    report.grid([
        report.kv("Shapes", {"tokens": len(TOKENS), "d_model": d_model, "Q/K/V": Q.shape,
                             "attention weights": weights.shape, "multi-head output": mha_out.shape,
                             "heads x weights": np.asarray(head_weights).shape}),
        report.table("Softmax with temperature (same logits [2, 1, 0.1])", temperature_rows),
    ])
    report.plots([
        (attention_heatmap(weights, TOKENS, "Bidirectional self-attention (encoder / BERT)"), "Full attention"),
        (attention_heatmap(causal_weights, TOKENS, "Causal self-attention (decoder / GPT, Qwen)"), "Causal mask"),
    ])
    report.plots([(attention_heatmap(np.asarray(head_weights)[h], TOKENS, f"Head {h + 1}"), f"Head {h + 1}")
                  for h in range(4)])
    report.full_plot(positional_encoding_heatmap(positional_encoding(50, 64)), "Positional encoding (50 positions x 64 dims)")
    report.full_text("Key takeaways", "\n".join([
        "* Attention lets every token build its representation from a weighted mix of other tokens.",
        "* Dividing by sqrt(d_k) keeps softmax from saturating as dimensions grow.",
        "* The causal mask (upper triangle = 0) is why LLMs generate left-to-right, one token at a time.",
        "* Attention is order-agnostic -> positional encodings inject word order.",
        "* Softmax temperature here is exactly the `temperature` parameter used when sampling from an LLM.",
    ]))
    report.save(__file__, "attention_from_scratch_report.html")


if __name__ == "__main__":
    main()
