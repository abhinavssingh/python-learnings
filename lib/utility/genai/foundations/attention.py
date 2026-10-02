import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def causal_mask(seq_len: int) -> np.ndarray:
    """True where attention is allowed (token i can see tokens <= i)."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    positions = np.arange(seq_len)[:, None]
    dims = np.arange(d_model)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / d_model)
    angles = positions * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None):
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Returns (output, attention_weights).
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


class MultiHeadAttention:
    """NumPy multi-head self-attention with random (untrained) projections."""

    def __init__(self, d_model: int, num_heads: int, seed: int = 42):
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        scale = 1 / np.sqrt(d_model)
        self.W_q = rng.normal(0, scale, (d_model, d_model))
        self.W_k = rng.normal(0, scale, (d_model, d_model))
        self.W_v = rng.normal(0, scale, (d_model, d_model))
        self.W_o = rng.normal(0, scale, (d_model, d_model))

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.num_heads, self.d_head).transpose(1, 0, 2)

    def __call__(self, x: np.ndarray, mask: np.ndarray | None = None):
        Q = self._split_heads(x @ self.W_q)
        K = self._split_heads(x @ self.W_k)
        V = self._split_heads(x @ self.W_v)
        heads, weights = scaled_dot_product_attention(Q, K, V, mask)
        concat = heads.transpose(1, 0, 2).reshape(x.shape[0], self.d_model)
        return concat @ self.W_o, weights
