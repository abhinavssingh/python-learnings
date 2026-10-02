from collections import Counter

import numpy as np


class Word2VecScratch:
    """
    Skip-gram Word2Vec with negative sampling, written in NumPy.

    For each (center, context) pair inside a window, the model learns
    vectors so that  sigmoid(v_center . u_context) -> 1  while random
    "negative" words give  sigmoid(v_center . u_negative) -> 0.
    """

    def __init__(self, embedding_dim: int = 50, window: int = 2, negative: int = 5,
                 min_count: int = 2, learning_rate: float = 0.025, epochs: int = 20, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.window = window
        self.negative = negative
        self.min_count = min_count
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: list[str] = []
        self.W_in: np.ndarray = np.array([])
        self.W_out: np.ndarray = np.array([])
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def _build_vocab(self, sentences: list[list[str]]) -> None:
        counts = Counter(w for s in sentences for w in s)
        self.id_to_word = [w for w, c in counts.most_common() if c >= self.min_count]
        self.word_to_id = {w: i for i, w in enumerate(self.id_to_word)}
        freqs = np.array([counts[w] for w in self.id_to_word], dtype=float) ** 0.75
        self.noise_dist = freqs / freqs.sum()

    def fit(self, sentences: list[list[str]]) -> "Word2VecScratch":
        self._build_vocab(sentences)
        vocab_size = len(self.id_to_word)
        self.W_in = self.rng.uniform(-0.5, 0.5, (vocab_size, self.embedding_dim)) / self.embedding_dim
        self.W_out = np.zeros((vocab_size, self.embedding_dim))

        encoded = [[self.word_to_id[w] for w in s if w in self.word_to_id] for s in sentences]
        pairs = [
            (center, sentence[j])
            for sentence in encoded
            for i, center in enumerate(sentence)
            for j in range(max(0, i - self.window), min(len(sentence), i + self.window + 1))
            if j != i
        ]
        if not pairs:
            raise ValueError("Corpus too small: no training pairs. Lower min_count or add text.")
        pairs_arr = np.array(pairs)

        for epoch in range(self.epochs):
            lr = self.learning_rate * (1 - epoch / self.epochs) + 1e-4
            self.rng.shuffle(pairs_arr)
            epoch_loss = 0.0
            for center, context in pairs_arr:
                negatives = self.rng.choice(vocab_size, size=self.negative, p=self.noise_dist)
                targets = np.concatenate(([context], negatives))
                labels = np.zeros(len(targets))
                labels[0] = 1.0

                v = self.W_in[center]
                u = self.W_out[targets]
                preds = self._sigmoid(u @ v)
                errors = preds - labels

                epoch_loss += -np.log(preds[0] + 1e-9) - np.log(1 - preds[1:] + 1e-9).sum()
                self.W_out[targets] -= lr * np.outer(errors, v)
                self.W_in[center] -= lr * (errors @ u)
            self.loss_history.append(epoch_loss / len(pairs_arr))
        return self

    @property
    def vectors(self) -> np.ndarray:
        return self.W_in

    def vector(self, word: str) -> np.ndarray:
        return self.W_in[self.word_to_id[word]]

    def most_similar(self, word: str, topn: int = 5) -> list[tuple[str, float]]:
        if word not in self.word_to_id:
            return []
        normed = self.W_in / np.linalg.norm(self.W_in, axis=1, keepdims=True).clip(min=1e-12)
        sims = normed @ normed[self.word_to_id[word]]
        order = np.argsort(sims)[::-1]
        return [(self.id_to_word[i], float(sims[i])) for i in order if self.id_to_word[i] != word][:topn]
