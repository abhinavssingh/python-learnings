import math
import random
from collections import Counter, defaultdict

BOS = "<s>"
EOS = "</s>"


def generate_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(generate_ngrams(tokens, n))


class NGramLanguageModel:
    """
    Statistical n-gram language model.

        P(w_i | w_{i-n+1} ... w_{i-1}) = count(context, w_i) / count(context)

    Smoothing
    ---------
    - "none":    maximum likelihood (unseen n-grams get probability 0)
    - "laplace": add-one smoothing
    - "add_k":   add-k smoothing (k < 1 usually works better)
    """

    def __init__(self, n: int = 2, smoothing: str = "laplace", k: float = 1.0):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.smoothing = smoothing
        self.k = 1.0 if smoothing == "laplace" else k
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.next_words: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def _pad(self, sentence: list[str]) -> list[str]:
        return [BOS] * (self.n - 1) + sentence + [EOS]

    def fit(self, sentences: list[list[str]]) -> "NGramLanguageModel":
        for sentence in sentences:
            padded = self._pad(sentence)
            self.vocab.update(padded)
            for gram in generate_ngrams(padded, self.n):
                context, word = gram[:-1], gram[-1]
                self.ngram_counts[gram] += 1
                self.context_counts[context] += 1
                self.next_words[context][word] += 1
        self.vocab.discard(BOS)
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def prob(self, word: str, context: tuple[str, ...]) -> float:
        context = tuple(context)[-(self.n - 1):] if self.n > 1 else ()
        count = self.ngram_counts[context + (word,)]
        context_count = self.context_counts[context]
        if self.smoothing == "none":
            return count / context_count if context_count else 0.0
        return (count + self.k) / (context_count + self.k * self.vocab_size)

    def sentence_log_prob(self, sentence: list[str]) -> float:
        padded = self._pad(sentence)
        total = 0.0
        for gram in generate_ngrams(padded, self.n):
            p = self.prob(gram[-1], gram[:-1])
            total += math.log(p) if p > 0 else float("-inf")
        return total

    def perplexity(self, sentences: list[list[str]]) -> float:
        """exp(-average log-probability per token). Lower is better."""
        log_prob, count = 0.0, 0
        for sentence in sentences:
            log_prob += self.sentence_log_prob(sentence)
            count += len(sentence) + 1
        if count == 0:
            return float("inf")
        return math.exp(-log_prob / count) if log_prob != float("-inf") else float("inf")

    def next_word_distribution(self, context: list[str], top: int = 10) -> list[tuple[str, float]]:
        ctx = tuple(([BOS] * (self.n - 1) + list(context))[-(self.n - 1):]) if self.n > 1 else ()
        candidates = self.next_words.get(ctx)
        if not candidates:
            return []
        return [(w, self.prob(w, ctx)) for w, _ in candidates.most_common(top)]

    def generate(self, seed: list[str] | None = None, max_words: int = 20, rng_seed: int = 42) -> list[str]:
        rng = random.Random(rng_seed)
        words = list(seed or [])
        for _ in range(max_words):
            ctx = tuple(([BOS] * (self.n - 1) + words)[-(self.n - 1):]) if self.n > 1 else ()
            candidates = self.next_words.get(ctx)
            if not candidates:
                break
            choices, weights = zip(*candidates.items())
            word = rng.choices(choices, weights=weights, k=1)[0]
            if word == EOS:
                break
            words.append(word)
        return words
