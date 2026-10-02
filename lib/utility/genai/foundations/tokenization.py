import re
from collections import Counter


class WhitespaceTokenizer:
    name = "whitespace"

    def tokenize(self, text: str) -> list[str]:
        return text.split()


class RegexWordTokenizer:
    """Words and individual punctuation marks: "don't!" -> ["don", "'", "t", "!"]."""

    name = "regex_word"
    PATTERN = re.compile(r"\w+|[^\w\s]")

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase

    def tokenize(self, text: str) -> list[str]:
        tokens = self.PATTERN.findall(text)
        return [t.lower() for t in tokens] if self.lowercase else tokens


class CharacterTokenizer:
    name = "character"

    def tokenize(self, text: str) -> list[str]:
        return list(text)


class SentenceTokenizer:
    """Rule-based sentence splitter (handles common abbreviations)."""

    name = "sentence"
    ABBREVIATIONS = {"e.g", "i.e", "etc", "mr", "mrs", "dr", "vs", "s.a", "ltd", "no"}

    def tokenize(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“(])", text)
        sentences: list[str] = []
        for piece in candidates:
            if sentences:
                last_word = sentences[-1].rstrip(".!?").split(" ")[-1].lower()
                if last_word in self.ABBREVIATIONS:
                    sentences[-1] = f"{sentences[-1]} {piece}"
                    continue
            sentences.append(piece)
        return sentences


class BPETokenizer:
    """
    Byte-Pair Encoding trained from scratch (the idea behind GPT / Qwen tokenizers).

    1. Start with characters (+ end-of-word marker "</w>").
    2. Count adjacent symbol pairs across the corpus.
    3. Merge the most frequent pair into a new symbol.
    4. Repeat `num_merges` times.
    """

    name = "bpe"
    END = "</w>"

    def __init__(self, num_merges: int = 200, lowercase: bool = True):
        self.num_merges = num_merges
        self.lowercase = lowercase
        self.merges: list[tuple[str, str]] = []
        self.vocab: set[str] = set()
        self.merge_history: list[dict] = []

    def _words(self, text: str) -> list[str]:
        text = text.lower() if self.lowercase else text
        return re.findall(r"\w+|[^\w\s]", text)

    def train(self, corpus: str | list[str]) -> "BPETokenizer":
        text = corpus if isinstance(corpus, str) else " ".join(corpus)
        word_freq = Counter(self._words(text))
        splits = {word: list(word) + [self.END] for word in word_freq}
        self.vocab = {ch for symbols in splits.values() for ch in symbols}

        for step in range(self.num_merges):
            pair_counts: Counter = Counter()
            for word, freq in word_freq.items():
                symbols = splits[word]
                for a, b in zip(symbols, symbols[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            best, count = pair_counts.most_common(1)[0]
            if count < 2:
                break
            self.merges.append(best)
            merged = best[0] + best[1]
            self.vocab.add(merged)
            self.merge_history.append({"step": step + 1, "pair": f"{best[0]} + {best[1]}", "new_token": merged, "frequency": count})
            for word in splits:
                splits[word] = self._apply_merge(splits[word], best)
        return self

    @staticmethod
    def _apply_merge(symbols: list[str], pair: tuple[str, str]) -> list[str]:
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out

    def tokenize_word(self, word: str) -> list[str]:
        symbols = list(word) + [self.END]
        for pair in self.merges:
            symbols = self._apply_merge(symbols, pair)
        return symbols

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for word in self._words(text):
            tokens.extend(self.tokenize_word(word))
        return tokens

    def encode(self, text: str) -> list[int]:
        index = {tok: i for i, tok in enumerate(sorted(self.vocab))}
        return [index.get(tok, -1) for tok in self.tokenize(text)]

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens).replace(self.END, " ").strip()
