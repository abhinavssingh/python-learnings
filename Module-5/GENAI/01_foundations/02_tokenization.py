"""
Tokenization — how text becomes the units a model actually sees.

Learn:
  * whitespace vs regex-word vs character vs sentence tokenizers
  * Byte-Pair Encoding (BPE) trained from scratch: merges, vocabulary, encode / decode
  * why sub-word tokenizers handle unseen words (no OOV problem)
  * token counts drive LLM cost & context-window usage
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.datasets import hr_policy_text  # noqa: E402
from lib.utility.genai.foundations import (  # noqa: E402
    BPETokenizer,
    CharacterTokenizer,
    RegexWordTokenizer,
    SentenceTokenizer,
    WhitespaceTokenizer,
)
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.visualization import line_plot  # noqa: E402

SAMPLE = "Nestlé's HR policy isn't complex: managers coach employees. Promotions reward sustained performance!"


def main():
    corpus = hr_policy_text()
    tokenizers = {
        "whitespace": WhitespaceTokenizer(),
        "regex word": RegexWordTokenizer(lowercase=True),
        "character": CharacterTokenizer(),
        "sentence": SentenceTokenizer(),
    }

    sample_rows = [{"tokenizer": n, "count": len(t.tokenize(SAMPLE)), "tokens": " | ".join(t.tokenize(SAMPLE)[:25])}
                   for n, t in tokenizers.items()]
    corpus_rows = []
    for name, tok in tokenizers.items():
        tokens = tok.tokenize(corpus)
        corpus_rows.append({"tokenizer": name, "tokens": len(tokens), "vocabulary": len(set(tokens))})

    bpe = BPETokenizer(num_merges=300).train(corpus)
    bpe_tokens = bpe.tokenize(corpus)
    corpus_rows.append({"tokenizer": "BPE (300 merges)", "tokens": len(bpe_tokens), "vocabulary": len(bpe.vocab)})

    word_vocab = set(tokenizers["regex word"].tokenize(corpus))
    unseen = ["unbelievable", "reorganisation", "nestléverse", "coaching"]
    oov_rows = [{"word": w, "in word vocab": w in word_vocab, "BPE pieces": " ".join(bpe.tokenize_word(w))} for w in unseen]

    ids = bpe.encode(SAMPLE)
    roundtrip = bpe.decode(bpe.tokenize(SAMPLE))

    report = GenAIReport("Tokenization: from words to sub-words (BPE)")
    report.full_table("Tokenizers on one sentence", sample_rows)
    report.grid([
        report.table("Tokenizers on the full HR policy", corpus_rows),
        report.table("BPE learned merges (first 20)", bpe.merge_history[:20]),
    ])
    report.grid([
        report.text("BPE tokens for the sample", " | ".join(bpe.tokenize(SAMPLE))),
        report.kv("Encode / decode", {"token ids (first 30)": ids[:30], "decoded": roundtrip,
                                      "vocab size": len(bpe.vocab), "merges": len(bpe.merges)}),
    ])
    report.full_table("Out-of-vocabulary words: word tokenizer vs BPE", oov_rows)
    report.full_plot(line_plot([m["frequency"] for m in bpe.merge_history],
                               "Pair frequency of each BPE merge", "merge step", "pair frequency"),
                     "Merges get rarer as the vocabulary grows")
    report.full_text("Key takeaways", "\n".join([
        "* Word tokenizers explode the vocabulary and cannot represent unseen words.",
        "* Character tokenizers never go OOV but produce very long sequences.",
        "* BPE is the middle ground used by GPT / Llama / Qwen: frequent words become one token,",
        "  rare words are split into reusable pieces.",
        "* Rule of thumb for English: ~0.75 words per token -> always budget context in TOKENS.",
    ]))
    report.save(__file__, "tokenization_report.html")


if __name__ == "__main__":
    main()
