"""
N-grams and statistical language models — the ancestors of LLMs.

Learn:
  * unigrams / bigrams / trigrams and their frequencies
  * P(word | context) with Laplace and add-k smoothing
  * perplexity on held-out text (lower = better model)
  * next-word prediction and text generation from an n-gram model
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

import pandas as pd  # noqa: E402

from lib.utility.genai.datasets import load_hr_policy  # noqa: E402
from lib.utility.genai.foundations import (  # noqa: E402
    NGramLanguageModel,
    RegexWordTokenizer,
    SentenceTokenizer,
    generate_ngrams,
    ngram_counts,
)
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.visualization import metric_bar, ngram_frequency_bar  # noqa: E402


def sentences_of(docs):
    splitter, words = SentenceTokenizer(), RegexWordTokenizer(lowercase=True)
    sentences = []
    for doc in docs:
        for sentence in splitter.tokenize(doc.page_content):
            tokens = words.tokenize(sentence)
            if len(tokens) > 2:
                sentences.append(tokens)
    return sentences


def main():
    pages = load_hr_policy()
    train_sents = sentences_of([p for p in pages if p.metadata["page"] < 7])
    test_sents = sentences_of([p for p in pages if p.metadata["page"] == 7])
    all_tokens = [t for s in train_sents + test_sents for t in s]

    example = "promotions are based on sustained performance".split()

    rows = []
    for n in (1, 2, 3):
        for smoothing, k in (("laplace", 1.0), ("add_k", 0.1), ("add_k", 0.01)):
            model = NGramLanguageModel(n=n, smoothing=smoothing, k=k).fit(train_sents)
            label = f"{n}-gram {smoothing}" + (f" k={k}" if smoothing == "add_k" else "")
            rows.append({"model": label, "n": n, "train perplexity": round(model.perplexity(train_sents), 1),
                         "test perplexity": round(model.perplexity(test_sents), 1)})

    best = min(rows, key=lambda r: r["test perplexity"])
    bigram = NGramLanguageModel(n=2, smoothing="add_k", k=0.01).fit(train_sents)
    trigram = NGramLanguageModel(n=3, smoothing="add_k", k=0.01).fit(train_sents)

    report = GenAIReport("N-grams & Statistical Language Models")
    report.grid([
        report.kv("Corpus", {"train sentences (pages 3-6)": len(train_sents), "test sentences (page 7)": len(test_sents),
                             "tokens": len(all_tokens), "vocabulary": bigram.vocab_size}),
        report.table("n-grams of an example sentence", [
            {"n": n, "n-grams": " | ".join(" ".join(g) for g in generate_ngrams(example, n))} for n in (1, 2, 3)
        ]),
    ])
    report.plots([
        (ngram_frequency_bar(ngram_counts(all_tokens, 2), title="Top bigrams"), "Bigrams"),
        (ngram_frequency_bar(ngram_counts(all_tokens, 3), title="Top trigrams"), "Trigrams"),
    ])
    report.full_table("Perplexity by model order and smoothing", rows)
    report.full_plot(metric_bar(pd.DataFrame(rows), "model", ["train perplexity", "test perplexity"],
                                "Train vs test perplexity (a big gap = overfitting)"), "Perplexity")
    report.grid([
        report.table("Bigram: P(next | 'total')",
                     [{"word": w, "p": round(p, 4)} for w, p in bigram.next_word_distribution(["total"])]),
        report.table("Trigram: P(next | 'line managers')",
                     [{"word": w, "p": round(p, 4)} for w, p in trigram.next_word_distribution(["line", "managers"])]),
    ])
    report.grid([
        report.text("Bigram generation (seed: 'employees')", " ".join(bigram.generate(["employees"], max_words=25))),
        report.text("Trigram generation (seed: 'line managers')",
                    " ".join(trigram.generate(["line", "managers"], max_words=25))),
    ])
    report.full_text("Key takeaways", "\n".join([
        f"* Best held-out model: {best['model']} (perplexity {best['test perplexity']}).",
        "* Higher n memorises the training text (low train perplexity) but suffers from sparsity on new text.",
        "* Smoothing reserves probability mass for unseen n-grams; smaller k usually fits better on small corpora.",
        "* LLMs solve the same task — predict the next token — with neural networks and very long contexts.",
    ]))
    report.save(__file__, "ngrams_report.html")


if __name__ == "__main__":
    main()
