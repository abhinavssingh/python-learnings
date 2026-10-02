"""
Text cleaning / normalisation — the first step of every classic NLP pipeline.

Learn:
  * unicode + whitespace normalisation, URL / email / number removal
  * lowercasing and punctuation stripping
  * stopword removal and (Porter-lite) stemming
  * how cleaning changes the vocabulary of a real document (HR policy PDF)
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.datasets import hr_policy_text  # noqa: E402
from lib.utility.genai.foundations import STOPWORDS, TextCleaner  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.visualization import ngram_frequency_bar  # noqa: E402

RAW_SAMPLE = (
    "  Contact HR at hr.team@nestle.com or visit https://www.nestle.com/jobs  for the 2012 Policy!!\n"
    "Employees   RECEIVED  a salary increase of 5% — training & development are PRIORITIES.  "
)


def main():
    cleaner = TextCleaner()

    trace = cleaner.trace(RAW_SAMPLE, remove_stopwords=True, stem=True)
    tokens_plain = cleaner.clean(RAW_SAMPLE, remove_stopwords=False)
    tokens_nostop = cleaner.clean(RAW_SAMPLE, remove_stopwords=True)
    tokens_stem = cleaner.clean(RAW_SAMPLE, remove_stopwords=True, stem=True)

    stem_examples = ["employees", "employment", "training", "trained", "policies", "development", "rewarding"]

    corpus = hr_policy_text()
    raw_tokens = corpus.split()
    clean_tokens = cleaner.clean(corpus, remove_stopwords=False)
    content_tokens = cleaner.clean(corpus, remove_stopwords=True)
    stemmed_tokens = cleaner.clean(corpus, remove_stopwords=True, stem=True)

    vocab = {
        "raw whitespace tokens": (len(raw_tokens), len(set(raw_tokens))),
        "lowercase + punctuation removed": (len(clean_tokens), len(set(clean_tokens))),
        "+ stopwords removed": (len(content_tokens), len(set(content_tokens))),
        "+ stemming": (len(stemmed_tokens), len(set(stemmed_tokens))),
    }

    report = GenAIReport("Text Cleaning & Normalisation")
    report.grid([
        report.text("Raw input", RAW_SAMPLE),
        report.table("Step-by-step trace", [{"step": s, "text": t} for s, t in trace]),
    ])
    report.grid([
        report.text("Tokens (stopwords kept)", tokens_plain),
        report.text("Tokens (stopwords removed)", tokens_nostop),
        report.text("Tokens (stemmed)", tokens_stem),
        report.table("Stemming examples", [{"word": w, "stem": cleaner.stem(w)} for w in stem_examples]),
    ])
    report.full_table("Vocabulary shrinkage on the HR policy", [
        {"stage": k, "tokens": v[0], "unique tokens": v[1]} for k, v in vocab.items()
    ])
    report.plots([
        (ngram_frequency_bar(Counter(clean_tokens), title="Top words BEFORE stopword removal"), "Raw frequency"),
        (ngram_frequency_bar(Counter(content_tokens), title="Top words AFTER stopword removal"), "Content words"),
    ])
    report.full_text("Key takeaways", "\n".join([
        f"* Stopword list size: {len(STOPWORDS)} — function words dominate raw counts.",
        "* Cleaning is great for sparse methods (BoW, TF-IDF, BM25) and n-gram statistics.",
        "* Do NOT aggressively clean text before sending it to an LLM or a neural embedding model —",
        "  they were trained on natural text and use casing, punctuation and stopwords as signal.",
        "* Stemming merges 'employees' / 'employment' — increases recall, can hurt precision.",
    ]))
    report.save(__file__, "text_cleaning_report.html")


if __name__ == "__main__":
    main()
