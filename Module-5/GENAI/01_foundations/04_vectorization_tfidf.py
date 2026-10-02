"""
Sparse text vectorization — Bag-of-Words and TF-IDF from scratch.

Learn:
  * document-term matrix (counts)
  * inverse document frequency: rare words matter more
  * TF-IDF implemented from scratch and verified against scikit-learn
  * n-gram features and cosine similarity between TF-IDF vectors
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from lib.utility.genai.datasets import SAMPLE_SENTENCES  # noqa: E402
from lib.utility.genai.foundations import (  # noqa: E402
    BagOfWordsVectorizer,
    TfidfVectorizerScratch,
    cosine_similarity_matrix,
    default_tokenizer,
)
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.visualization import similarity_heatmap  # noqa: E402


def main():
    texts = [s for s, _ in SAMPLE_SENTENCES]
    labels = [f"{lab}: {s[:30]}" for s, lab in SAMPLE_SENTENCES]

    bow = BagOfWordsVectorizer()
    counts = bow.fit_transform(texts[:4])
    bow_df = pd.DataFrame(counts, columns=bow.get_feature_names(), index=[f"doc{i}" for i in range(4)])
    bow_df = bow_df.iloc[:, :14]

    tfidf = TfidfVectorizerScratch().fit(texts)
    matrix = tfidf.transform(texts)

    sk = TfidfVectorizer(tokenizer=default_tokenizer, lowercase=False, token_pattern=None).fit(texts)
    sk_df = pd.DataFrame(sk.transform(texts).toarray(), columns=sk.get_feature_names_out())
    ours_df = pd.DataFrame(matrix, columns=tfidf.get_feature_names())
    common = [c for c in ours_df.columns if c in sk_df.columns]
    max_diff = float(np.abs(ours_df[common].values - sk_df[common].values).max())

    idf = tfidf.idf_table()
    bigram_tfidf = TfidfVectorizerScratch(ngram_range=(1, 2)).fit(texts)

    report = GenAIReport("Bag-of-Words & TF-IDF (from scratch)")
    report.full_table("Bag-of-Words document-term matrix (first 4 docs, first 14 terms)", bow_df.reset_index())
    report.grid([
        report.table("Highest IDF (rare, informative)", [{"term": t, "idf": round(v, 3)} for t, v in idf[:10]]),
        report.table("Lowest IDF (common)", [{"term": t, "idf": round(v, 3)} for t, v in idf[-10:]]),
    ])
    report.grid([
        report.table("Top TF-IDF terms per document", [
            {"doc": texts[i][:45], "top terms": ", ".join(f"{t} ({w:.2f})" for t, w in tfidf.top_terms(matrix[i], 4))}
            for i in range(len(texts))
        ], rows=12),
        report.kv("Scratch vs scikit-learn", {"shared features": len(common), "max |difference|": f"{max_diff:.2e}",
                                              "unigram features": matrix.shape[1],
                                              "uni+bigram features": bigram_tfidf.transform(texts).shape[1]}),
    ])
    report.full_plot(similarity_heatmap(cosine_similarity_matrix(matrix), labels, "TF-IDF cosine similarity"),
                     "Lexical similarity only: paraphrases with different words score ~0")
    report.full_text("Key takeaways", "\n".join([
        "* TF-IDF = term frequency x log inverse document frequency, then L2-normalised.",
        "* It captures WHICH words appear, not what they MEAN: 'salary increase' vs 'compensation raised' look unrelated.",
        "* Still a strong, cheap baseline for keyword search — BM25 (next script) is its retrieval-tuned cousin.",
    ]))
    report.save(__file__, "vectorization_tfidf_report.html")


if __name__ == "__main__":
    main()
