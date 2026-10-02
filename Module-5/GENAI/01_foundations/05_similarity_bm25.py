"""
Similarity measures and lexical ranking (BM25).

Learn:
  * cosine vs dot product vs euclidean vs manhattan — and when they agree
  * jaccard similarity on token sets
  * BM25: the classic keyword ranking function behind search engines
  * BM25 vs TF-IDF ranking on HR policy chunks
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

import numpy as np  # noqa: E402

from lib.utility.genai.datasets import hr_policy_chunks  # noqa: E402
from lib.utility.genai.foundations import (  # noqa: E402
    BM25,
    TextCleaner,
    TfidfVectorizerScratch,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    jaccard_similarity,
    l2_normalize,
    manhattan_distance,
)
from lib.utility.genai.reports import GenAIReport  # noqa: E402

QUERIES = ["what are promotions based on", "collective bargaining rights", "harassment discrimination tolerance"]


def main():
    a, b, c = np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]), np.array([3.0, -1.0, 0.5])
    metric_rows = []
    for name, (x, y) in {"a vs b (same direction, 2x length)": (a, b), "a vs c (different direction)": (a, c)}.items():
        nx, ny = l2_normalize(np.vstack([x, y]))
        metric_rows.append({"pair": name, "cosine": round(cosine_similarity(x, y), 3), "dot": round(dot_product(x, y), 3),
                            "euclidean": round(euclidean_distance(x, y), 3), "manhattan": round(manhattan_distance(x, y), 3),
                            "dot (L2-normalised)": round(dot_product(nx, ny), 3)})

    s1, s2 = "line managers coach employees", "employees are coached by line managers"
    jac = jaccard_similarity(set(s1.split()), set(s2.split()))

    chunks = hr_policy_chunks(400, 50)
    cleaner = TextCleaner()
    bm25 = BM25().fit([cleaner.clean(c.page_content, stem=True) for c in chunks])
    tfidf = TfidfVectorizerScratch().fit([c.page_content for c in chunks])
    chunk_matrix = tfidf.transform([c.page_content for c in chunks])

    ranking_rows = []
    for q in QUERIES:
        bm_top = bm25.top_n(cleaner.clean(q, stem=True), 3)
        tf_scores = chunk_matrix @ tfidf.transform([q])[0]
        tf_top = np.argsort(-tf_scores)[:3]
        for rank in range(3):
            bi, bs = bm_top[rank]
            ti = int(tf_top[rank])
            ranking_rows.append({"query": q, "rank": rank + 1,
                                 "BM25 (page, score)": f"p{chunks[bi].metadata['page']} ({bs:.2f})",
                                 "BM25 chunk": chunks[bi].preview(80),
                                 "TF-IDF (page, score)": f"p{chunks[ti].metadata['page']} ({tf_scores[ti]:.2f})"})

    report = GenAIReport("Similarity Measures & BM25")
    report.full_table("Vector similarity / distance metrics", metric_rows)
    report.grid([
        report.kv("Jaccard similarity", {"s1": s1, "s2": s2, "jaccard(tokens)": round(jac, 3)}),
        report.kv("BM25 parameters", {"k1 (term-frequency saturation)": bm25.k1, "b (length normalisation)": bm25.b,
                                      "chunks indexed": len(chunks), "avg chunk length (tokens)": round(bm25.avgdl, 1)}),
    ])
    report.full_table("BM25 vs TF-IDF: top-3 chunks per query", ranking_rows, rows=12)
    report.full_text("Key takeaways", "\n".join([
        "* Cosine ignores vector length; dot product rewards it. After L2 normalisation they are identical.",
        "* Most embedding models are compared with cosine (or dot on normalised vectors).",
        "* BM25 saturates repeated terms (k1) and penalises long documents (b) — better than raw TF-IDF for search.",
        "* BM25 is still used in production RAG as the keyword half of HYBRID retrieval (see 07_retrieval).",
    ]))
    report.save(__file__, "similarity_measures_report.html")


if __name__ == "__main__":
    main()
