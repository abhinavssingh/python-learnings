"""
Word2Vec (skip-gram with negative sampling) implemented in NumPy.

Learn:
  * the distributional hypothesis: words in similar contexts get similar vectors
  * skip-gram training pairs, negative sampling and the training loss
  * nearest neighbours in embedding space and 2-D projections
  * why tiny corpora give noisy embeddings (motivation for pre-trained models)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.datasets import hr_policy_text  # noqa: E402
from lib.utility.genai.embeddings import Word2VecScratch, project_2d  # noqa: E402
from lib.utility.genai.foundations import SentenceTokenizer, TextCleaner  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.visualization import embedding_scatter, line_plot  # noqa: E402

PROBE_WORDS = ["employees", "managers", "performance", "training", "rewards", "development"]


def main():
    cleaner = TextCleaner()
    sentences = [cleaner.clean(s) for s in SentenceTokenizer().tokenize(hr_policy_text())]
    sentences = [s for s in sentences if len(s) > 2]

    model = Word2VecScratch(embedding_dim=40, window=3, negative=5, min_count=2, epochs=40).fit(sentences)

    neighbour_rows = []
    for word in PROBE_WORDS:
        if word in model.word_to_id:
            neighbour_rows.append({"word": word, "most similar": ", ".join(f"{w} ({s:.2f})" for w, s in model.most_similar(word, 5))})

    top_words = model.id_to_word[:45]
    points = project_2d(model.vectors[: len(top_words)], method="pca")

    report = GenAIReport("Word2Vec from scratch (skip-gram + negative sampling)")
    report.grid([
        report.kv("Training setup", {"sentences": len(sentences), "vocabulary (min_count=2)": len(model.id_to_word),
                                     "embedding dim": model.embedding_dim, "window": model.window,
                                     "negative samples": model.negative, "epochs": model.epochs}),
        report.table("Nearest neighbours (cosine)", neighbour_rows),
    ])
    report.plots([
        (line_plot(model.loss_history, "Average negative-sampling loss per epoch", "epoch", "loss"), "Training loss"),
        (embedding_scatter(points, top_words, title="45 most frequent words (PCA)"), "Embedding space"),
    ])
    report.full_text("Key takeaways", "\n".join([
        "* Each word gets a dense vector learned by predicting its context words (skip-gram).",
        "* Negative sampling turns a huge softmax into a few binary classifications -> fast training.",
        "* With ~1.5k words of text the neighbours are noisy: real Word2Vec/GloVe train on billions of tokens.",
        "* Static embeddings give ONE vector per word ('bank' river vs money). Transformer embeddings",
        "  (next script, via Ollama) are contextual and embed whole sentences.",
    ]))
    report.save(__file__, "word2vec_scratch_report.html")


if __name__ == "__main__":
    main()
