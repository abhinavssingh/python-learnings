import math
import re
from collections import Counter

from ..foundations.ngrams import generate_ngrams
from ..foundations.text_cleaning import STOPWORDS


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _content_tokens(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in STOPWORDS]


def exact_match(prediction: str, reference: str) -> float:
    return float(" ".join(_tokens(prediction)) == " ".join(_tokens(reference)))


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token overlap F1 (content words only)."""
    pred, ref = _content_tokens(prediction), _content_tokens(reference)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if not pred or not ref or overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-score based on the longest common subsequence."""
    a, b = _tokens(prediction), _tokens(reference)
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """Sentence BLEU with add-one smoothing and brevity penalty."""
    pred, ref = _tokens(prediction), _tokens(reference)
    if not pred:
        return 0.0
    log_precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams, ref_ngrams = Counter(generate_ngrams(pred, n)), Counter(generate_ngrams(ref, n))
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = max(sum(pred_ngrams.values()), 0)
        log_precisions.append(math.log((overlap + 1) / (total + 1)))
    brevity = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / len(pred))
    return brevity * math.exp(sum(log_precisions) / max_n)


def keyword_recall(prediction: str, keywords: list[str]) -> float:
    """Fraction of expected keywords that appear in the answer."""
    if not keywords:
        return 0.0
    text = prediction.lower()
    return sum(k.lower() in text for k in keywords) / len(keywords)


def generation_scores(prediction: str, reference: str = "", keywords: list[str] | None = None) -> dict[str, float]:
    scores = {"answer_words": len(prediction.split())}
    if reference:
        scores.update({"token_f1": token_f1(prediction, reference), "rouge_l": rouge_l(prediction, reference),
                       "bleu": bleu(prediction, reference)})
    if keywords:
        scores["keyword_recall"] = keyword_recall(prediction, keywords)
    return {k: round(v, 3) for k, v in scores.items()}
