import re
import string
import unicodedata

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she should
so some such than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves also may must shall within without whilst upon via per
""".split())


class TextCleaner:
    """
    Classical text pre-processing pipeline.

    Modern LLM pipelines usually keep text almost raw, but these
    steps are still essential for n-grams, TF-IDF and BM25.
    """

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
    NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")

    def __init__(self, stopwords: frozenset[str] = STOPWORDS):
        self.stopwords = stopwords

    # ------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------

    @staticmethod
    def normalize_unicode(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    @staticmethod
    def lowercase(text: str) -> str:
        return text.lower()

    def remove_urls(self, text: str) -> str:
        return self.URL_PATTERN.sub(" ", text)

    def remove_emails(self, text: str) -> str:
        return self.EMAIL_PATTERN.sub(" ", text)

    def remove_numbers(self, text: str) -> str:
        return self.NUMBER_PATTERN.sub(" ", text)

    @staticmethod
    def remove_punctuation(text: str) -> str:
        return text.translate(str.maketrans({ch: " " for ch in string.punctuation + "’‘“”–—°©"}))

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t.lower() not in self.stopwords]

    @staticmethod
    def stem(word: str) -> str:
        """
        A tiny suffix-stripping stemmer (Porter-inspired).

        running -> run, policies -> polici, employees -> employe
        """
        w = word.lower()
        if len(w) <= 3:
            return w
        for suffix, replacement in (
            ("ational", "ate"), ("ization", "ize"), ("fulness", "ful"),
            ("iveness", "ive"), ("ments", ""), ("ment", ""), ("ies", "i"),
            ("sses", "ss"), ("ness", ""), ("ing", ""), ("edly", ""), ("ed", ""),
            ("ly", ""), ("es", "e"), ("s", ""),
        ):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                w = w[: -len(suffix)] + replacement
                break
        if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":
            w = w[:-1]
        return w

    # ------------------------------------------------------
    # Pipelines
    # ------------------------------------------------------

    def clean(self, text: str, remove_stopwords: bool = True, stem: bool = False) -> list[str]:
        """Full pipeline returning clean tokens."""
        return self.trace(text, remove_stopwords=remove_stopwords, stem=stem)[-1][1].split()

    def clean_text(self, text: str, remove_stopwords: bool = True, stem: bool = False) -> str:
        return " ".join(self.clean(text, remove_stopwords=remove_stopwords, stem=stem))

    def trace(self, text: str, remove_stopwords: bool = True, stem: bool = False) -> list[tuple[str, str]]:
        """Run every step and return (step_name, output) pairs for learning."""
        steps: list[tuple[str, str]] = [("original", text)]
        current = text
        for name, fn in (
            ("normalize_unicode", self.normalize_unicode),
            ("lowercase", self.lowercase),
            ("remove_urls", self.remove_urls),
            ("remove_emails", self.remove_emails),
            ("remove_numbers", self.remove_numbers),
            ("remove_punctuation", self.remove_punctuation),
            ("normalize_whitespace", self.normalize_whitespace),
        ):
            current = fn(current)
            steps.append((name, current))

        tokens = current.split()
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
            steps.append(("remove_stopwords", " ".join(tokens)))
        if stem:
            tokens = [self.stem(t) for t in tokens]
            steps.append(("stem", " ".join(tokens)))
        return steps
