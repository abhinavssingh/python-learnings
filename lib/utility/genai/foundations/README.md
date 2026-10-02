# Foundations

Small, framework-independent implementations of core NLP and transformer
concepts. These are useful for learning and lightweight baselines.

The package exports text cleaning and tokenizers, BPE, n-gram language models,
bag-of-words and TF-IDF vectorizers, similarity functions, BM25, and attention
helpers including causal masks and positional encodings.

```python
from lib.utility.genai.foundations import NGramLanguageModel, TextCleaner
```

See `Module-5/GENAI/01_foundations/` and
`Module-5/GENAI/12_transformers/01_attention_from_scratch.py` for guided
examples.
