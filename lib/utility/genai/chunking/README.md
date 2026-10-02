# Chunking

Split documents into smaller pieces for indexing, retrieval, or model context
windows. Chunkers implement the shared `BaseChunker` interface and preserve
document metadata where appropriate.

Available strategies include fixed-size, recursive-character, sentence,
paragraph, semantic, Markdown-header, and Python-code chunking. `ChunkerRegistry`
provides a central place to look up chunker implementations, and `ChunkInspector`
supports inspecting chunk output.

```python
from lib.utility.genai.chunking import RecursiveCharacterChunker
```

See `Module-5/GENAI/05_chunking/` for strategy comparisons and examples.
