# Chunking Documents

Chunking divides long documents into smaller passages that can be embedded and retrieved. The goal is not simply to make every piece the same size: boundaries should preserve enough context for a later question to be answered accurately.

## Lessons

1. `01_chunking_strategies.py` compares fixed-size, recursive, sentence, paragraph, Markdown, and code-aware splitting. For a policy with a heading followed by several rules, compare a fixed character split with a Markdown-aware split: the former may separate a rule from its heading, while the latter can preserve the section structure.
2. `02_semantic_chunking.py` uses embedding similarity to find topic changes and create semantic breakpoints. Imagine a passage moving from parental leave to expense reimbursement; a semantic boundary may be more useful than cutting at an arbitrary character count.

## Concepts to notice

- Chunk size trades context against retrieval precision and embedding cost.
- Overlap can preserve a sentence split across a boundary, but excessive overlap duplicates content and may crowd retrieval results.
- Metadata such as source and page should travel with each chunk so answers can be traced back to their origin.
- Semantic chunking requires an embedding model; the first lesson is suitable for quick local experiments without one.

## Run

From the repository root:

```powershell
python Module-5/GENAI/05_chunking/01_chunking_strategies.py
python Module-5/GENAI/05_chunking/02_semantic_chunking.py
```

The semantic lesson uses the configured local embedding model. See the [GenAI setup guide](../README.md) for Ollama prerequisites. Reports are saved in this folder's `reports/` directory.

## Practice

Choose a short document with headings and multi-sentence paragraphs. Run multiple splitting strategies, then inspect the resulting chunks manually. Test each chunk by asking: could someone understand it without the previous paragraph, and does it contain more unrelated information than necessary?
