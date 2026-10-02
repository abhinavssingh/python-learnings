"""
Chunking strategies — how documents are split before embedding.

Learn:
  * fixed-size (characters / words) with overlap
  * recursive character splitting (LangChain's default idea)
  * sentence- and paragraph-based chunking
  * structure-aware chunking: Markdown headers and Python code (AST)
  * measuring chunk quality: size distribution, overlap, clean sentence endings
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

import pandas as pd  # noqa: E402

from lib.utility.genai.abstractions import Document  # noqa: E402
from lib.utility.genai.chunking import ChunkerRegistry, ChunkInspector, MarkdownHeaderChunker, PythonCodeChunker  # noqa: E402
from lib.utility.genai.datasets import load_hr_policy  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.visualization import chunk_size_histogram  # noqa: E402

STRATEGIES = {
    "fixed_char": {"chunk_size": 500, "chunk_overlap": 50},
    "fixed_word": {"chunk_size": 80, "chunk_overlap": 10},
    "recursive": {"chunk_size": 500, "chunk_overlap": 80},
    "sentence": {"max_chars": 500, "overlap_sentences": 1},
    "paragraph": {"max_chars": 800},
}

MARKDOWN = """# Employee Handbook
Welcome to the company.
## Leave
### Annual leave
Employees get 24 days of paid leave per year.
### Sick leave
Up to 10 days of paid sick leave with a medical certificate.
## Pay
Salaries are reviewed every April based on performance.
"""


def main():
    pages = load_hr_policy()

    stat_rows, sizes, previews = [], {}, {}
    for name, params in STRATEGIES.items():
        chunks = ChunkerRegistry.create(name, **params).split_documents(pages)
        stat_rows.append({"strategy": name, "params": str(params), **ChunkInspector.stats(chunks)})
        sizes[name] = [len(c.page_content) for c in chunks]
        previews[name] = ChunkInspector.to_frame(chunks)

    md_chunks = MarkdownHeaderChunker().split_documents([Document(MARKDOWN, {"source": "handbook.md"})])
    code_file = ROOT / "lib" / "utility" / "genai" / "foundations" / "similarity.py"
    code_blocks = PythonCodeChunker().blocks(code_file.read_text(encoding="utf-8"))

    fixed = ChunkerRegistry.create("fixed_char", chunk_size=120, chunk_overlap=30).split_text(pages[0].page_content[:400])

    report = GenAIReport("Chunking strategies")
    report.full_table("Strategy comparison on the HR policy", pd.DataFrame(stat_rows))
    report.full_plot(chunk_size_histogram(sizes), "Chunk size distribution per strategy")
    report.full_text("Fixed-size chunks (120 chars, 30 overlap) — note words cut in half",
                     "\n---\n".join(fixed))
    report.grid([report.table(f"{name} chunks", previews[name], rows=8) for name in ("recursive", "sentence")])
    report.grid([
        report.table("Markdown header chunker (headers become metadata)", [
            {**{k: v for k, v in c.metadata.items() if k.startswith("h")}, "text": c.page_content} for c in md_chunks
        ]),
        report.table(f"Python code chunker ({code_file.name})", [
            {"block": name, "lines": code.count("\n") + 1} for name, code in code_blocks
        ]),
    ])
    report.full_text("Key takeaways", "\n".join([
        "* Chunk size is a trade-off: small chunks = precise retrieval but little context; large = the opposite.",
        "* Overlap preserves information that straddles a boundary (at the cost of duplicate tokens).",
        "* Recursive splitting keeps paragraphs/sentences intact when possible — a great default.",
        "* Use structure when you have it: headers for Markdown/HTML, functions/classes for code.",
        "* Carry metadata (source, page, headers) into every chunk — needed for citations and filters.",
    ]))
    report.save(__file__, "chunking_strategies_report.html")


if __name__ == "__main__":
    main()
