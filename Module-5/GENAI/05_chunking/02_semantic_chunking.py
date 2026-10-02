"""
Semantic chunking — split where the MEANING changes, not at a fixed size.

Learn:
  * embed every sentence (with a small buffer of neighbours)
  * cosine distance between consecutive sentences
  * breakpoints where the distance exceeds a percentile threshold
  * effect of the percentile on the number and size of chunks
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.chunking import ChunkInspector, RecursiveCharacterChunker, SemanticChunker  # noqa: E402
from lib.utility.genai.datasets import hr_policy_text  # noqa: E402
from lib.utility.genai.factory import get_embedder  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.visualization import breakpoint_plot, chunk_size_histogram  # noqa: E402


def main():
    text = hr_policy_text()
    embedder = get_embedder()

    chunker = SemanticChunker(embedder, breakpoint_percentile=85, buffer_size=1)
    chunks = chunker.split_text(text)

    rows, sizes = [], {}
    for percentile in (70, 85, 95):
        result = SemanticChunker(embedder, breakpoint_percentile=percentile).split_text(text)
        rows.append({"percentile": percentile, **ChunkInspector.stats(result)})
        sizes[f"semantic p{percentile}"] = [len(c) for c in result]
    recursive = RecursiveCharacterChunker(600, 100).split_text(text)
    rows.append({"percentile": "recursive 600/100", **ChunkInspector.stats(recursive)})
    sizes["recursive 600"] = [len(c) for c in recursive]

    report = GenAIReport("Semantic chunking")
    report.grid([
        report.kv("Setup", {"sentences": len(chunker.last_sentences), "embedder": embedder.name,
                            "threshold (p85 distance)": round(float(chunker.last_threshold), 3), "chunks": len(chunks)}),
        report.table("Percentile vs chunk statistics", rows),
    ])
    report.full_plot(breakpoint_plot(chunker.last_distances, chunker.last_threshold),
                     "Peaks above the red line = topic shifts = chunk boundaries")
    report.full_plot(chunk_size_histogram(sizes), "Semantic vs recursive chunk sizes")
    report.full_table("Semantic chunks (p85)", ChunkInspector.to_frame(chunks, preview=140), rows=12)
    report.full_text("Key takeaways", "\n".join([
        "* Semantic chunks follow topic boundaries -> each chunk is about ONE thing (better embeddings).",
        "* Higher percentile = fewer breakpoints = bigger chunks.",
        "* Costs one embedding call per sentence at indexing time and gives uneven sizes (use max_chars).",
        "* Not always better than recursive splitting — compare with retrieval metrics (07_retrieval).",
    ]))
    report.save(__file__, "semantic_chunking_report.html")


if __name__ == "__main__":
    main()
