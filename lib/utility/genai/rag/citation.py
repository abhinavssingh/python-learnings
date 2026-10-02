import re

from ..abstractions.document import SearchResult


def format_context(results: list[SearchResult], max_chars: int | None = None) -> str:
    """Number each chunk so the LLM can cite it: [1] (source p.4) text..."""
    blocks = []
    for i, result in enumerate(results, start=1):
        meta = result.metadata
        location = meta.get("source", "doc")
        if "page" in meta:
            location += f" p.{meta['page']}"
        text = " ".join(result.text.split())
        if max_chars:
            text = text[:max_chars]
        blocks.append(f"[{i}] ({location}) {text}")
    return "\n\n".join(blocks)


def extract_citations(answer: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


def citation_table(answer: str, results: list[SearchResult]) -> list[dict]:
    cited = set(extract_citations(answer))
    return [
        {
            "ref": f"[{i}]",
            "cited": "yes" if i in cited else "",
            "source": r.metadata.get("source", ""),
            "page": r.metadata.get("page", ""),
            "score": round(r.score, 4),
            "preview": " ".join(r.text.split())[:110],
        }
        for i, r in enumerate(results, start=1)
    ]
