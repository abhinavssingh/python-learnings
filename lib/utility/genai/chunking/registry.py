from typing import Callable

from ..abstractions.base_chunker import BaseChunker
from .fixed_size import FixedSizeChunker
from .recursive import RecursiveCharacterChunker
from .semantic import SemanticChunker
from .sentence import ParagraphChunker, SentenceChunker
from .structure_aware import MarkdownHeaderChunker, PythonCodeChunker


class ChunkerRegistry:
    """Create chunkers by name so strategies can be compared in a loop."""

    _registry: dict[str, Callable[..., BaseChunker]] = {
        "fixed_char": lambda **kw: FixedSizeChunker(unit="char", **kw),
        "fixed_word": lambda **kw: FixedSizeChunker(unit="word", **kw),
        "recursive": RecursiveCharacterChunker,
        "sentence": SentenceChunker,
        "paragraph": ParagraphChunker,
        "semantic": SemanticChunker,
        "markdown_header": MarkdownHeaderChunker,
        "python_code": PythonCodeChunker,
    }

    @classmethod
    def register(cls, name: str, factory: Callable[..., BaseChunker]) -> None:
        cls._registry[name] = factory

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseChunker:
        if name not in cls._registry:
            raise KeyError(f"Unknown chunker '{name}'. Available: {cls.names()}")
        return cls._registry[name](**kwargs)

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._registry)
