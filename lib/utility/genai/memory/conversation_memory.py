from abc import ABC, abstractmethod

import numpy as np

from ..abstractions.base_embedder import BaseEmbedder
from ..abstractions.base_llm import BaseLLM, Message
from ..prompting.templates import PromptLibrary


class ConversationMemory(ABC):
    """Stores chat turns and decides what history is sent back to the LLM."""

    @abstractmethod
    def add(self, role: str, content: str) -> None:
        pass

    @abstractmethod
    def messages(self, query: str | None = None) -> list[Message]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    def add_exchange(self, user: str, assistant: str) -> None:
        self.add("user", user)
        self.add("assistant", assistant)

    def as_text(self, query: str | None = None) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages(query))


class BufferMemory(ConversationMemory):
    """Keep everything (simple, but the prompt grows forever)."""

    def __init__(self):
        self._messages: list[Message] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def messages(self, query: str | None = None) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


class WindowMemory(BufferMemory):
    """Keep only the last `k` exchanges (2k messages)."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

    def messages(self, query: str | None = None) -> list[Message]:
        return self._messages[-2 * self.k:]


class SummaryMemory(ConversationMemory):
    """
    Keep the last `keep_last` messages verbatim; older ones are folded
    into a running LLM-written summary.
    """

    def __init__(self, llm: BaseLLM, keep_last: int = 4):
        self.llm = llm
        self.keep_last = keep_last
        self.summary = ""
        self._messages: list[Message] = []
        self.prompt = PromptLibrary.get("summarize_memory")

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.keep_last:
            overflow = self._messages[: -self.keep_last]
            self._messages = self._messages[-self.keep_last:]
            new_lines = "\n".join(f"{m['role']}: {m['content']}" for m in overflow)
            self.summary = self.llm.generate(
                self.prompt.format(summary=self.summary or "(empty)", new_lines=new_lines), max_tokens=150
            ).text

    def messages(self, query: str | None = None) -> list[Message]:
        prefix = [{"role": "system", "content": f"Summary of earlier conversation: {self.summary}"}] if self.summary else []
        return prefix + list(self._messages)

    def clear(self) -> None:
        self.summary = ""
        self._messages.clear()


class VectorMemory(ConversationMemory):
    """Long-term memory: embed every exchange and recall the most similar ones to the new query."""

    def __init__(self, embedder: BaseEmbedder, k: int = 2):
        self.embedder = embedder
        self.k = k
        self.exchanges: list[str] = []
        self.vectors: list[np.ndarray] = []
        self._pending_user: str | None = None

    def add(self, role: str, content: str) -> None:
        if role == "user":
            self._pending_user = content
            return
        text = f"user: {self._pending_user or ''}\nassistant: {content}"
        self.exchanges.append(text)
        self.vectors.append(self.embedder.embed_documents([text])[0])
        self._pending_user = None

    def recall(self, query: str) -> list[tuple[str, float]]:
        if not self.vectors:
            return []
        matrix = np.vstack(self.vectors)
        q = self.embedder.embed_query(query)
        sims = matrix @ q / (np.linalg.norm(matrix, axis=1) * max(np.linalg.norm(q), 1e-12)).clip(min=1e-12)
        order = np.argsort(sims)[::-1][: self.k]
        return [(self.exchanges[i], float(sims[i])) for i in order]

    def messages(self, query: str | None = None) -> list[Message]:
        if not query:
            return []
        recalled = "\n---\n".join(text for text, _ in self.recall(query))
        return [{"role": "system", "content": f"Relevant past conversation:\n{recalled}"}] if recalled else []

    def clear(self) -> None:
        self.exchanges.clear()
        self.vectors.clear()
