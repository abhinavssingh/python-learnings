from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

Message = dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.latency_s if self.latency_s else 0.0


class BaseLLM(ABC):
    """Minimal chat-model interface used by every GenAI utility."""

    model: str = ""

    @abstractmethod
    def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        pass

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> LLMResponse:
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs).text
