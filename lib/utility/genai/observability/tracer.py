import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..abstractions.base_llm import LLMResponse


@dataclass
class Span:
    name: str
    start: float
    end: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.end - self.start


class Tracer:
    """
    Lightweight observability (the idea behind LangSmith / Langfuse):
    timed spans for pipeline steps + token usage of every LLM call.

        tracer = Tracer()
        llm = OllamaLLM(tracer=tracer)
        with tracer.span("retrieve", k=4):
            ...
        tracer.to_frame()
    """

    def __init__(self):
        self.spans: list[Span] = []
        self.llm_calls: list[dict[str, Any]] = []
        self._origin = time.perf_counter()

    @contextmanager
    def span(self, name: str, **attributes):
        span = Span(name, time.perf_counter(), attributes=attributes)
        try:
            yield span
        finally:
            span.end = time.perf_counter()
            self.spans.append(span)

    def record_llm(self, response: LLMResponse, messages: list[dict] | None = None) -> None:
        prompt_preview = (messages[-1]["content"] if messages else "")[:80].replace("\n", " ")
        self.llm_calls.append({
            "call": len(self.llm_calls) + 1,
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "latency_s": round(response.latency_s, 2),
            "tokens_per_s": round(response.tokens_per_second, 1),
            "tool_calls": len(response.tool_calls),
            "prompt_preview": prompt_preview,
        })

    def spans_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"span": s.name, "start_s": round(s.start - self._origin, 2), "duration_s": round(s.duration_s, 3), **s.attributes}
            for s in self.spans
        ])

    def llm_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.llm_calls)

    def summary(self) -> dict[str, Any]:
        calls = self.llm_calls
        return {
            "spans": len(self.spans),
            "llm_calls": len(calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "llm_latency_s": round(sum(c["latency_s"] for c in calls), 2),
            "wall_time_s": round(time.perf_counter() - self._origin, 2),
        }
