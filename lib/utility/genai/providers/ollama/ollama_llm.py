import re
import time
from typing import Any, Iterator

import ollama

from ...abstractions.base_llm import BaseLLM, LLMResponse, Message
from ...config.genai_config import GenAIConfig

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaNotAvailableError(RuntimeError):
    pass


class OllamaLLM(BaseLLM):
    """
    Chat model served by a local Ollama instance (default: qwen3:8b).

    Every call is recorded in `self.history` so scripts can report
    latency, token usage and throughput.
    """

    def __init__(self, config: GenAIConfig | None = None, model: str | None = None, tracer=None, **default_options):
        self.config = config or GenAIConfig.from_env()
        self.model = model or self.config.chat_model
        self.client = ollama.Client(host=self.config.ollama_host, timeout=self.config.request_timeout)
        self.tracer = tracer
        self.default_options = default_options
        self.history: list[LLMResponse] = []

    # ------------------------------------------------------
    # Health checks
    # ------------------------------------------------------

    def list_models(self) -> list[str]:
        return [m.model for m in self.client.list().models]

    def ensure_available(self) -> None:
        try:
            models = self.list_models()
        except Exception as exc:
            raise OllamaNotAvailableError(
                f"Ollama is not reachable at {self.config.ollama_host}. Start it with `ollama serve`."
            ) from exc
        if not any(m == self.model or m.split(":")[0] == self.model for m in models):
            raise OllamaNotAvailableError(f"Model '{self.model}' not found. Run `ollama pull {self.model}`.")

    # ------------------------------------------------------
    # Chat
    # ------------------------------------------------------

    def _options(self, **kwargs) -> dict[str, Any]:
        options = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "num_predict": self.config.max_tokens,
            "num_ctx": self.config.context_window,
            **self.default_options,
        }
        mapping = {"max_tokens": "num_predict", "temperature": "temperature", "top_p": "top_p",
                   "top_k": "top_k", "seed": "seed", "num_ctx": "num_ctx", "repeat_penalty": "repeat_penalty"}
        for key, option in mapping.items():
            if kwargs.get(key) is not None:
                options[option] = kwargs[key]
        return options

    def chat(self, messages: list[Message], tools: list[dict] | None = None, format: Any = None,
             think: bool | None = None, **kwargs) -> LLMResponse:
        think = self.config.think if think is None else think
        start = time.perf_counter()
        raw = self.client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            format=format,
            think=think,
            options=self._options(**kwargs),
        )
        latency = time.perf_counter() - start
        message = raw.message
        text = THINK_PATTERN.sub("", message.content or "").strip()
        tool_calls = [
            {"name": call.function.name, "arguments": dict(call.function.arguments or {})}
            for call in (message.tool_calls or [])
        ]
        response = LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=raw.prompt_eval_count or 0,
            completion_tokens=raw.eval_count or 0,
            latency_s=latency,
            thinking=message.thinking or "",
            tool_calls=tool_calls,
        )
        self.history.append(response)
        if self.tracer is not None:
            self.tracer.record_llm(response, messages=messages)
        return response

    def stream(self, messages: list[Message], think: bool | None = None, **kwargs) -> Iterator[str]:
        think = self.config.think if think is None else think
        for chunk in self.client.chat(model=self.model, messages=messages, stream=True, think=think,
                                      options=self._options(**kwargs)):
            if chunk.message.content:
                yield chunk.message.content

    # ------------------------------------------------------
    # Stats
    # ------------------------------------------------------

    def usage_summary(self) -> dict[str, Any]:
        calls = len(self.history)
        latency = sum(r.latency_s for r in self.history)
        completion = sum(r.completion_tokens for r in self.history)
        return {
            "model": self.model,
            "calls": calls,
            "prompt_tokens": sum(r.prompt_tokens for r in self.history),
            "completion_tokens": completion,
            "total_latency_s": round(latency, 2),
            "avg_tokens_per_s": round(completion / latency, 2) if latency else 0.0,
        }
