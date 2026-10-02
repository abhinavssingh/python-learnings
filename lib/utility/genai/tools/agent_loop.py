import json
from dataclasses import dataclass, field
from typing import Any

from ..abstractions.base_llm import BaseLLM
from .tool import Tool, ToolRegistry


@dataclass
class AgentStep:
    step: int
    kind: str
    content: str


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)


class ToolCallingAgent:
    """
    A ReAct-style agent loop using the model's *native* tool calling
    (no framework) - this is exactly what LangGraph automates later.

        loop:
            LLM(messages + tool schemas)
            if it requested tools -> run them, append results, continue
            else -> final answer
    """

    SYSTEM = ("You are a helpful assistant. Use the available tools whenever they help; "
              "for arithmetic always use the calculator. Answer concisely once you have the information.")

    def __init__(self, llm: BaseLLM, tools: list[Tool], max_steps: int = 5, system_prompt: str | None = None):
        self.llm = llm
        self.registry = ToolRegistry(tools)
        self.max_steps = max_steps
        self.system_prompt = system_prompt or self.SYSTEM

    def run(self, question: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        steps: list[AgentStep] = []
        for step in range(1, self.max_steps + 1):
            response = self.llm.chat(messages, tools=self.registry.schemas())
            if not response.tool_calls:
                steps.append(AgentStep(step, "final_answer", response.text))
                messages.append({"role": "assistant", "content": response.text})
                return AgentResult(response.text, steps, messages)

            messages.append({
                "role": "assistant",
                "content": response.text,
                "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in response.tool_calls],
            })
            for call in response.tool_calls:
                steps.append(AgentStep(step, "tool_call", f"{call['name']}({json.dumps(call['arguments'])})"))
                output = self.registry.execute(call["name"], call["arguments"])
                steps.append(AgentStep(step, "tool_result", output[:500]))
                messages.append({"role": "tool", "content": output, "tool_name": call["name"]})

        final = "Stopped: reached max_steps without a final answer."
        steps.append(AgentStep(self.max_steps, "stopped", final))
        return AgentResult(final, steps, messages)
