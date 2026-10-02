import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


@dataclass
class Tool:
    """A Python function the LLM may call, described by a JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_schema(self) -> dict[str, Any]:
        """OpenAI / Ollama function-calling format."""
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def run(self, arguments: dict[str, Any] | str) -> str:
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        try:
            result = self.func(**arguments)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        return result if isinstance(result, str) else json.dumps(result, default=str)


def _param_descriptions(doc: str) -> dict[str, str]:
    """Read 'name: description' lines from an Args: section of a docstring."""
    descriptions = {}
    in_args = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower() in {"args:", "arguments:", "parameters:"}:
            in_args = True
            continue
        if in_args:
            match = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", stripped)
            if match:
                descriptions[match.group(1)] = match.group(2)
            elif not stripped:
                break
    return descriptions


def tool(func: Callable | None = None, *, name: str | None = None):
    """
    Decorator that turns a typed, documented function into a Tool.

        @tool
        def add(a: int, b: int) -> int:
            '''Add two numbers.

            Args:
                a: first number
                b: second number
            '''
    """
    def wrap(fn: Callable) -> Tool:
        hints = get_type_hints(fn)
        doc = inspect.getdoc(fn) or ""
        arg_docs = _param_descriptions(doc)
        properties, required = {}, []
        for pname, param in inspect.signature(fn).parameters.items():
            properties[pname] = {"type": PY_TO_JSON.get(hints.get(pname, str), "string"),
                                 "description": arg_docs.get(pname, pname)}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        summary = doc.split("\n\n")[0].strip() or fn.__name__
        return Tool(name or fn.__name__, summary, {"type": "object", "properties": properties, "required": required}, fn)

    return wrap(func) if func else wrap


class ToolRegistry:
    """Holds tools by name and executes the calls an LLM requests."""

    def __init__(self, tools: list[Tool] | None = None):
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}

    def register(self, t: Tool) -> None:
        self.tools[t.name] = t

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self.tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self.tools:
            return f"ERROR: unknown tool '{name}'. Available: {list(self.tools)}"
        return self.tools[name].run(arguments)
