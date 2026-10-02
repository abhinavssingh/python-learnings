import json
import re
from typing import Any, Type

from pydantic import BaseModel, ValidationError

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class OutputParserError(ValueError):
    pass


class JsonOutputParser:
    """Extract the first JSON object/array from LLM text (tolerates code fences and chatter)."""

    def parse(self, text: str) -> Any:
        text = THINK_BLOCK.sub("", text).strip()
        fenced = CODE_FENCE.search(text)
        if fenced:
            text = fenced.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = text.find(opener), text.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start: end + 1])
                except json.JSONDecodeError:
                    continue
        raise OutputParserError(f"Could not parse JSON from: {text[:200]}")

    def safe_parse(self, text: str, default: Any = None) -> Any:
        try:
            return self.parse(text)
        except OutputParserError:
            return default


class PydanticOutputParser:
    """Validate LLM JSON against a Pydantic model; also exposes its JSON schema for Ollama `format=`."""

    def __init__(self, model: Type[BaseModel]):
        self.model = model
        self.json_parser = JsonOutputParser()

    @property
    def schema(self) -> dict:
        return self.model.model_json_schema()

    def format_instructions(self) -> str:
        return ("Respond with a JSON object that matches this JSON schema:\n"
                f"{json.dumps(self.schema, indent=1)}")

    def parse(self, text: str) -> BaseModel:
        data = self.json_parser.parse(text)
        try:
            return self.model.model_validate(data)
        except ValidationError as exc:
            raise OutputParserError(str(exc)) from exc


class ListOutputParser:
    """Parse numbered / bulleted / one-per-line lists."""

    def parse(self, text: str) -> list[str]:
        text = THINK_BLOCK.sub("", text)
        items = []
        for line in text.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
            if line:
                items.append(line)
        return items


class YesNoParser:
    """Read a yes/no decision from JSON ({"relevant": "yes"}) or plain text."""

    def __init__(self, key: str | None = None):
        self.key = key
        self.json_parser = JsonOutputParser()

    def parse(self, text: str) -> bool:
        data = self.json_parser.safe_parse(text)
        if isinstance(data, dict):
            value = data.get(self.key) if self.key else next(iter(data.values()), "")
            return str(value).strip().lower() in {"yes", "true", "1"}
        return bool(re.search(r"\byes\b", text, re.IGNORECASE))
