import re
import string
from pathlib import Path

from ..config.genai_config import PROMPTS_DIR


class PromptTemplate:
    """
    A string template with {placeholders}.

    Literal braces (e.g. JSON examples) must be doubled: {{"key": "value"}}.
    """

    def __init__(self, template: str, partial_variables: dict[str, str] | None = None):
        self.template = template
        self.partial_variables = partial_variables or {}

    @property
    def input_variables(self) -> list[str]:
        names = [f for _, f, _, _ in string.Formatter().parse(self.template) if f]
        return [n for n in dict.fromkeys(names) if n not in self.partial_variables]

    def format(self, **kwargs) -> str:
        missing = [v for v in self.input_variables if v not in kwargs]
        if missing:
            raise KeyError(f"Missing prompt variables: {missing}")
        return self.template.format(**self.partial_variables, **kwargs)

    def partial(self, **kwargs) -> "PromptTemplate":
        return PromptTemplate(self.template, {**self.partial_variables, **kwargs})

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptTemplate":
        return cls(Path(path).read_text(encoding="utf-8").strip())


class ChatPromptTemplate:
    """A list of (role, template) pairs that formats into chat messages."""

    def __init__(self, messages: list[tuple[str, str]]):
        self.messages = [(role, PromptTemplate(t)) for role, t in messages]

    def format_messages(self, **kwargs) -> list[dict[str, str]]:
        out = []
        for role, template in self.messages:
            needed = {k: v for k, v in kwargs.items() if k in template.input_variables}
            out.append({"role": role, "content": template.format(**needed)})
        return out


class FewShotPromptTemplate:
    """prefix + formatted examples + suffix (the classic few-shot pattern)."""

    def __init__(self, examples: list[dict[str, str]], example_template: str, prefix: str = "",
                 suffix: str = "", separator: str = "\n\n"):
        self.examples = examples
        self.example_prompt = PromptTemplate(example_template)
        self.prefix = prefix
        self.suffix = PromptTemplate(suffix)
        self.separator = separator

    def format(self, **kwargs) -> str:
        shots = [self.example_prompt.format(**ex) for ex in self.examples]
        parts = ([self.prefix] if self.prefix else []) + shots + [self.suffix.format(**kwargs)]
        return self.separator.join(parts)


class PromptLibrary:
    """Loads versioned prompt files from lib/utility/genai/config/prompts/*.txt."""

    _cache: dict[str, PromptTemplate] = {}

    @classmethod
    def get(cls, name: str) -> PromptTemplate:
        if name not in cls._cache:
            path = PROMPTS_DIR / f"{name}.txt"
            if not path.exists():
                raise FileNotFoundError(f"Prompt '{name}' not found in {PROMPTS_DIR}")
            cls._cache[name] = PromptTemplate.from_file(path)
        return cls._cache[name]

    @classmethod
    def names(cls) -> list[str]:
        return sorted(p.stem for p in PROMPTS_DIR.glob("*.txt"))


# ----------------------------------------------------------
# Classic prompting techniques
# ----------------------------------------------------------

ZERO_SHOT = PromptTemplate("{instruction}\n\nInput: {input}\nOutput:")

CHAIN_OF_THOUGHT = PromptTemplate(
    "{question}\n\nLet's think step by step. Show your reasoning as numbered steps, "
    "then give the final answer on the last line as 'Answer: <answer>'."
)

ROLE_PROMPT = ChatPromptTemplate([
    ("system", "You are {role}. {style}"),
    ("user", "{question}"),
])


def extract_final_answer(text: str) -> str:
    match = re.search(r"Answer:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip().splitlines()[-1] if text.strip() else ""
