import re
from dataclasses import dataclass, field


class PIIMasker:
    """Mask personally identifiable information before text reaches an LLM or a log."""

    PATTERNS = {
        "EMAIL": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
        "CREDIT_CARD": r"\b(?:\d[ -]?){13,16}\b",
        "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
        "PHONE": r"(?:\+\d{1,3}[\s-]?)?\b\d{3,5}[\s-]?\d{3,4}[\s-]?\d{3,4}\b",
        "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    def __init__(self, entities: list[str] | None = None):
        self.entities = entities or list(self.PATTERNS)

    def find(self, text: str) -> list[tuple[str, str]]:
        found = []
        for entity in self.entities:
            found.extend((entity, m.group(0)) for m in re.finditer(self.PATTERNS[entity], text))
        return found

    def mask(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        found = []
        for entity in self.entities:
            def repl(match, entity=entity):
                found.append((entity, match.group(0)))
                return f"<{entity}>"
            text = re.sub(self.PATTERNS[entity], repl, text)
        return text, found


class PromptInjectionDetector:
    """Heuristic detector for common prompt-injection / jailbreak phrases."""

    SIGNALS = [
        r"ignore (?:all |any )?(?:previous|prior|above) (?:instructions|prompts?)",
        r"disregard (?:the |all )?(?:previous|above|system)",
        r"you are now",
        r"act as (?:an? )?(?:unfiltered|jailbroken|dan)",
        r"(?:reveal|show|print) (?:your|the) (?:system prompt|instructions|hidden)",
        r"pretend (?:that )?you",
        r"developer mode",
        r"bypass (?:the |your )?(?:rules|filters|safety)",
    ]

    def score(self, text: str) -> tuple[float, list[str]]:
        hits = [p for p in self.SIGNALS if re.search(p, text, re.IGNORECASE)]
        return min(1.0, len(hits) / 2), hits

    def is_suspicious(self, text: str, threshold: float = 0.5) -> bool:
        return self.score(text)[0] >= threshold


@dataclass
class ValidationReport:
    passed: bool
    issues: list[str] = field(default_factory=list)


class OutputValidator:
    """Simple output guardrails: length limits, banned terms and required citations."""

    def __init__(self, max_chars: int = 2000, banned_terms: list[str] | None = None, require_citation: bool = False):
        self.max_chars = max_chars
        self.banned_terms = banned_terms or []
        self.require_citation = require_citation

    def validate(self, text: str) -> ValidationReport:
        issues = []
        if len(text) > self.max_chars:
            issues.append(f"too long ({len(text)} > {self.max_chars} chars)")
        for term in self.banned_terms:
            if term.lower() in text.lower():
                issues.append(f"contains banned term '{term}'")
        if self.require_citation and not re.search(r"\[\d+\]", text):
            issues.append("missing citation like [1]")
        if PIIMasker().find(text):
            issues.append("contains PII")
        return ValidationReport(passed=not issues, issues=issues)
