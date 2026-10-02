import ast
import datetime as dt
import operator

from ..abstractions.base_retriever import BaseRetriever
from .tool import Tool, tool

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic without `eval` (only numbers and + - * / ** % //)."""
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")
    return _eval(ast.parse(expression.replace("^", "**"), mode="eval"))


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(12 * 4) / 3' and return the result.

    Args:
        expression: arithmetic expression using numbers and + - * / ** %
    """
    result = safe_eval(expression)
    return str(round(result, 6) if isinstance(result, float) else result)


@tool
def current_datetime() -> str:
    """Return the current local date and time in ISO format."""
    return dt.datetime.now().isoformat(timespec="seconds")


@tool
def word_count(text: str) -> str:
    """Count the words and characters in a text.

    Args:
        text: the text to analyse
    """
    return f"{len(text.split())} words, {len(text)} characters"


def make_document_search_tool(retriever: BaseRetriever, k: int = 3, name: str = "search_hr_policy",
                              description: str = "Search the company HR policy documents and return the most relevant passages.") -> Tool:
    def search(query: str) -> str:
        results = retriever.retrieve(query, k)
        return "\n\n".join(f"[page {r.metadata.get('page', '?')}] {' '.join(r.text.split())}" for r in results) or "No results."

    return Tool(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {"query": {"type": "string", "description": "search query"}}, "required": ["query"]},
        func=search,
    )
