from ..abstractions.base_llm import BaseLLM
from ..prompting.output_parsers import JsonOutputParser
from ..prompting.templates import PromptLibrary


class LLMJudge:
    """
    LLM-as-a-judge: one call scores faithfulness, answer relevance and
    context relevance on a 1-5 scale (the RAG "triad").
    """

    CRITERIA = ("faithfulness", "answer_relevance", "context_relevance")

    def __init__(self, llm: BaseLLM, max_context_chars: int = 2500):
        self.llm = llm
        self.max_context_chars = max_context_chars
        self.prompt = PromptLibrary.get("judge_rag")

    def evaluate(self, question: str, answer: str, context: str, reference: str | None = None) -> dict:
        reference_block = f"\nReference answer (ground truth): {reference}\n" if reference else ""
        prompt = self.prompt.format(question=question, answer=answer, context=context[: self.max_context_chars],
                                    reference_block=reference_block)
        response = self.llm.generate(prompt, max_tokens=120, format="json")
        data = JsonOutputParser().safe_parse(response.text, default={}) or {}
        result = {}
        for criterion in self.CRITERIA:
            try:
                result[criterion] = max(1.0, min(5.0, float(data[criterion])))
            except (KeyError, TypeError, ValueError):
                result[criterion] = float("nan")
        result["reason"] = str(data.get("reason", ""))
        return result
