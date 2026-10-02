"""
Generation parameters — controlling how the next token is sampled.

Learn:
  * temperature: 0 = deterministic/greedy, higher = more random & creative
  * seed: reproducible sampling
  * top_p / top_k: restrict sampling to the most likely tokens
  * max_tokens (num_predict): hard cap on output length (answers get cut off)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.factory import get_llm  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402

PROMPT = "Suggest a creative name for an employee wellbeing programme. Reply with the name only."


def main():
    llm = get_llm()

    rows = []
    for temperature in (0.0, 0.8, 1.5):
        for sample in (1, 2):
            r = llm.generate(PROMPT, temperature=temperature, max_tokens=15)
            rows.append({"temperature": temperature, "sample": sample, "output": r.text})

    seeded = [llm.generate(PROMPT, temperature=1.0, seed=7, max_tokens=15).text for _ in range(2)]
    narrow = llm.generate(PROMPT, temperature=1.0, top_k=5, top_p=0.5, max_tokens=15).text
    truncated = llm.generate("Explain what a human resources policy is.", max_tokens=12)

    report = GenAIReport("Generation parameters")
    report.full_table("Temperature: two samples per setting", rows)
    report.grid([
        report.kv("Seed = 7, temperature = 1.0 (run twice)", {"run 1": seeded[0], "run 2": seeded[1],
                                                               "identical": seeded[0] == seeded[1]}),
        report.kv("top_k = 5, top_p = 0.5", {"output": narrow}),
        report.kv("max_tokens = 12 (truncated!)", {"output": truncated.text, "completion tokens": truncated.completion_tokens}),
    ], columns=3)
    report.full_text("Key takeaways", "\n".join([
        "* Use temperature 0 for RAG, extraction, grading and evaluation — you want repeatable answers.",
        "* Raise temperature for brainstorming / creative writing.",
        "* A fixed seed makes sampling reproducible on the same model + hardware.",
        "* max_tokens protects latency & cost but can cut answers mid-sentence — size it to the task.",
    ]))
    report.save(__file__, "generation_parameters_report.html")


if __name__ == "__main__":
    main()
