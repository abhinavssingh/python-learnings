"""
Prompt engineering techniques.

Learn:
  * zero-shot vs few-shot prompting (classification of HR sentences)
  * chain-of-thought prompting and extracting the final answer
  * role / system prompts that change tone & audience
  * reusable, versioned prompt templates (config/prompts/*.txt)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.factory import get_llm  # noqa: E402
from lib.utility.genai.prompting import (  # noqa: E402
    CHAIN_OF_THOUGHT,
    ROLE_PROMPT,
    ZERO_SHOT,
    FewShotPromptTemplate,
    PromptLibrary,
    extract_final_answer,
)
from lib.utility.genai.reports import GenAIReport  # noqa: E402

TEST_SENTENCES = [
    "My bonus was lower than expected this year.",
    "I would like to enrol in the leadership course.",
    "A colleague keeps making offensive jokes about my religion.",
]
LABELS = "pay, learning, conduct, other"


def main():
    llm = get_llm()

    zero_rows = []
    for sentence in TEST_SENTENCES:
        prompt = ZERO_SHOT.format(instruction=f"Classify the HR ticket into one of: {LABELS}. Reply with the label only.",
                                  input=sentence)
        zero_rows.append({"ticket": sentence, "zero-shot": llm.generate(prompt, max_tokens=5).text})

    few_shot = FewShotPromptTemplate(
        examples=[
            {"ticket": "When will my salary increase be applied?", "label": "pay"},
            {"ticket": "Is there a course on negotiation skills?", "label": "learning"},
            {"ticket": "My manager shouted at me in front of the team.", "label": "conduct"},
            {"ticket": "Where is the parking garage?", "label": "other"},
        ],
        example_template="Ticket: {ticket}\nLabel: {label}",
        prefix=f"Classify HR tickets. Labels: {LABELS}.",
        suffix="Ticket: {ticket}\nLabel:",
    )
    few_prompt = few_shot.format(ticket=TEST_SENTENCES[0])
    for row, sentence in zip(zero_rows, TEST_SENTENCES):
        row["few-shot"] = llm.generate(few_shot.format(ticket=sentence), max_tokens=5).text

    cot_question = ("An employee earns 4000 per month. Variable pay is 10% of annual fixed pay. "
                    "What is the total annual pay?")
    cot_raw = llm.generate(CHAIN_OF_THOUGHT.format(question=cot_question), max_tokens=220).text

    role_rows = []
    for role, style in (("a senior HR business partner", "Use formal language."),
                        ("a friendly buddy explaining to a new intern", "Use simple words and one emoji.")):
        messages = ROLE_PROMPT.format_messages(role=role, style=style + " Max 2 sentences.",
                                               question="Why do we have performance evaluations?")
        role_rows.append({"role": role, "answer": llm.chat(messages, max_tokens=70).text})

    report = GenAIReport("Prompt engineering techniques")
    report.full_table("Zero-shot vs few-shot classification", zero_rows)
    report.full_text("The few-shot prompt actually sent", few_prompt)
    report.grid([
        report.text("Chain-of-thought output", cot_raw, max_lines=20),
        report.kv("Extracted final answer", {"question": cot_question, "answer": extract_final_answer(cot_raw),
                                             "expected": "4000*12 = 48000 + 4800 = 52800"}),
    ])
    report.full_table("Role prompting (same question, different persona)", role_rows)
    report.full_table("Versioned prompt library (config/prompts)", [
        {"name": n, "variables": ", ".join(PromptLibrary.get(n).input_variables)} for n in PromptLibrary.names()
    ], rows=15)
    report.full_text("Key takeaways", "\n".join([
        "* Few-shot examples teach the output FORMAT and label boundaries better than instructions alone.",
        "* Chain-of-thought helps arithmetic / multi-step reasoning; ask for a parseable final line.",
        "* System / role prompts steer tone, audience and constraints.",
        "* Keep prompts in files under version control — they are code.",
    ]))
    report.save(__file__, "prompting_techniques_report.html")


if __name__ == "__main__":
    main()
