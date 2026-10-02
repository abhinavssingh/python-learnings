"""
Talking to a local LLM (Ollama qwen3:8b) — the raw building blocks.

Learn:
  * chat messages: system / user / assistant roles and multi-turn context
  * streaming tokens and time-to-first-token
  * qwen3 "thinking" mode (reasoning trace) on vs off
  * token usage, latency and tokens/second — tracked with a Tracer

Requires:  ollama pull qwen3:8b
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.factory import get_llm  # noqa: E402
from lib.utility.genai.observability import Tracer  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402


def main():
    tracer = Tracer()
    llm = get_llm(tracer=tracer)

    single = llm.generate("In one sentence, what is a large language model?", max_tokens=60)

    conversation = [
        {"role": "system", "content": "You are a concise HR assistant. Answer in at most 2 sentences."},
        {"role": "user", "content": "My name is Asha and I just joined the company."},
        {"role": "assistant", "content": "Welcome, Asha! Let me know how I can help with HR questions."},
        {"role": "user", "content": "What is my name, and what is one thing I should do in my first week?"},
    ]
    multi = llm.chat(conversation, max_tokens=80)

    t0 = time.perf_counter()
    first_token_s, pieces = None, []
    for piece in llm.stream([{"role": "user", "content": "List three benefits of mentoring, comma separated."}], max_tokens=50):
        if first_token_s is None:
            first_token_s = time.perf_counter() - t0
        pieces.append(piece)
    stream_total = time.perf_counter() - t0

    puzzle = "A team has 12 people. One third are managers. Half of the managers are women. How many women managers?"
    no_think = llm.generate(puzzle + " Reply with just the number.", max_tokens=20, think=False)
    with_think = llm.generate(puzzle + " Reply with just the number.", max_tokens=600, think=True)

    report = GenAIReport(f"Ollama chat basics ({llm.model})")
    report.grid([
        report.kv("Single prompt", {"answer": single.text, "prompt tokens": single.prompt_tokens,
                                    "completion tokens": single.completion_tokens, "latency (s)": round(single.latency_s, 2),
                                    "tokens / s": round(single.tokens_per_second, 1)}),
        report.kv("Multi-turn (model remembers only what is in `messages`)", {"answer": multi.text,
                                                                               "messages sent": len(conversation)}),
    ])
    report.grid([
        report.kv("Streaming", {"chunks received": len(pieces), "time to first token (s)": round(first_token_s or 0, 2),
                                "total (s)": round(stream_total, 2), "text": "".join(pieces)}),
        report.table("Available local models", [{"model": m} for m in llm.list_models()]),
    ])
    report.grid([
        report.kv("think = False", {"answer": no_think.text, "completion tokens": no_think.completion_tokens,
                                    "latency (s)": round(no_think.latency_s, 1)}),
        report.kv("think = True", {"answer": with_think.text, "completion tokens (incl. reasoning)": with_think.completion_tokens,
                                   "latency (s)": round(with_think.latency_s, 1)}),
    ])
    report.full_text("qwen3 reasoning trace (think=True)", with_think.thinking or "(no thinking returned)")
    report.full_table("Tracer: every LLM call", tracer.llm_frame())
    report.full_text("Key takeaways", "\n".join([
        "* LLMs are stateless — 'memory' is just the list of messages you resend every turn.",
        "* Streaming does not make generation faster, it lowers PERCEIVED latency (time to first token).",
        "* Thinking mode improves multi-step reasoning but costs many extra tokens and seconds.",
        f"* Usage so far: {llm.usage_summary()}",
    ]))
    report.save(__file__, "ollama_chat_basics_report.html")


if __name__ == "__main__":
    main()
