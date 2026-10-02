"""
LangChain Expression Language (LCEL) basics with ChatOllama.

Learn:
  * Runnables composed with `|` : prompt | model | output parser
  * invoke / batch / stream — the same interface on every runnable
  * RunnableParallel, RunnablePassthrough.assign, RunnableLambda
  * structured output with `llm.with_structured_output(PydanticModel)`
  * visualising a chain graph
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from lib.utility.genai.frameworks.langchain import build_structured_chain, get_chat_model  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.visualization import mermaid_html  # noqa: E402


class PolicyFacts(BaseModel):
    topic: str = Field(description="main topic of the text")
    key_points: list[str] = Field(description="2-3 short key points")
    audience: str = Field(description="who the text is written for")


POLICY = ("Nestlé focuses on Fixed Pay, Variable Pay, Benefits, Personal Growth and Development and Work Life "
          "Environment as the key elements that define Total Rewards. Programmes must respect local law and "
          "collective agreements.")


def main():
    llm = get_chat_model(num_predict=120)

    explain = ChatPromptTemplate.from_messages([
        ("system", "You explain HR concepts to {audience} in one sentence."),
        ("human", "What is {concept}?"),
    ])
    chain = explain | llm | StrOutputParser()

    single = chain.invoke({"audience": "a new graduate", "concept": "variable pay"})
    batch = chain.batch([{"audience": "a child", "concept": "a salary"},
                         {"audience": "a CFO", "concept": "total rewards"}])
    streamed = [piece for piece in chain.stream({"audience": "a manager", "concept": "a 360 assessment"})]

    raw_message = (explain | llm).invoke({"audience": "an engineer", "concept": "collective bargaining"})

    summarize = ChatPromptTemplate.from_template("Summarise in max 12 words: {text}") | llm | StrOutputParser()
    keywords = ChatPromptTemplate.from_template("Give 4 comma-separated keywords for: {text}") | llm | StrOutputParser()
    parallel = RunnableParallel(summary=summarize, keywords=keywords)
    enriched = (RunnablePassthrough.assign(chars=RunnableLambda(lambda x: len(x["text"])))
                | RunnablePassthrough.assign(analysis=parallel))
    enriched_out = enriched.invoke({"text": POLICY})

    structured = build_structured_chain(llm, PolicyFacts, "Extract structured facts from the HR text.")
    facts = structured.invoke({"input": POLICY})

    report = GenAIReport("LangChain LCEL basics (ChatOllama)")
    report.grid([
        report.kv("invoke", {"chain": "prompt | ChatOllama | StrOutputParser", "output": single}),
        report.kv("batch (runs inputs concurrently)", {f"input {i + 1}": out for i, out in enumerate(batch)}),
    ])
    report.grid([
        report.kv("stream", {"chunks": len(streamed), "joined": "".join(streamed)}),
        report.kv("AIMessage metadata (no parser)", {"content": raw_message.content,
                                                     "usage_metadata": raw_message.usage_metadata,
                                                     "model": raw_message.response_metadata.get("model")}),
    ])
    report.grid([
        report.kv("RunnableParallel + assign", {"chars": enriched_out["chars"], **enriched_out["analysis"]}),
        report.kv("with_structured_output(PolicyFacts)", facts.model_dump()),
    ])
    report.full("Chain graph: RunnablePassthrough.assign -> RunnableParallel", mermaid_html(enriched.get_graph().draw_mermaid()))
    report.full_text("Key takeaways", "\n".join([
        "* Everything in LCEL is a Runnable with invoke / batch / stream / ainvoke.",
        "* `|` pipes the output of one runnable into the next; dicts become RunnableParallel.",
        "* Output parsers turn AIMessage into str / JSON / Pydantic objects.",
        "* LangChain wraps the same Ollama calls you made by hand in 03_llm_basics — learn both levels.",
    ]))
    report.save(__file__, "langchain_lcel_basics_report.html")


if __name__ == "__main__":
    main()
