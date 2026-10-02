"""
RAG with LangChain — two ways.

Learn:
  1. LangChain-native: OllamaEmbeddings + langchain_core InMemoryVectorStore + retriever + LCEL chain
  2. Adapter: plug OUR framework retriever (hybrid dense + BM25) into an LCEL chain
  * returning sources alongside the answer
  * a multi-query expansion chain
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from langchain_core.vectorstores import InMemoryVectorStore as LCInMemoryVectorStore  # noqa: E402

from lib.utility.genai.factory import build_hr_vectorstore  # noqa: E402
from lib.utility.genai.frameworks.langchain import (  # noqa: E402
    as_langchain_retriever,
    build_multi_query_chain,
    build_rag_chain,
    build_rag_chain_with_sources,
    get_chat_model,
    get_embeddings,
    to_lc_documents,
)
from lib.utility.genai.reports import GenAIReport  # noqa: E402
from lib.utility.genai.retrieval import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402

QUESTION = "Which tools are used to give employees feedback on performance?"


def main():
    llm = get_chat_model(num_predict=160)
    store, chunks = build_hr_vectorstore()
    lc_docs = to_lc_documents(chunks)

    lc_store = LCInMemoryVectorStore.from_documents(lc_docs, embedding=get_embeddings())
    lc_retriever = lc_store.as_retriever(search_kwargs={"k": 3})
    native_chain = build_rag_chain(lc_retriever, llm)
    native_answer = native_chain.invoke(QUESTION)

    hybrid = HybridRetriever([DenseRetriever(store), BM25Retriever(chunks)], fetch_k=10)
    adapted = as_langchain_retriever(hybrid, k=3)
    sourced = build_rag_chain_with_sources(adapted, llm).invoke(QUESTION)

    variants = build_multi_query_chain(llm, n=3).invoke({"question": "how do I get promoted?"})

    report = GenAIReport("RAG with LangChain (LCEL)")
    report.grid([
        report.kv("1) LangChain-native pipeline", {
            "embeddings": "langchain_ollama.OllamaEmbeddings", "vector store": "langchain_core InMemoryVectorStore",
            "documents": len(lc_docs), "question": QUESTION, "answer": native_answer}),
        report.kv("2) Our hybrid retriever via adapter", {
            "retriever": "HybridRetriever(dense + BM25) -> LangChain BaseRetriever", "answer": sourced["answer"]}),
    ])
    report.full_table("Sources returned by chain 2", [
        {"rank": i + 1, "page": d.metadata.get("page"), "score": round(d.metadata.get("score", 0), 4),
         "text": " ".join(d.page_content.split())[:150]} for i, d in enumerate(sourced["docs"])])
    report.full_table("Multi-query expansion chain", [{"variant": v} for v in variants])
    report.full_text("Chain structure", "\n".join([
        "native:  {context: retriever | format_docs, question: passthrough} | prompt | ChatOllama | StrOutputParser",
        "sources: RunnableParallel(docs=retriever, question=passthrough).assign(answer=prompt | llm | parser)",
    ]))
    report.full_text("Key takeaways", "\n".join([
        "* A LangChain retriever is just a Runnable: str -> list[Document].",
        "* Adapters let you keep framework-agnostic core logic and still use LangChain / LangGraph.",
        "* Return sources from the chain, not only the answer — citations & debugging need them.",
    ]))
    report.save(__file__, "langchain_rag_chain_report.html")


if __name__ == "__main__":
    main()
