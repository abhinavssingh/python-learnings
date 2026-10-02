from typing import Type

from langchain_core.documents import Document as LCDocument
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import BaseModel

from ...prompting.output_parsers import ListOutputParser
from ...prompting.templates import PromptLibrary


def format_docs(docs: list[LCDocument]) -> str:
    return "\n\n".join(
        f"[{i}] (p.{d.metadata.get('page', '?')}) {' '.join(d.page_content.split())}"
        for i, d in enumerate(docs, start=1)
    )


def rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(PromptLibrary.get("rag_answer").template)


def build_rag_chain(retriever: Runnable, llm: BaseChatModel) -> Runnable:
    """
    Classic LCEL RAG chain:

        {"context": retriever | format_docs, "question": passthrough}
            | prompt | llm | StrOutputParser()
    """
    return (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | rag_prompt()
        | llm
        | StrOutputParser()
    )


def build_rag_chain_with_sources(retriever: Runnable, llm: BaseChatModel) -> Runnable:
    """Same chain but returns {"question", "docs", "answer"} so sources can be shown."""
    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | rag_prompt()
        | llm
        | StrOutputParser()
    )
    return RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=answer_chain)


def build_structured_chain(llm: BaseChatModel, schema: Type[BaseModel], instructions: str) -> Runnable:
    """prompt | llm.with_structured_output(schema) -> returns a validated Pydantic object."""
    prompt = ChatPromptTemplate.from_messages([("system", instructions), ("human", "{input}")])
    return prompt | llm.with_structured_output(schema)


def build_multi_query_chain(llm: BaseChatModel, n: int = 3) -> Runnable:
    prompt = ChatPromptTemplate.from_template(PromptLibrary.get("multi_query").template).partial(n=str(n))
    return prompt | llm | StrOutputParser() | RunnableLambda(lambda text: ListOutputParser().parse(text)[:n])
