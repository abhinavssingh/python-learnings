"""Local conversational assistant for the Nestle HR policy PDF."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

import gradio as gr  # noqa: E402

from lib.utility.genai.config import GenAIConfig  # noqa: E402
from lib.utility.genai.factory import build_hr_vectorstore, get_llm  # noqa: E402
from lib.utility.genai.memory import WindowMemory  # noqa: E402
from lib.utility.genai.rag import ConversationalRAG  # noqa: E402
from lib.utility.genai.retrieval import DenseRetriever  # noqa: E402


def create_app() -> gr.Blocks:
    """Build the policy index once and create a per-conversation RAG interface."""
    config = GenAIConfig.from_env()
    llm = get_llm(config)
    store, chunks = build_hr_vectorstore(config)
    retriever = DenseRetriever(store)

    def respond(message: str, history: list[dict[str, str]] | None) -> tuple[str, list[dict[str, str]]]:
        question = message.strip()
        if not question:
            raise gr.Error("Enter a question about the HR policy.")

        memory = WindowMemory(k=4)
        for turn in (history or [])[-8:]:
            role = turn.get("role")
            content = turn.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                memory.add(role, content)

        rag = ConversationalRAG(retriever, llm, memory=memory, k=config.top_k)
        result = rag.ask(question)
        if not result.sources:
            answer = "I could not find relevant information in the HR policy. Please try rephrasing your question."
        else:
            answer = result.answer
        updated_history = list(history or [])
        updated_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        return "", updated_history

    with gr.Blocks(title="Nestle HR Policy Assistant") as app:
        gr.Markdown(
            "# Nestle HR Policy Assistant\n"
            "Ask any question about the HR policy. Answers are grounded in retrieved text "
            f"({len(chunks)} chunks indexed with {config.embedding_model}; chat model: {config.chat_model})."
        )
        chatbot = gr.Chatbot(height=500)
        with gr.Row():
            question = gr.Textbox(
                placeholder="Type any HR policy question here",
                lines=1,
                max_lines=4,
                label="Your question",
                scale=5,
            )
            submit = gr.Button("Ask", variant="primary", scale=1)
        gr.Examples(
            examples=[
                "What are the key elements of Total Rewards?",
                "What is the policy on harassment?",
                "What are promotions based on?",
            ],
            inputs=question,
            label="Example questions (optional)",
        )
        submit.click(respond, inputs=[question, chatbot], outputs=[question, chatbot])
        question.submit(respond, inputs=[question, chatbot], outputs=[question, chatbot])

    return app

def main() -> None:
    app = create_app()
    app.launch(
        server_name=os.getenv("GENAI_GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.getenv("GENAI_GRADIO_PORT", "7860")),
        show_error=True,
    )


if __name__ == "__main__":
    main()
