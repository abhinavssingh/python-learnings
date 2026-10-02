# HR Policy Assistant

An interactive, locally hosted chat application for asking questions about the
Nestle HR policy PDF. The Gradio UI retains a short conversation window; each
answer is generated from policy passages retrieved for the current question.

## Run

From the repository root, install the GenAI requirements and start Ollama:

```powershell
.\.venv\Scripts\python.exe -m pip install -r Module-5\GENAI\requirements-genai.txt
ollama pull qwen3:8b
ollama pull nomic-embed-text
.\.venv\Scripts\python.exe Module-5\GENAI\projects\hr_assistant.py
```

Open `http://127.0.0.1:7860`, type any policy question into **Your question**,
then click **Ask** (or press Enter). The example questions are optional
shortcuts, not the only supported questions. On first launch the app reads
`datasets/GEN AI/the_nestle_hr_policy_pdf_2012.pdf`, chunks it, creates
embeddings, and saves the reusable index under `saved_models/genai/`. Subsequent
launches reuse the index when the source documents and embedding model match.

The chatbot uses the framework's local Ollama and RAG utilities rather than
OpenAI, LangChain's `PyPDFLoader`, or ChromaDB. This keeps the project aligned
with the existing local-first GenAI stack: `PDFLoader`, `RecursiveCharacterChunker`,
`IndexManager`, `InMemoryVectorStore`, `DenseRetriever`, and
`ConversationalRAG`. The utility supports FAISS as an alternative vector store.

Configure the chat model, embedding model, and Ollama server with
`GENAI_CHAT_MODEL`, `GENAI_EMBEDDING_MODEL`, and `OLLAMA_HOST`. To change the
local UI bind address or port, set `GENAI_GRADIO_HOST` or
`GENAI_GRADIO_PORT`. The default host only accepts local connections.

If Ollama is unavailable, a model is missing, or document indexing fails, the
application reports the underlying setup error rather than starting with an
empty knowledge base.
