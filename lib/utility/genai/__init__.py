"""
GenAI learning framework (framework-agnostic core + LangChain / LangGraph integrations).

Layout mirrors lib/utility/deeplearning:
    abstractions/  base interfaces (LLM, embedder, chunker, vector store, retriever)
    config/        GenAIConfig + versioned prompt files
    foundations/   tokenization, n-grams, TF-IDF, BM25, similarity, attention (from scratch)
    providers/     Ollama chat + embedding clients, provider registry
    embeddings/    hashing / TF-IDF / Word2Vec embedders, disk cache, analysis
    loaders/       PDF, text, CSV/Excel, directory loaders
    chunking/      fixed, recursive, sentence, paragraph, semantic, markdown, code chunkers
    vectorstores/  in-memory (NumPy) and FAISS stores, metadata filters, index manager
    retrieval/     dense, BM25, hybrid (RRF), MMR, multi-query, HyDE, reranking, compression
    prompting/     templates, output parsers, guardrails
    rag/           naive, advanced and conversational RAG pipelines + citations
    memory/        buffer, window, summary and vector conversation memory
    tools/         tool decorator, built-in tools, native tool-calling agent loop
    frameworks/    langchain/ (LCEL chains, adapters) and langgraph/ (graphs, nodes, state)
    evaluation/    retrieval metrics, generation metrics, LLM-as-judge, RAG evaluator
    observability/ tracer for spans and token usage
    visualization/ Plotly figures for NLP / embeddings / RAG metrics
    reports/       GenAIReport helper for HTML reports
"""
