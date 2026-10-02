# Observability

Lightweight tracing for model and pipeline operations. `Tracer` records spans,
and `Span` represents an individual operation with timing and metadata.

Pass a tracer to supported components to capture calls and token-usage details;
consult the component constructors for the specific instrumentation options.
The Ollama chat basics lesson demonstrates tracing in
`Module-5/GENAI/03_llm_basics/01_ollama_chat_basics.py`.
