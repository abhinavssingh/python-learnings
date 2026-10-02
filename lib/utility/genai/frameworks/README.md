# Framework Integrations

Optional integrations that adapt the framework-agnostic GenAI core to popular
orchestration frameworks. Install the dependencies listed in
`Module-5/GENAI/requirements-genai.txt` before importing an integration.

- `langchain/`: model and embedding helpers, document/retriever adapters, LCEL
  chains, and tool-calling helpers.
- `langgraph/`: graph states, nodes, checkpointing, and prebuilt RAG and agent
  graphs.

Import from the integration package you need rather than this namespace.
Examples are in `Module-5/GENAI/09_langchain/` and
`Module-5/GENAI/10_langgraph/`.
