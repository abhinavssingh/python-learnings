# LangGraph and Stateful Workflows

LangGraph models an application as nodes that read and update shared state, with edges controlling what runs next. It is useful when a workflow needs branching, retries, memory, checkpoints, or human approval rather than a single linear prompt chain.

## Lessons

1. `01_langgraph_basics.py` introduces graph state, reducers, conditional edges, and checkpoints. A small support-ticket workflow can classify a ticket, route it to a specialist node, and append each action to a trace.
2. `02_agentic_rag_graph.py` demonstrates adaptive RAG: route a question, grade retrieved passages, rewrite a query when evidence is poor, and check whether an answer is grounded. Compare a policy question with `"What is 2 + 2?"` and inspect the route and trace.
3. `03_tool_agent_memory_graph.py` builds a tool-using agent with `ToolNode`, separate conversation threads, and a SQLite checkpointer. A `thread_id` lets the graph resume the appropriate conversation.
4. `04_supervisor_multi_agent.py` uses a supervisor to delegate work to researcher, calculator, and writer workers. Observe the accumulated findings and bounded iteration count.
5. `05_human_in_the_loop.py` pauses at `interrupt()` for approval, edits, or rejection, then resumes with `Command(resume=...)`. Set `GENAI_INTERACTIVE=1` to enter decisions yourself; otherwise the demo simulates them.

## Important design lessons

- State is the data contract shared between nodes. Keep it explicit and update only the fields a node owns.
- Conditional edges need clear exit conditions; cap retries and iteration counts to avoid loops.
- Checkpointing is required for durable state and human interruption workflows.
- A graph makes control flow inspectable, but it does not make model decisions reliable by itself. Examine traces and enforce tool boundaries.

## Run

From the repository root:

```powershell
python Module-5/GENAI/10_langgraph/01_langgraph_basics.py
python Module-5/GENAI/10_langgraph/05_human_in_the_loop.py
```

Most lessons use the local chat model and some also build the HR vector store. Follow the [GenAI setup guide](../README.md). Reports are written to this folder's `reports/`; checkpoints may be stored under `saved_models/genai/checkpoints`.

## Practice

Change one routing condition in the basic graph and predict the trace before running it. In the human-review demo, compare approve, edit, and reject paths. Confirm that each path terminates and that resumed state belongs to the same thread.
