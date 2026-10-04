# LangGraph and Stateful Workflows

LangGraph models an application as nodes that read and update shared state, with edges controlling what runs next. It is useful when a workflow needs branching, retries, memory, checkpoints, or human approval rather than a single linear prompt chain.

## Why stateful workflows matter

A simple prompt chain is often enough for a single ask. But real AI systems often need to:
- decide which tool or step to run next
- retry after a failed retrieval
- maintain conversation memory
- ask for human approval before acting
- track intermediate results across many nodes

Example workflow:

```text
User question -> route to retrieval or calculator -> generate answer -> check if grounded -> ask human for approval -> finalize
```

This is not a single linear call. It is a graph of states and transitions.

## Core LangGraph concepts

### State
State is a shared object that each node can read and update.

Example:

```python
state = {
    "question": "...",
    "retrieved_docs": [],
    "answer": None,
    "status": "pending"
}
```

Each node updates a specific part of that state.

### Nodes
A node is a function or step in the graph.

Example nodes:
- classify question
- retrieve relevant passages
- generate an answer
- validate evidence
- ask for human approval

### Conditional edges
Edges decide which node runs next.

Example:

```text
If evidence is weak -> rewrite the query
Else -> generate final answer
```

This makes the workflow adaptive.

### Checkpointing
Checkpointing stores the state so the workflow can resume later.

This matters for:
- long-running flows
- human-in-the-loop approval
- multi-turn conversation recovery

## Lessons

### 1. LangGraph basics (`01_langgraph_basics.py`)

This lesson introduces:
- graph state
- reducers
- conditional edges
- checkpoints

Example support-ticket workflow:

```text
Input: "My manager needs approval for travel reimbursement."
```

A support workflow may:
1. classify the request
2. route to a travel node
3. append the step to a trace
4. return a final decision

This is a clear example of stateful orchestration.

### 2. Adaptive RAG graph (`02_agentic_rag_graph.py`)

This demonstrates more advanced behavior:
- route the question
- grade the retrieved passages
- rewrite the query if needed
- check whether the answer is grounded

Example:

```text
Question: "What is 2 + 2?"
```

This should route to a calculator or a simple answer node, not to document retrieval.

But a policy question such as:

```text
"How does the company support gender balance?"
```

should route to retrieval and evidence checking.

This is a good example of routing logic inside a graph-based app.

### 3. Tool-using agent memory graph (`03_tool_agent_memory_graph.py`)

This adds:
- a tool node
- conversation threads
- SQLite checkpointer

The graph can maintain separate conversation state per `thread_id`.

Example:

```text
Thread A: employee leave question
Thread B: payroll question
```

The system should keep each thread separate so the correct state is resumed later.

### 4. Supervisor multi-agent workflow (`04_supervisor_multi_agent.py`)

This uses a supervisor to delegate work to specialist workers such as:
- researcher
- calculator
- writer

Example:

```text
User asks: "Compare the annual leave policy with the sick leave policy and summarize the difference."
```

The supervisor may:
1. ask a researcher to fetch relevant policy passages
2. ask a writer to summarize the difference
3. call a calculator only if numerical data is needed

This pattern is useful when one model is not enough to manage all tasks.

### 5. Human-in-the-loop graph (`05_human_in_the_loop.py`)

This introduces `interrupt()` for approval or review.

Example:

```text
The system proposes a final policy answer.
Human reviewer can approve, edit, or reject it.
```

Then the graph resumes with `Command(resume=...)`.

This is valuable when actions have real consequences, such as:
- approving a workflow step
- reviewing generated text
- confirming a policy decision

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

### Example exercise

Create a simple graph:

```text
Question -> classify intent -> if leave policy -> retrieve docs -> answer
if math question -> calculator -> answer
```

Then test:
- a leave policy question
- a math question
- an invalid question with no clear route

This shows how routing and state management change the workflow outcome.

## Big picture

LangGraph is about workflows, not just prompts.

```text
single prompt -> simple app
state + routing + tools + checkpoints -> agentic workflow
```

This is the foundation for more advanced real-world AI systems where control flow matters as much as model quality.
