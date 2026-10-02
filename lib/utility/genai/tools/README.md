# Tools and Agents

Define callable tools and run tool-using agents. `tool` decorates a function,
`Tool` and `ToolRegistry` represent and organize tools, and `ToolCallingAgent`
provides a model-driven tool loop. Built-in examples include a calculator,
date/time, word count, and document search.

```python
from lib.utility.genai.tools import calculator, safe_eval
```

`safe_eval` is intended for constrained arithmetic expressions; do not treat
it as a general-purpose Python execution environment. For LangChain tools and
LangGraph agents, see `frameworks/langchain/` and `frameworks/langgraph/`.
Runnable examples are in `Module-5/GENAI/03_llm_basics/`.
