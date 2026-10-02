from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..state import AgentState

DEFAULT_SYSTEM = ("You are a helpful HR assistant. Use the tools when they help: search the HR policy for company "
                  "questions and always use the calculator for arithmetic. Be concise.")


def build_tool_agent_graph(llm: BaseChatModel, tools: list[BaseTool], system_prompt: str = DEFAULT_SYSTEM,
                           checkpointer=None):
    """
    ReAct agent as an explicit graph:

        START -> agent --(tool calls?)--> tools -> agent ... --(no tool calls)--> END

    With a checkpointer, every `thread_id` keeps its own conversation memory.
    """
    model = llm.bind_tools(tools)

    def agent(state: AgentState) -> dict:
        response = model.invoke([SystemMessage(system_prompt)] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
