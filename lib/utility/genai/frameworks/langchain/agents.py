import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool


def run_tool_calling_loop(llm: BaseChatModel, tools: list[BaseTool], question: str, max_steps: int = 5,
                          system_prompt: str = "You are a helpful assistant. Use tools when useful; always use the calculator for arithmetic.") -> tuple[str, list[dict[str, Any]]]:
    """
    Manual agent loop with LangChain primitives:
        llm.bind_tools(tools) -> AIMessage.tool_calls -> tool.invoke -> ToolMessage -> repeat
    """
    by_name = {t.name: t for t in tools}
    model = llm.bind_tools(tools)
    messages = [SystemMessage(system_prompt), HumanMessage(question)]
    trace: list[dict[str, Any]] = []
    for step in range(1, max_steps + 1):
        ai = model.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            trace.append({"step": step, "type": "final_answer", "content": ai.content})
            return ai.content, trace
        for call in ai.tool_calls:
            output = by_name[call["name"]].invoke(call["args"])
            trace.append({"step": step, "type": "tool_call", "content": f"{call['name']}({json.dumps(call['args'])}) -> {str(output)[:200]}"})
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return "Stopped after max_steps.", trace
