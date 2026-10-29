"""LangGraph Supervisor：ROUTE → SUPERVISOR(LLM) → Agent 执行 → CHECKER。

langgraph 未安装时 raise RuntimeError，其余模块不受影响。
"""

from __future__ import annotations

from typing import Any

try:
    from langgraph.graph import END, StateGraph
    from langgraph.types import Command

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    LANGGRAPH_AVAILABLE = False

from loguru import logger

from agent_platform.orchestrator.registry import AgentRegistry
from agent_platform.orchestrator.router import SemanticRouter
from agent_platform.orchestrator.state import AgentState


def build_supervisor(
    registry: AgentRegistry,
    llm_router: Any,
    tool_registry: Any,
    checkpointer: Any = None,
    max_attempts: int = 2,
) -> Any:
    """构建并编译 Supervisor 图。

    节点：
    - ROUTE：关键词语义路由，命中则直达对应 Agent，否则交 SUPERVISOR。
    - SUPERVISOR：LLM 从注册 Agent 中选一个，通过 Command(goto=...) 移交。
    - AGENT_EXEC：运行当前 active_agent 的图，产出 final_answer。
    - CHECKER：反思校验，失败且未超次数则回到 AGENT_EXEC，否则结束。
    """
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph 未安装，无法构建 Supervisor")

    router = SemanticRouter()
    from agent_platform.orchestrator.checker import Checker

    checker = Checker(llm_router)

    async def route_node(state: AgentState) -> dict[str, Any]:
        text = state["messages"][-1]["content"] if state.get("messages") else ""
        agent, confidence = router.route(text)
        if agent and confidence >= 0.5 and registry.has(agent):
            return Command(goto=agent, update={"active_agent": agent})
        return Command(goto="supervisor")

    async def supervisor_node(state: AgentState) -> dict[str, Any]:
        text = state["messages"][-1]["content"] if state.get("messages") else ""
        agents = registry.names()
        descriptions = {m.name: m.description for m in registry.list_manifests()}
        chosen = await router.ensure_llm_route(text, llm_router, agents, descriptions)
        if chosen is None:
            chosen = agents[0] if agents else "unknown"
        logger.info("Supervisor 选择 Agent: {}", chosen)
        return Command(goto=chosen, update={"active_agent": chosen})

    async def agent_node(state: AgentState) -> dict[str, Any]:
        agent_name = state.get("active_agent", "")
        if not registry.has(agent_name):
            return {"errors": [f"Agent 不可用: {agent_name}"], "final_answer": "", "done": True}
        graph = registry.build(agent_name, llm_router, tool_registry)
        answer = await _invoke_agent(graph, state["messages"])
        attempts = int(state.get("attempts", 0)) + 1
        return {"final_answer": answer, "attempts": attempts}

    async def checker_node(state: AgentState) -> dict[str, Any]:
        task = state["messages"][0]["content"] if state.get("messages") else ""
        answer = state.get("final_answer", "")
        passed, reason = await checker.check(task, answer)
        if passed or int(state.get("attempts", 0)) >= max_attempts:
            return {"done": True}
        return Command(goto="agent_exec", update={"errors": [reason], "task_plan": reason})

    builder = StateGraph(AgentState)
    builder.add_node("route", route_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("agent_exec", agent_node)
    builder.add_node("checker", checker_node)

    builder.set_entry_point("route")
    builder.add_edge("route", "supervisor")
    builder.add_edge("supervisor", "agent_exec")
    builder.add_edge("agent_exec", "checker")
    builder.add_edge("checker", END)

    graph = builder.compile(checkpointer=checkpointer)
    return graph


async def _invoke_agent(graph: Any, messages: list[Any]) -> str:
    """调用 Agent 图并提取最终答案，兼容 dict/字符串返回。"""
    from agent_platform.core.execution import execution_context
    ctx = execution_context.get()
    state = {"messages": messages}
    if ctx:
        state["original_goal"] = ctx.profile.get("original_goal", "")
    if hasattr(graph, "ainvoke"):
        import inspect
        parameters = inspect.signature(graph.ainvoke).parameters
        accepts_config = "config" in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values())
        config = ctx.graph_config if ctx else None
        if ctx:
            from langgraph.config import get_config
            try:
                config = get_config()  # retain parent namespace and inherited checkpointer
            except RuntimeError:
                pass
        result = await graph.ainvoke(state, **({"config": config} if ctx and accepts_config else {}))
    else:
        result = await graph(messages)
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if result.get("__interrupt__"):
            from langgraph.errors import GraphInterrupt
            raise GraphInterrupt(result["__interrupt__"])
        if result.get("final_answer"):
            return str(result["final_answer"])
        msgs = result.get("messages") or []
        if msgs:
            last = msgs[-1]
            content = last.content if hasattr(last, "content") else last.get("content", "")
            if content:
                return str(content)
    return str(result)
