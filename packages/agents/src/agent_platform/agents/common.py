"""Agent 构建公共工具：langgraph/langchain 可用时构建 ReAct Agent，否则降级简化 Agent。"""

from __future__ import annotations

from typing import Any

from loguru import logger


def build_react_agent(
    llm_router: Any,
    agent_name: str,
    system_prompt: str,
    tools: list[Any],
) -> Any:
    """返回可被 ainvoke({"messages": [...]}) 调用的 Agent 图。"""
    try:
        from langchain_core.tools import StructuredTool
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent
    except ImportError:  # pragma: no cover
        logger.warning("langchain/langgraph 未安装，{} 使用简化 Agent", agent_name)
        return SimpleAgent(llm_router, agent_name, system_prompt)

    from pydantic import Field, create_model

    from agent_platform.tools.gateway import ToolGateway
    from agent_platform.tools.identity import current_principal_role
    from agent_platform.core.execution import execution_context
    ctx = execution_context.get()
    if ctx and ctx.resources and ctx.resources.get("fixture"):
        from agent_platform.evaluation.tools import fixture_tools
        tools = list(fixture_tools(tools, ctx.resources["fixture"]).values())
    if ctx and ctx.run_id:
        from agent_platform.tools.runtime_tools import runtime_tools
        tools = list(tools) + runtime_tools()
        tools = list({tool.name: tool for tool in tools}.values())
        if ctx.profile.get("tool_policy") == "python_container":
            tools.append(tool_registry_python(ctx))
    if ctx and ctx.permitted_tools is not None:
        tools = [tool for tool in tools if tool.name in ctx.permitted_tools]
    if ctx and ctx.profile.get("tool_policy") == "discover_authorized":
        from agent_platform.tools.discovery import DiscoverToolsTool, InvokeDiscoveredTool
        from agent_platform.tools.base import ToolRegistry
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)
        ctx.permitted_tools = (ctx.permitted_tools or frozenset(registry.names())) | {"discover_authorized_tools", "invoke_authorized_tool"}
        tools = [DiscoverToolsTool(registry), InvokeDiscoveredTool(tools)] + [t for t in tools if t.name in {"request_run_input", "read_run_artifact", "read_skill_resource"}]

    _settings = getattr(llm_router, "settings", None)
    gateway = ToolGateway(
        tools=tools,
        timeout=getattr(_settings, "tool_call_timeout", 30.0) if _settings is not None else 30.0,
    )
    lc_tools = []
    for tool in tools:
        args_model = _schema_to_model(tool.name, tool.args_schema)

        async def _run(tool: Any = tool, **kwargs: Any) -> str:
            # 角色来自接入层认证身份（contextvar），不再写死 user
            result = await gateway.execute(
                tool.name, kwargs, principal_role=current_principal_role()
            )
            import json
            return json.dumps(result, ensure_ascii=False, default=str)

        lc_tools.append(
            StructuredTool.from_function(
                coroutine=_run,
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_schema,
            )
        )

    kwargs = llm_router.settings.llm_kwargs()
    from agent_platform.core.llm.telemetry import ModelTelemetry
    model_name = llm_router.resolve_model(agent_name)
    model = ChatOpenAI(model=model_name, max_retries=0, max_tokens=4096,
                       temperature=ctx.profile.get("temperature", 0) if ctx else 0,
                       callbacks=[ModelTelemetry(model_name)], **kwargs)
    from agent_platform.orchestrator.context import bounded_model_messages
    def context_hook(state):
        return bounded_model_messages(state, prompt=system_prompt, tool_definitions=[tool.manifest() for tool in tools])
    graph = create_react_agent(model, lc_tools, prompt=system_prompt,
                              pre_model_hook=context_hook if ctx and ctx.profile.get("context_policy") == "bounded" else None,
                              checkpointer=True if ctx and ctx.checkpointer else None)
    logger.info("{} 使用 LangGraph ReAct Agent（{} 个工具）", agent_name, len(lc_tools))
    # ChatOpenAI 直连会绕过 LLMRouter 的熔断，这里在图外层补上守卫
    from agent_platform.core.llm.guard import wrap_graph_with_breaker

    return wrap_graph_with_breaker(graph, llm_router)


def tool_registry_python(ctx):
    tool = ctx.resources.get("python_tool")
    if tool is None:
        raise RuntimeError("受限 Python 环境尚未通过验收")
    return tool


def _schema_to_model(name: str, schema: dict[str, Any]):
    """把 JSON Schema 参数定义转成 pydantic 模型（供 StructuredTool 使用）。"""
    from pydantic import Field, create_model

    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for pname, pinfo in props.items():
        ptype = _map_type(pinfo.get("type", "string"))
        if pname in required:
            fields[pname] = (ptype, Field(description=pinfo.get("description", "")))
        else:
            fields[pname] = (ptype | None, Field(default=None, description=pinfo.get("description", "")))
    return create_model(name, **fields)


def _map_type(t: str) -> Any:
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    return str


def pick_tools(tool_registry: Any, names: list[str]) -> list[Any]:
    """从共享注册表按名取工具，缺失的工具跳过（降级容错）。"""
    if tool_registry is None:
        return []
    available = set(tool_registry.names())
    return [tool_registry.get(n) for n in names if n in available]


class SimpleAgent:
    """无 langgraph 时的降级 Agent：系统提示 + LLM 单轮回答。"""

    def __init__(self, llm_router: Any, agent_name: str, system_prompt: str) -> None:
        self.llm_router = llm_router
        self.agent_name = agent_name
        self.system_prompt = system_prompt

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(state.get("messages") or [])
        answer = await self.llm_router.chat(messages, agent_name=self.agent_name)
        return {
            "final_answer": answer,
            "messages": [{"role": "assistant", "content": answer}],
        }
