"""Router-as-ReAct：主 Agent 以工具方式自主调度多个子 Agent。"""

from __future__ import annotations

from typing import Any

from loguru import logger

try:
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    LANGGRAPH_AVAILABLE = False

ROUTER_PROMPT = (
    "你是多 Agent 平台的主调度 Agent。你可以调用以下子 Agent 工具完成用户请求：\n"
    "1) 自主判断是否需要拆分子任务、是否需要调用多个子 Agent；\n"
    "2) 调用子 Agent 前明确传入任务描述；\n"
    "3) 必须整合所有子 Agent 的结果，输出最终回答（含关键结论与必要说明）。\n"
    "不要虚构子 Agent 结果；如果子 Agent 返回失败，请如实说明。"
)


def build_router_graph(
    registry: Any,
    llm_router: Any,
    tool_registry: Any,
    max_agent_calls: int = 4,
) -> Any:
    """构建 Router ReAct 图：只包装 handoff 非空的 Agent 为 AgentTool。"""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph 未安装，无法构建 Router 图")
    from pydantic import Field, create_model

    from agent_platform.orchestrator.agent_tool import AgentTool

    tools = []
    for manifest in registry.list_manifests():
        if not manifest.handoff:
            continue
        tool = AgentTool(
            manifest.name,
            registry,
            llm_router,
            tool_registry,
            max_depth=3,
            max_calls=max_agent_calls,
        )
        args_model = create_model(
            f"AgentTool_{manifest.name}",
            task=(str, Field(description=manifest.input_hint or "任务描述")),
        )

        async def _run(task: str = "", tool: Any = tool) -> str:
            from agent_platform.core.execution import execution_context
            ctx = execution_context.get()
            handoff = {}
            if ctx and ctx.run_id:
                current = await ctx.resources["store"].get(ctx.run_id)
                handoff = {"original_goal": ctx.profile.get("original_goal"), "authorization": sorted(ctx.tool_scopes),
                           "previous_results": [c.get("result", {}) for c in current["tool_calls"]]}
            return await tool.execute(task=task, **handoff)

        tools.append(
            StructuredTool.from_function(
                coroutine=_run,
                name=tool.name,
                description=manifest.description,
                args_schema=args_model,
            )
        )
    if not tools:
        raise RuntimeError("没有可调度的子 Agent（所有 handoff 均为空）")
    settings = getattr(llm_router, "settings", None)
    kwargs = (
        settings.llm_kwargs()
        if settings is not None
        else {"api_key": "sk-none", "base_url": "https://api.openai.com/v1"}
    )
    from agent_platform.core.execution import execution_context
    from agent_platform.core.llm.telemetry import ModelTelemetry
    ctx = execution_context.get()
    model_name = llm_router.resolve_model("router")
    model = ChatOpenAI(model=model_name, temperature=0, max_tokens=4096, max_retries=0,
                       callbacks=[ModelTelemetry(model_name)], **kwargs)
    from agent_platform.orchestrator.context import bounded_model_messages
    def context_hook(state):
        return bounded_model_messages(state, prompt=ROUTER_PROMPT,
                                      tool_definitions=[{"name": t.name, "parameters": t.args_schema.model_json_schema()} for t in tools])
    graph = create_react_agent(model, tools, prompt=ROUTER_PROMPT,
                              pre_model_hook=context_hook if ctx and ctx.profile.get("context_policy") == "bounded" else None,
                              checkpointer=True if ctx and ctx.checkpointer else None)
    graph.agent_tool_names = [t.name for t in tools]
    logger.info("Router 图构建完成，可调度子 Agent: {}", [t.name for t in tools])
    from agent_platform.core.llm.guard import wrap_graph_with_breaker

    return wrap_graph_with_breaker(graph, llm_router)
