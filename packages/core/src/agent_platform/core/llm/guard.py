"""熔断保护包装：让直接构建 LangChain/ LangGraph 模型的 Agent 也受 LLMRouter 熔断约束。

专家 Agent 与 Router 图内部使用 ``ChatOpenAI``（为了复用 LangChain 的工具绑定
能力），这条路径绕过了 :class:`LLMRouter` 自带的重试与熔断。这里在图的外层
加一层守卫，使连续失败同样能触发熔断，避免故障时持续打爆下游。
"""

from __future__ import annotations

from typing import Any


class GuardedGraph:
    """透明代理：透传属性，调用前检查熔断、调用后记录结果。"""

    def __init__(self, graph: Any, llm_router: Any) -> None:
        self._graph = graph
        self._llm_router = llm_router

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        async with self._llm_router.breaker_guard():
            return await self._graph.ainvoke(*args, **kwargs)

    async def astream(self, *args: Any, **kwargs: Any):
        async with self._llm_router.breaker_guard():
            async for item in self._graph.astream(*args, **kwargs):
                yield item


def wrap_graph_with_breaker(graph: Any, llm_router: Any) -> Any:
    """llm_router 提供 ``breaker_guard`` 时包装，否则原样返回（便于测试替身）。"""
    if graph is None or not hasattr(llm_router, "breaker_guard"):
        return graph
    if isinstance(graph, GuardedGraph):
        return graph
    return GuardedGraph(graph, llm_router)


__all__ = ["GuardedGraph", "wrap_graph_with_breaker"]
