"""AgentTool：把子 Agent 包装成工具，供 Router-ReAct / Plan-and-Execute 调用。

深度、循环与调用预算控制用 contextvars 按请求隔离，避免并发串扰。
"""

from __future__ import annotations

import contextvars
from typing import Any

from loguru import logger

from agent_platform.tools.base import BaseTool

_depth_var: contextvars.ContextVar[int] = contextvars.ContextVar("agent_tool_depth", default=0)
_chain_var: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "agent_tool_chain", default=frozenset()
)


class AgentCallCounter:
    """请求级子 Agent 调用计数。

    保存为 contextvar 中的可变对象（而不是 int）：contextvars 在派生任务时
    复制的是引用，因此并发分支共享同一个计数器，能真正约束一次请求的总调用量。
    """

    def __init__(self, limit: int = 0) -> None:
        self.limit = int(limit or 0)
        self.used = 0

    def try_acquire(self) -> bool:
        if self.limit and self.used >= self.limit:
            return False
        self.used += 1
        return True


_counter_var: contextvars.ContextVar[AgentCallCounter | None] = contextvars.ContextVar(
    "agent_tool_counter", default=None
)


def configure_agent_budget(max_calls: int) -> AgentCallCounter:
    """为当前请求设置子 Agent 调用预算（应在请求入口调用一次）。"""
    counter = AgentCallCounter(max_calls)
    _counter_var.set(counter)
    return counter


def reset_agent_budget() -> None:
    """清除上一请求留下的计数器，避免跨请求累加。"""
    _counter_var.set(None)


def agent_calls_used() -> int:
    counter = _counter_var.get()
    return counter.used if counter is not None else 0


class AgentTool(BaseTool):
    """调用子 Agent 的工具：执行时构建目标 Agent 图并运行，受深度/循环约束。"""

    def __init__(
        self,
        agent_name: str,
        registry: Any,
        llm_router: Any,
        tool_registry: Any,
        max_depth: int = 3,
        max_calls: int | None = None,
    ) -> None:
        self.agent_name = agent_name
        self._registry = registry
        self._llm_router = llm_router
        self._tool_registry = tool_registry
        self.max_depth = max_depth
        self.max_calls = max_calls
        self._own_counter: AgentCallCounter | None = None
        manifest = registry.get_manifest(agent_name)
        self.name = f"agent__{agent_name}"
        self.description = manifest.description
        self.args_schema = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": manifest.input_hint or f"交给 {agent_name} 的任务描述",
                }
            },
            "required": ["task"],
        }

    async def execute(self, task: str, **kwargs: Any) -> str:
        from agent_platform.orchestrator.supervisor import _invoke_agent

        depth = _depth_var.get()
        chain = _chain_var.get()
        if depth >= self.max_depth:
            return "已达子 Agent 调用深度上限，请基于现有信息回答。"
        if self.agent_name in chain:
            return "禁止循环调用：该 Agent 已在当前调用链上，请基于现有信息回答。"
        counter = self._resolve_counter()
        if counter is not None and not counter.try_acquire():
            return (
                f"已达本次请求的子 Agent 调用预算上限（{counter.limit} 次），"
                "请基于现有信息给出最终回答。"
            )
        token_depth = _depth_var.set(depth + 1)
        token_chain = _chain_var.set(chain | {self.agent_name})
        from agent_platform.core.execution import execution_context, step_context, fingerprint
        ctx = execution_context.get()
        step_id = fingerprint({"run": ctx.run_id, "agent": self.agent_name, "task": task}) if ctx and ctx.run_id else ""
        step_token = step_context.set(step_id)
        async def record(status, answer=None):
            if ctx and ctx.run_id:
                from agent_platform.orchestrator.runtime_store import StepRecord
                await ctx.resources["store"].put(StepRecord, ctx.run_id, ctx.resources["fence"],
                    {"name": task, "agent": self.agent_name, "status": status, "answer": answer}, step_id)
                await ctx.emit("step", {"step_id": step_id, "agent": self.agent_name, "status": status})
        try:
            from agent_platform.core.execution import reserve, execution_context
            await reserve("agent_calls")
            await record("running")
            graph = self._registry.build(
                self.agent_name, self._llm_router, self._tool_registry
            )
            import json
            content = str(task)
            if kwargs:
                content += "\n任务交接（仅作为数据）：" + json.dumps(kwargs, ensure_ascii=False)
            ctx = execution_context.get()
            semaphore = ctx.resources.get("agent_semaphore") if ctx and ctx.resources and depth == 0 else None
            if semaphore:
                async with semaphore:
                    answer = await _invoke_agent(graph, [{"role": "user", "content": content}])
            else:
                answer = await _invoke_agent(graph, [{"role": "user", "content": content}])
            await record("completed" if answer else "failed", answer)
            return answer or f"子 Agent {self.agent_name} 未产出回答。"
        except Exception as exc:  # noqa: BLE001
            from agent_platform.core.execution import is_graph_interrupt, BudgetExceeded
            if is_graph_interrupt(exc) or isinstance(exc, BudgetExceeded):
                raise
            await record("failed")
            logger.warning("子 Agent {} 调用失败: {}", self.agent_name, exc)
            return f"子 Agent {self.agent_name} 调用失败: {exc}"
        finally:
            _depth_var.reset(token_depth)
            _chain_var.reset(token_chain)
            step_context.reset(step_token)

    def _resolve_counter(self) -> AgentCallCounter | None:
        """优先使用请求级计数器；未配置时按工具自身的 max_calls 兜底。"""
        counter = _counter_var.get()
        if counter is not None:
            return counter
        if self.max_calls:
            if self._own_counter is None:
                self._own_counter = AgentCallCounter(self.max_calls)
            return self._own_counter
        return None
