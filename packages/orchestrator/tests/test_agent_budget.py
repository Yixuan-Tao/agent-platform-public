# Test values are synthetic fixtures for software behavior.
"""子 Agent 调用预算与工具角色透传测试。"""

from __future__ import annotations

import asyncio

from agent_platform.orchestrator.agent_tool import (
    AgentCallCounter,
    AgentTool,
    agent_calls_used,
    configure_agent_budget,
    reset_agent_budget,
)
from agent_platform.orchestrator.registry import AgentRegistry
from agent_platform.core.models import AgentManifest


class FakeGraph:
    async def ainvoke(self, state):
        return {"final_answer": "ok", "messages": []}


class FakeLLM:
    settings = None

    async def chat(self, messages, temperature=0.7, **kw):
        return "PASS"


def _registry() -> AgentRegistry:
    reg = AgentRegistry()
    for name in ("a", "b", "c", "d"):
        reg.register(AgentManifest(name=name, description=name), lambda *a, **k: FakeGraph())
    return reg


async def test_tool_level_max_calls_blocks_after_budget():
    tool = AgentTool("a", _registry(), FakeLLM(), None, max_depth=5, max_calls=2)
    results = [await tool.execute(task="x") for _ in range(4)]
    assert results[:2] == ["ok", "ok"]
    assert "预算上限" in results[2]
    assert "预算上限" in results[3]


async def test_request_budget_is_shared_across_sibling_tools():
    """同一请求内的多个 AgentTool 共享预算，而不是各自计数。"""
    reg = _registry()
    configure_agent_budget(2)
    try:
        tool_a = AgentTool("a", reg, FakeLLM(), None, max_depth=5)
        tool_b = AgentTool("b", reg, FakeLLM(), None, max_depth=5)
        tool_c = AgentTool("c", reg, FakeLLM(), None, max_depth=5)
        outcomes = [
            await tool_a.execute(task="1"),
            await tool_b.execute(task="2"),
            await tool_c.execute(task="3"),
        ]
    finally:
        reset_agent_budget()
    assert outcomes[0] == "ok"
    assert outcomes[1] == "ok"
    assert "预算上限" in outcomes[2]


async def test_request_budget_shared_across_concurrent_tasks():
    """contextvars 复制的是引用，并发分支也应共享同一计数器。"""
    reg = _registry()
    configure_agent_budget(2)
    try:
        tools = [
            AgentTool(name, reg, FakeLLM(), None, max_depth=5)
            for name in ("a", "b", "c", "d")
        ]
        outcomes = await asyncio.gather(*(t.execute(task=str(i)) for i, t in enumerate(tools)))
    finally:
        reset_agent_budget()
    assert sum(1 for o in outcomes if o == "ok") == 2
    assert sum(1 for o in outcomes if "预算上限" in o) == 2


async def test_reset_clears_counter_between_requests():
    reg = _registry()
    tool = AgentTool("a", reg, FakeLLM(), None, max_depth=5)
    configure_agent_budget(1)
    assert await tool.execute(task="1") == "ok"
    assert "预算上限" in await tool.execute(task="2")
    reset_agent_budget()
    configure_agent_budget(1)
    assert await tool.execute(task="3") == "ok"
    reset_agent_budget()
    assert agent_calls_used() == 0


def test_counter_without_limit_never_blocks():
    counter = AgentCallCounter(0)
    assert all(counter.try_acquire() for _ in range(100))
    assert counter.used == 100
