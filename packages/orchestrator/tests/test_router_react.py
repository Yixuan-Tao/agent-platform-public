# Test values are synthetic fixtures for software behavior.
"""Router-as-ReAct 编排测试：快速路径 / 复杂度分派 / AgentTool 约束。"""

from __future__ import annotations

import asyncio

import pytest

from agent_platform.core.models import AgentManifest
from agent_platform.orchestrator.agent_tool import AgentTool
from agent_platform.orchestrator.registry import AgentRegistry
from agent_platform.orchestrator.router_react import build_router_graph
from agent_platform.orchestrator.service import OrchestratorService


class FakeLLM:
    def __init__(self, route_to="it_support"):
        self.route_to = route_to
        self.settings = None

    async def chat(self, messages, temperature=0.7, **kw):
        return self.route_to

    def resolve_model(self, agent_name=None):
        return "fake-model"


class FakeGraph:
    async def ainvoke(self, state):
        return {"final_answer": "router-answer", "messages": []}


def _agent_builder(llm_router, tool_registry):
    return FakeGraph()


def _registry() -> AgentRegistry:
    reg = AgentRegistry()
    reg.register(
        AgentManifest(name="it_support", description="IT 支持"), _agent_builder
    )
    reg.register(
        AgentManifest(name="hardware_qa", description="硬件问答"), _agent_builder
    )
    reg.register(
        AgentManifest(name="emc_compare", description="EMC 对比", handoff=["hardware_qa"]),
        _agent_builder,
    )
    return reg


async def test_fast_path_agent_hint(monkeypatch):
    svc = OrchestratorService(
        registry=_registry(), llm_router=FakeLLM("hardware_qa"), tool_registry=None,
        use_router_react=True,
    )
    events = [e async for e in svc.route_and_run_react("s1", "随便问问", agent_hint="it_support")]
    types = [e["type"] for e in events]
    assert "router_start" not in types
    assert "agent_switch" in types
    assert types[-1] == "done"


async def test_fast_path_high_confidence(monkeypatch):
    """'VPN 故障报修' 命中 it_support 多关键词，置信度 >=0.9 走单 Agent。"""
    svc = OrchestratorService(
        registry=_registry(), llm_router=FakeLLM("hardware_qa"), tool_registry=None,
        use_router_react=True,
    )
    events = [e async for e in svc.route_and_run_react("s1", "VPN 故障报修")]
    types = [e["type"] for e in events]
    assert "router_start" not in types
    assert types[-1] == "done"


async def test_simple_goes_router(monkeypatch):
    svc = OrchestratorService(
        registry=_registry(), llm_router=FakeLLM("it_support"), tool_registry=None,
        use_router_react=True,
    )
    monkeypatch.setattr(svc, "_router_graph", lambda: FakeGraph())
    events = [e async for e in svc.route_and_run_react("s1", "帮我看看这个怎么弄")]
    types = [e["type"] for e in events]
    assert "router_start" in types
    assert types[-1] == "done"
    assert events[-1]["agent"] == "router"


async def test_complex_goes_plan(monkeypatch):
    svc = OrchestratorService(
        registry=_registry(), llm_router=FakeLLM("it_support"), tool_registry=None,
        use_router_react=True,
    )
    monkeypatch.setattr(svc, "_plan_graph", lambda: FakeGraph())
    events = [e async for e in svc.route_and_run_react("s1", "对比并分析 A 与 B 的硬件方案")]
    types = [e["type"] for e in events]
    assert "plan_start" in types
    assert types[-1] == "done"
    assert events[-1]["agent"] == "plan_execute"


async def test_no_agents_error():
    svc = OrchestratorService(
        registry=AgentRegistry(), llm_router=FakeLLM(), tool_registry=None,
        use_router_react=True,
    )
    events = [e async for e in svc.route_and_run_react("s1", "hi")]
    assert events[0]["type"] == "error"


class FakeToolGraph:
    """ainvoke 时调用指定 AgentTool（模拟子 Agent 内部再调度）。"""

    def __init__(self, tool):
        self._tool = tool

    async def ainvoke(self, state):
        messages = state.get("messages") or []
        task = messages[-1]["content"] if messages else ""
        return {"final_answer": await self._tool.execute(task=task), "messages": []}


async def test_agent_tool_cycle_guard():
    reg = AgentRegistry()
    tool_a: AgentTool | None = None

    def builder_a(llm_router, tool_registry):
        return FakeToolGraph(tool_a)

    reg.register(AgentManifest(name="a", description="A"), builder_a)
    tool_a = AgentTool("a", reg, FakeLLM(), None, max_depth=3)
    result = await tool_a.execute(task="do")
    assert "禁止循环调用" in result


async def test_agent_tool_depth_limit():
    reg = AgentRegistry()
    tool_a: AgentTool | None = None
    tool_b: AgentTool | None = None

    def builder_a(llm_router, tool_registry):
        return FakeToolGraph(tool_b)

    def builder_b(llm_router, tool_registry):
        return FakeToolGraph(tool_a)

    reg.register(AgentManifest(name="a", description="A"), builder_a)
    reg.register(AgentManifest(name="b", description="B"), builder_b)
    tool_a = AgentTool("a", reg, FakeLLM(), None, max_depth=1)
    tool_b = AgentTool("b", reg, FakeLLM(), None, max_depth=1)
    result = await tool_a.execute(task="do")
    assert "深度上限" in result


async def test_agent_tool_concurrency_isolated():
    reg = AgentRegistry()

    def builder_a(llm_router, tool_registry):
        return FakeGraph()

    reg.register(AgentManifest(name="a", description="A"), builder_a)
    reg.register(AgentManifest(name="b", description="B"), builder_a)
    tool_a = AgentTool("a", reg, FakeLLM(), None, max_depth=3)
    tool_b = AgentTool("b", reg, FakeLLM(), None, max_depth=3)
    results = await asyncio.gather(tool_a.execute(task="x"), tool_b.execute(task="y"))
    assert results == ["router-answer", "router-answer"]


async def test_build_router_graph_with_handoff():
    reg = AgentRegistry()
    reg.register(
        AgentManifest(name="a", description="A", handoff=["b"]), _agent_builder
    )
    reg.register(AgentManifest(name="b", description="B"), _agent_builder)
    graph = build_router_graph(reg, FakeLLM(), None)
    assert graph.agent_tool_names == ["agent__a"]


def test_complexity_judge_rules():
    from agent_platform.orchestrator.planner_execute import ComplexityJudge

    judge = ComplexityJudge()

    async def judge_text(text: str) -> str:
        return await judge.judge(text)

    assert asyncio.run(judge_text("x" * 300)) == "complex"  # 超长
    assert asyncio.run(judge_text("VPN 故障 硬件参数 怎么查")) == "complex"  # 多领域
    assert asyncio.run(judge_text("对比两份标准")) == "complex"  # 复杂指令词
    assert asyncio.run(judge_text("打印机卡纸怎么办")) == "simple"
