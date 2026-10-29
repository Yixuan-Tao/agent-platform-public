# Test values are synthetic fixtures for software behavior.
from __future__ import annotations

from agent_platform.core.models import AgentManifest
from agent_platform.orchestrator.registry import AgentRegistry
from agent_platform.orchestrator.service import OrchestratorService


class FakeLLM:
    def __init__(self, route_to="it_support"):
        self.route_to = route_to

    async def chat(self, messages, temperature=0.7, **kw):
        return self.route_to


class FakeGraph:
    async def ainvoke(self, state):
        FakeGraph.recorded = state
        return {"final_answer": "这是测试回答内容", "messages": []}


class FakeLongTerm:
    def __init__(self, hits=None, raise_on_search=False):
        self.hits = hits or []
        self.raise_on_search = raise_on_search

    def search_filtered(self, query, *, owner="", k=3, min_score=0.5, max_tokens=500):
        if self.raise_on_search:
            raise RuntimeError("长期记忆检索失败")
        return self.hits


class FakeMemory:
    few_shot_enabled = True
    few_shot_k = 3
    few_shot_min_score = 0.5
    few_shot_max_tokens = 500

    def __init__(self, long_term=None, enabled=True):
        self.long_term = long_term or FakeLongTerm()
        self.short_term = None
        self.few_shot_enabled = enabled


def _agent_builder(llm_router, tool_registry):
    return FakeGraph()


async def test_route_and_run_events():
    reg = AgentRegistry()
    reg.register(
        AgentManifest(name="it_support", description="IT 支持"), _agent_builder
    )
    reg.register(
        AgentManifest(name="hardware_qa", description="硬件问答"), _agent_builder
    )
    svc = OrchestratorService(
        registry=reg, llm_router=FakeLLM("hardware_qa"), tool_registry=None
    )
    events = [e async for e in svc.route_and_run("s1", "VPN 故障报修")]
    types = [e["type"] for e in events]
    assert "agent_switch" in types
    assert types[-1] == "done"
    switch = next(e for e in events if e["type"] == "agent_switch")
    # "VPN/故障/报修" 命中 it_support 关键词，语义路由优先于 LLM 兜底
    assert switch["agent"] == "it_support"
    done = events[-1]
    assert "测试回答" in done["answer"]


async def test_no_agents_error():
    svc = OrchestratorService(registry=AgentRegistry(), llm_router=FakeLLM(), tool_registry=None)
    events = [e async for e in svc.route_and_run("s1", "hi")]
    assert events[0]["type"] == "error"


async def test_route_and_run_few_shot_injection():
    """命中长期记忆时注入 few-shot，主流程不中断。"""
    reg = AgentRegistry()
    reg.register(
        AgentManifest(name="it_support", description="IT 支持"), _agent_builder
    )
    mem = FakeMemory(
        FakeLongTerm(hits=[{"question": "VPN 怎么连", "answer": "按手册配置", "score": 0.9}])
    )
    svc = OrchestratorService(
        registry=reg, llm_router=FakeLLM("it_support"), tool_registry=None, memory_service=mem
    )
    events = [e async for e in svc.route_and_run("s1", "VPN 故障报修")]
    assert events[-1]["type"] == "done"
    assert "测试回答" in events[-1]["answer"]
    system_msgs = [
        m["content"]
        for m in FakeGraph.recorded.get("messages", [])
        if m.get("role") == "system"
    ]
    assert any("参考历史优质问答" in content for content in system_msgs)


async def test_route_and_run_few_shot_degradation():
    """检索异常/关闭时降级为空，不影响对话链路。"""
    reg = AgentRegistry()
    reg.register(
        AgentManifest(name="it_support", description="IT 支持"), _agent_builder
    )
    svc_broken = OrchestratorService(
        registry=reg,
        llm_router=FakeLLM("it_support"),
        tool_registry=None,
        memory_service=FakeMemory(FakeLongTerm(raise_on_search=True)),
    )
    events_broken = [e async for e in svc_broken.route_and_run("s1", "VPN 故障")]
    assert events_broken[-1]["type"] == "done"

    svc_disabled = OrchestratorService(
        registry=reg,
        llm_router=FakeLLM("it_support"),
        tool_registry=None,
        memory_service=FakeMemory(enabled=False),
    )
    events_disabled = [e async for e in svc_disabled.route_and_run("s1", "VPN 故障")]
    assert events_disabled[-1]["type"] == "done"
