# Test values are synthetic fixtures for software behavior.
"""Plan-and-Execute 图测试：计划生成 / 逐步执行 / REPLAN / 步数上限。"""

from __future__ import annotations

import pytest

from agent_platform.core.models import AgentManifest
from agent_platform.orchestrator.planner_execute import build_plan_execute_graph
from agent_platform.orchestrator.registry import AgentRegistry


class FakeLLM:
    def __init__(self, fail_step: bool = False):
        self.settings = None
        self.fail_step = fail_step

    def resolve_model(self, agent_name=None):
        return "fake-model"

    async def chat(self, messages, temperature=0.7, **kw):
        text = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        if '"step_results"' in text:
            import json
            return "综合结果：\n" + "\n".join(json.loads(text)["step_results"])
        if "上一步执行异常" in text:
            return '{"steps": ["新步骤"]}'
        if "任务规划器" in text:
            return '{"steps": ["对比参数", "生成报告", "输出结论", "s4", "s5", "s6", "s7", "s8"]}'
        if "任务分发器" in text:
            return "a"
        return "ok"


class FakeGraph:
    def __init__(self, name: str, fail: bool = False):
        self.name = name
        self.fail = fail

    async def ainvoke(self, state):
        if self.fail:
            return {"final_answer": "步骤执行调用失败，请重试", "messages": []}
        return {"final_answer": f"answer-{self.name}", "messages": []}


class FailOnceGraph:
    """首次执行失败、后续成功（用于验证 REPLAN 改写剩余步骤）。"""

    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    async def ainvoke(self, state):
        self.calls += 1
        if self.calls == 1:
            return {"final_answer": "步骤执行调用失败，请重试", "messages": []}
        return {"final_answer": f"answer-{self.name}", "messages": []}


def _registry(fail_a: bool = False) -> AgentRegistry:
    reg = AgentRegistry()
    reg.register(
        AgentManifest(name="a", description="A", handoff=["b"]),
        lambda llm, tools: FakeGraph("a", fail=fail_a),
    )
    reg.register(
        AgentManifest(name="b", description="B"),
        lambda llm, tools: FakeGraph("b"),
    )
    return reg


async def test_plan_execute_full_run():
    reg = _registry()
    graph = build_plan_execute_graph(reg, FakeLLM(), None, max_plan_steps=6)
    result = await graph.ainvoke({"messages": [{"role": "user", "content": "复杂任务"}]})
    assert result["done"] is True
    assert result["current_step"] == 6  # 8 步被截断到 6
    assert len(result["steps"]) == 6
    assert "【步骤1/6】对比参数" in result["final_answer"]
    assert "answer-a" in result["final_answer"]


async def test_plan_execute_replan():
    reg = AgentRegistry()
    reg.register(
        AgentManifest(name="a", description="A", handoff=["b"]),
        lambda llm, tools: FailOnceGraph("a"),
    )
    reg.register(
        AgentManifest(name="b", description="B"),
        lambda llm, tools: FakeGraph("b"),
    )
    graph = build_plan_execute_graph(reg, FakeLLM(), None, max_plan_steps=6)
    result = await graph.ainvoke({"messages": [{"role": "user", "content": "复杂任务"}]})
    # 第一步失败 → REPLAN 把剩余步骤改写为 ["新步骤"]
    assert result["steps"][-1] == "新步骤"
    assert "answer-a" in result["final_answer"]


async def test_plan_graph_no_handoff_raises():
    reg = AgentRegistry()
    reg.register(
        AgentManifest(name="a", description="A"),
        lambda llm, tools: FakeGraph("a"),
    )
    with pytest.raises(RuntimeError):
        build_plan_execute_graph(reg, FakeLLM(), None)

async def test_original_goal_and_previous_real_evidence_reach_second_step():
    import json
    received, planned = [], []
    class Router(FakeLLM):
        async def chat(self, messages, **kwargs):
            text = messages[-1].content
            if "任务规划器" in text:
                planned.append(text)
                return '{"steps":["检索真实证据","依据证据生成报告"]}'
            return await super().chat(messages, **kwargs)
    class Graph:
        async def ainvoke(self, state):
            received.append(state["messages"][0]["content"])
            return {"final_answer": "真实证据 30 dBµV/m 来源A:2026"}
    reg = AgentRegistry()
    reg.register(AgentManifest(name="a", description="A", handoff=["b"]), lambda *_: Graph())
    graph = build_plan_execute_graph(reg, Router(), None)
    result = await graph.ainvoke({"original_goal": "当前 EMC 对比", "messages": [{"role":"user","content":"当前 EMC 对比"}, {"role":"system","content":"历史示例：创建IT工单"}]})
    assert "当前 EMC 对比" in planned[0] and "历史示例" not in planned[0]
    assert "真实证据 30 dBµV/m 来源A:2026" in received[1]
    assert result["step_outputs"][0]["answer"] == "真实证据 30 dBµV/m 来源A:2026"
