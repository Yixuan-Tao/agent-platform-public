# Test values are synthetic fixtures for software behavior.
"""回答流式：增量下发、工具消息过滤、非流式图自动回退。"""

from __future__ import annotations

from agent_platform.core.models import AgentManifest
from agent_platform.orchestrator.registry import AgentRegistry
from agent_platform.orchestrator.service import OrchestratorService


class AIMessageChunk:
    def __init__(self, content: str) -> None:
        self.content = content


class ToolMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class StreamingGraph:
    """最小可用的 messages 流：含工具消息与两段回答增量。"""

    def __init__(self, chunks=None) -> None:
        self.chunks = chunks or [
            ToolMessage("检索片段（不应出现在答案里）"),
            AIMessageChunk("你好，"),
            AIMessageChunk("这是流式回答。"),
        ]

    async def astream(self, state, stream_mode="messages"):
        for chunk in self.chunks:
            yield chunk, {"langgraph_node": "agent"}

    async def ainvoke(self, state):
        return {"messages": []}


class BufferedGraph:
    """没有 astream 的图（如 SimpleAgent / 测试替身）。"""

    async def ainvoke(self, state):
        return {"final_answer": "非流式回答", "messages": []}


class FakeLLM:
    def __init__(self, verdict: str = "PASS") -> None:
        self.verdict = verdict
        self.settings = None

    async def chat(self, messages, temperature=0.7, **kw):
        return self.verdict

    def resolve_model(self, agent_name=None):
        return "fake-model"


def _service(answer_streaming: bool) -> OrchestratorService:
    return OrchestratorService(
        registry=AgentRegistry(),
        llm_router=FakeLLM(),
        tool_registry=None,
        answer_streaming=answer_streaming,
    )


async def _collect(gen):
    return [event async for event in gen]


async def test_stream_answer_emits_deltas_and_filters_tool_messages():
    svc = _service(True)
    events = [item async for item in svc._stream_answer(StreamingGraph(), [])]
    kinds = [kind for kind, _ in events]
    assert kinds == ["delta", "delta", "final"]
    assert events[0][1] == "你好，"
    assert events[1][1] == "这是流式回答。"
    assert events[2][1] == "你好，这是流式回答。"
    assert "检索片段" not in events[2][1]


async def test_stream_answer_falls_back_when_graph_cannot_stream():
    svc = _service(True)
    events = [item async for item in svc._stream_answer(BufferedGraph(), [])]
    assert events == [("final", "非流式回答")]


async def test_stream_answer_reports_error_after_partial_output():
    """已经下发过增量后失败，必须报错而不是重发整段答案。"""

    class BrokenGraph(StreamingGraph):
        async def astream(self, state, stream_mode="messages"):
            yield AIMessageChunk("前半段"), {}
            raise RuntimeError("连接中断")

        async def ainvoke(self, state):
            raise AssertionError("不应回退为非流式，否则会重复输出")

    svc = _service(True)
    events = [item async for item in svc._stream_answer(BrokenGraph(), [])]
    assert events[0] == ("delta", "前半段")
    assert events[-1][0] == "error"
    assert "连接中断" in events[-1][1]


async def test_route_and_run_streams_tokens_when_enabled():
    reg = AgentRegistry()
    reg.register(
        AgentManifest(name="it_support", description="IT 支持"),
        lambda llm_router, tool_registry: StreamingGraph(),
    )
    svc = OrchestratorService(
        registry=reg,
        llm_router=FakeLLM(),
        tool_registry=None,
        answer_streaming=True,
    )
    events = await _collect(svc.route_and_run("s1", "VPN 故障", agent_hint="it_support"))
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == ["你好，", "这是流式回答。"]
    done = events[-1]
    assert done["type"] == "done"
    assert done["answer"] == "你好，这是流式回答。"


async def test_route_and_run_buffers_tokens_when_disabled():
    """默认关闭流式时保持既有行为：一次性生成后按块下发。"""
    reg = AgentRegistry()
    reg.register(
        AgentManifest(name="it_support", description="IT 支持"),
        lambda llm_router, tool_registry: BufferedGraph(),
    )
    svc = OrchestratorService(
        registry=reg,
        llm_router=FakeLLM(),
        tool_registry=None,
        answer_streaming=False,
    )
    events = await _collect(svc.route_and_run("s1", "VPN 故障", agent_hint="it_support"))
    done = events[-1]
    assert done["type"] == "done"
    assert done["answer"] == "非流式回答"

async def test_stream_fails_before_tokens_without_replaying_graph():
    class Broken(StreamingGraph):
        async def astream(self, *args, **kwargs):
            if False:
                yield None
            raise RuntimeError("socket closed before first token")
        async def ainvoke(self, state):
            raise AssertionError("Graph replay would repeat an external write")
    events = [event async for event in _service(True)._stream_answer(Broken(), [])]
    assert len(events) == 1 and events[0][0] == "error"

async def test_empty_stream_does_not_replay_graph():
    class Empty(StreamingGraph):
        async def astream(self, *args, **kwargs):
            if False:
                yield None
        async def ainvoke(self, state):
            raise AssertionError("Do not execute a second time")
    assert [event async for event in _service(True)._stream_answer(Empty(), [])] == [("error", "执行结束但没有最终回答")]
