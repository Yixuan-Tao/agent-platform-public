# Test values are synthetic fixtures for software behavior.
from __future__ import annotations

import pytest

from agent_platform.core.models import AgentManifest
from agent_platform.orchestrator.registry import AgentRegistry


def _builder(llm_router, tool_registry):
    return "fake-graph"


def test_register_and_list():
    reg = AgentRegistry()
    reg.register(
        AgentManifest(name="alpha", description="测试 Agent", tools=["rag_search"]),
        _builder,
    )
    assert reg.names() == ["alpha"]
    assert reg.get_manifest("alpha").tools == ["rag_search"]
    assert reg.build("alpha", None, None) == "fake-graph"


def test_duplicate_raises():
    reg = AgentRegistry()
    reg.register(AgentManifest(name="alpha", description="a"), _builder)
    with pytest.raises(ValueError):
        reg.register(AgentManifest(name="alpha", description="b"), _builder)


def test_disabled_filtered():
    reg = AgentRegistry()
    reg.register(AgentManifest(name="a", description="a", enabled=True), _builder)
    reg.register(AgentManifest(name="b", description="b", enabled=False), _builder)
    assert reg.names() == ["a"]

def test_cache_fingerprint_changes_with_model_and_skill_version():
    from agent_platform.core.execution import ExecutionContext, execution_context
    from agent_platform.core.config import Settings
    from agent_platform.core.llm.router import LLMRouter
    reg = AgentRegistry()
    reg.register(AgentManifest(name="a", description="a"), lambda *_: object())
    router = LLMRouter(Settings(_env_file=None, CHAT_MODEL="model-a"))
    ctx = ExecutionContext(actor="alice", profile={"skills": [{"version": "1"}]})
    token = execution_context.set(ctx)
    try:
        first = reg.build("a", router, None)
        assert reg.build("a", router, None) is first
        router.settings.chat_model = "model-b"
        second = reg.build("a", router, None)
        assert second is not first
        ctx.profile = {"skills": [{"version": "2"}]}
        assert reg.build("a", router, None) is not second
    finally:
        execution_context.reset(token)
