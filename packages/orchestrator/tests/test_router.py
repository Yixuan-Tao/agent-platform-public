# Test values are synthetic fixtures for software behavior.
from __future__ import annotations

from agent_platform.orchestrator.router import SemanticRouter


def test_keyword_hit():
    router = SemanticRouter()
    agent, conf = router.route("VPN 无法连接，报修")
    assert agent == "it_support"
    assert conf > 0


def test_emc_hit():
    router = SemanticRouter()
    agent, _ = router.route("帮我对比两份 EMC 标准的辐射发射限值")
    assert agent == "emc_compare"


def test_no_hit():
    router = SemanticRouter()
    agent, conf = router.route("今天天气怎么样")
    assert agent is None
    assert conf == 0


async def test_llm_fallback():
    class FakeLLM:
        async def chat(self, messages, temperature=0.7, **kw):
            return "hardware_qa"

    router = SemanticRouter()
    result = await router.ensure_llm_route(
        "随便问问", FakeLLM(), ["it_support", "hardware_qa"]
    )
    assert result == "hardware_qa"


async def test_llm_fallback_invalid():
    class FakeLLM:
        async def chat(self, messages, temperature=0.7, **kw):
            return "nonsense"

    router = SemanticRouter()
    result = await router.ensure_llm_route("随便问问", FakeLLM(), ["it_support"])
    assert result is None
