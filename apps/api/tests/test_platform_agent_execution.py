# Test values are synthetic fixtures for software behavior.
"""平台 Agent 执行评估：走真实注册表 + 真实语义路由 + 真实工具网关。

与 evaluation/testset_v6.json 中"用角色提示词直接生成"的题目不同，这里的
断言覆盖的是平台自身的路由与工具执行代码路径，能发现"提示词答对但平台没接通"
的情况。
"""

from __future__ import annotations

import pytest

from agent_platform.agents import register_all
from agent_platform.core.models import AgentManifest
from agent_platform.orchestrator.registry import AgentRegistry
from agent_platform.orchestrator.router import SemanticRouter
from agent_platform.tools.base import BaseTool, ToolRegistry
from agent_platform.tools.gateway import ToolGateway
from agent_platform.tools.identity import (
    current_principal_role,
    reset_principal_role,
    set_principal_role,
)


@pytest.fixture
def registry() -> AgentRegistry:
    reg = AgentRegistry()
    register_all(reg)
    return reg


def test_registry_exposes_the_four_expert_agents(registry):
    assert set(registry.names()) == {
        "it_support",
        "hardware_qa",
        "emc_compare",
        "info_collect",
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("VPN 无法连接，帮我报修", "it_support"),
        ("这个元器件的击穿电压参数是多少", "hardware_qa"),
        ("对比两份 EMC 标准的辐射限值", "emc_compare"),
        ("帮我做一下竞品市场动态调研", "info_collect"),
    ],
)
def test_real_router_dispatches_to_registered_agent(registry, text, expected):
    agent, confidence = SemanticRouter().route(text)
    assert agent == expected
    assert confidence > 0
    assert registry.has(agent), "路由结果必须存在于注册表"


def test_router_returns_no_agent_for_out_of_scope_text():
    agent, confidence = SemanticRouter().route("今天天气怎么样")
    assert agent is None
    assert confidence == 0.0


async def test_router_llm_fallback_rejects_unsupported():
    class Llm:
        async def chat(self, messages, temperature=0.0, **kw):
            return "unsupported"

    reg = AgentRegistry()
    register_all(reg)
    chosen = await SemanticRouter().ensure_llm_route(
        "今天天气怎么样",
        Llm(),
        reg.names(),
        {m.name: m.description for m in reg.list_manifests()},
    )
    assert chosen is None


async def test_router_llm_fallback_picks_agent_by_name():
    class Llm:
        async def chat(self, messages, temperature=0.0, **kw):
            return "emc_compare"

    chosen = await SemanticRouter().ensure_llm_route(
        "帮我看看这个", Llm(), ["it_support", "emc_compare"], {}
    )
    assert chosen == "emc_compare"


# ------------------------------------------------------------
# 工具 ACL 与角色透传
# ------------------------------------------------------------


class _AdminTool(BaseTool):
    name = "admin_only"
    description = "需要管理员角色"
    required_role = "admin"
    args_schema = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        return "done"


class _UserTool(BaseTool):
    name = "user_level"
    description = "登录用户可用"
    required_role = "user"
    args_schema = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        return "done"


def _gateway() -> ToolGateway:
    reg = ToolRegistry()
    reg.register(_AdminTool())
    reg.register(_UserTool())
    return ToolGateway(registry=reg)


async def test_gateway_enforces_declared_role():
    gateway = _gateway()
    ok = await gateway.execute("admin_only", {}, principal_role="admin")
    denied = await gateway.execute("admin_only", {}, principal_role="user")
    assert ok["ok"] is True
    assert denied["ok"] is False
    assert denied["error_code"] == "FORBIDDEN"

    assert (await gateway.execute("user_level", {}, principal_role="user"))["ok"] is True
    assert (await gateway.execute("user_level", {}, principal_role="viewer"))["ok"] is False


def test_principal_role_context_is_request_scoped_and_restorable():
    assert current_principal_role() == "user"
    token = set_principal_role("admin")
    try:
        assert current_principal_role() == "admin"
    finally:
        reset_principal_role(token)
    assert current_principal_role() == "user"


def test_manifests_declare_capabilities_for_routing():
    reg = AgentRegistry()
    register_all(reg)
    manifests = {m.name: m for m in reg.list_manifests()}
    assert all(isinstance(m, AgentManifest) for m in manifests.values())
    assert manifests["it_support"].description
    assert manifests["emc_compare"].tools, "EMC Agent 应声明可用工具"
