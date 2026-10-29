# Test values are synthetic fixtures for software behavior.
"""工具运行时网关测试。"""

from __future__ import annotations

import asyncio

from agent_platform.tools.base import BaseTool, ToolRegistry
from agent_platform.tools.gateway import ToolError, ToolGateway


class EchoTool(BaseTool):
    name = "echo"
    description = "回显"
    args_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, text: str, **kwargs):
        return f"echo:{text}"


class SlowTool(BaseTool):
    name = "slow"
    description = "慢工具"
    args_schema = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        await asyncio.sleep(5)
        return "done"


class SecretTool(BaseTool):
    name = "secret"
    description = "仅管理员"
    required_role = "admin"
    args_schema = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        return "secret-data"


async def test_gateway_validate_fails():
    gw = ToolGateway(tools=[EchoTool()])
    result = await gw.execute("echo", {"text": 123})
    assert result["ok"] is False
    assert result["error_code"] == "INVALID_ARGS"


async def test_gateway_ok():
    gw = ToolGateway(tools=[EchoTool()])
    result = await gw.execute("echo", {"text": "hi"})
    assert result["ok"] is True
    assert result["result"] == "echo:hi"


async def test_gateway_acl():
    gw = ToolGateway(tools=[SecretTool()])
    denied = await gw.execute("secret", {}, principal_role="user")
    assert denied["ok"] is False
    assert denied["error_code"] == "FORBIDDEN"
    allowed = await gw.execute("secret", {}, principal_role="admin")
    assert allowed["ok"] is True


async def test_gateway_timeout_retryable():
    gw = ToolGateway(tools=[SlowTool()], timeout=0.2, max_retries=1)
    result = await gw.execute("slow", {})
    assert result["ok"] is False
    assert result["error_code"] == "TIMEOUT"
    assert result["retryable"] is True


async def test_gateway_missing_tool():
    gw = ToolGateway(tools=[EchoTool()])
    result = await gw.execute("nope", {})
    assert result["ok"] is False
    assert result["error_code"] == "TOOL_NOT_FOUND"


async def test_gateway_retry_success():
    """可重试错误第一次失败、第二次成功。"""
    calls = {"n": 0}

    class FlakyTool(BaseTool):
        name = "flaky"
        description = "抖动工具"
        args_schema = {"type": "object", "properties": {}, "required": []}

        async def execute(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ToolError("TEMP_FAIL", "暂时失败", retryable=True)
            return "ok-after-retry"

    gw = ToolGateway(tools=[FlakyTool()], max_retries=1)
    result = await gw.execute("flaky", {})
    assert result["ok"] is True
    assert result["result"] == "ok-after-retry"


async def test_gateway_execute_many_isolated():
    gw = ToolGateway(tools=[EchoTool(), SecretTool()])
    results = await gw.execute_many(
        [
            {"name": "echo", "arguments": {"text": "a"}},
            {"name": "secret", "arguments": {}},
        ]
    )
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False
    assert results[1]["error_code"] == "FORBIDDEN"


async def test_register_from_config():
    import sys
    import types

    mod = types.ModuleType("_fake_tool_mod")

    async def _hello(name: str):
        return f"hello:{name}"

    mod.hello = _hello
    sys.modules["_fake_tool_mod"] = mod
    reg = ToolRegistry()
    n = reg.register_from_config(
        [
            {
                "name": "hello",
                "description": "问候",
                "module": "_fake_tool_mod",
                "function": "hello",
            }
        ]
    )
    assert n == 1
    assert await reg.get("hello").execute(name="x") == "hello:x"
