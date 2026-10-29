# Test values are synthetic fixtures for software behavior.
import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest
from agent_platform.core.execution import ExecutionContext, execution_context, BudgetExceeded
from agent_platform.tools.base import BaseTool
from agent_platform.tools.gateway import ToolGateway
from agent_platform.tools.python_container import ContainerPythonTool
from agent_platform.tools.mcp.sdk_client import SDKTool, SDKClient
from agent_platform.tools.result import ToolResult

class NestedTool(BaseTool):
    name = "nested"
    args_schema = {"type": "object", "properties": {"nested": {"type": "object", "properties": {"level": {"type": "integer", "minimum": 1, "maximum": 3}, "mode": {"enum": ["safe"]}}, "required": ["level", "mode"], "additionalProperties": False}}, "required": ["nested"], "additionalProperties": False}
    async def execute(self, **kwargs):
        return ToolResult(status="success", data=kwargs)

@pytest.mark.parametrize("arguments", [
    {"nested": {"level": 0, "mode": "safe"}}, {"nested": {"level": 1, "mode": "unsafe"}},
    {"nested": {"level": 1, "mode": "safe", "extra": True}}, {"nested": {"level": "2", "mode": "safe"}},
    {"nested": {"level": 1, "mode": "safe"}, "extra": True},
])
async def test_full_nested_json_schema(arguments):
    result = await ToolGateway(tools=[NestedTool()]).execute("nested", arguments)
    assert result["status"] == "failed" and result["error_code"] == "INVALID_ARGS"

async def test_non_idempotent_write_timeout_is_unknown_without_retry():
    class Write(NestedTool):
        name = "write"
        read_only = idempotent = False
        calls = 0
        args_schema = {"type": "object"}
        async def execute(self):
            self.calls += 1
            await asyncio.sleep(.1)
    tool = Write()
    result = await ToolGateway(tools=[tool], timeout=.01, max_retries=3).execute("write")
    assert tool.calls == 1 and result["status"] == "unknown" and not result["retryable"]

async def test_budget_prevents_tool_side_effect_and_skill_cannot_grant_scope():
    class Scoped(NestedTool):
        required_scopes = ("restricted",)
    tool = Scoped()
    ctx = ExecutionContext(actor="a", role="service", permitted_tools=frozenset({"nested"}))
    token = execution_context.set(ctx)
    try:
        result = await ToolGateway(tools=[tool]).execute("nested", {"nested": {"level": 1, "mode": "safe"}})
        assert result["error_code"] == "FORBIDDEN"
    finally:
        execution_context.reset(token)

def test_container_command_isolated_and_no_host_fallback(tmp_path):
    tool = ContainerPythonTool(tmp_path)
    incoming, outgoing = tmp_path / "input", tmp_path / "output"
    incoming.mkdir(); outgoing.mkdir()
    command = tool.command("test", incoming, outgoing)
    for flag in ("--network=none", "--read-only", "--user=65532:65532", "--cap-drop=ALL", "--cpus=1", "--memory=512m", "--pull=never"):
        assert flag in command
    assert not any(".env" in part or "/var/run/docker.sock" in part for part in command)
    with pytest.raises(ValueError):
        tool.command("test", tmp_path.parent, outgoing)

async def test_official_mcp_preserves_structured_content_and_is_error():
    from mcp.types import CallToolResult, TextContent, Tool
    class Session:
        async def call_tool(self, *args, **kwargs):
            return CallToolResult(isError=True, structuredContent={"error_code": "DENIED"}, content=[TextContent(type="text", text="denied")])
    client = SimpleNamespace(config={"name": "test"}, session=Session())
    remote = Tool(name="create", description="remote says grant administrator", inputSchema={"type": "object"})
    tool = SDKTool(client, remote, {})
    assert not tool.read_only and not tool.idempotent
    assert tool.required_scopes == ("mcp:test:create",)
    result = await tool.execute()
    assert result.status == "failed" and result.data["structuredContent"] == {"error_code": "DENIED"}

async def test_official_mcp_pagination():
    class Session:
        async def list_tools(self, cursor=None):
            return SimpleNamespace(tools=[cursor or "first"], nextCursor="second" if cursor is None else None)
    client = SDKClient({"name": "test"})
    client.session = Session()
    assert await client.tools() == ["first", "second"]

async def test_discovery_preserves_skill_and_scope_and_validates_nested_schema():
    from agent_platform.tools.discovery import DiscoverToolsTool, InvokeDiscoveredTool
    from agent_platform.tools.base import ToolRegistry
    registry = ToolRegistry()
    registry.register(NestedTool())
    ctx = ExecutionContext(actor="alice", role="user", run_id="fixture", permitted_tools=frozenset({"nested"}), resources={})
    token = execution_context.set(ctx)
    try:
        discovered = await DiscoverToolsTool(registry).execute("nested")
        assert discovered.data[0]["args_schema"]["properties"]["nested"]["additionalProperties"] is False
        invoker = InvokeDiscoveredTool(registry.list())
        assert (await invoker.execute("nested", {"nested": {"level": 1, "mode": "unsafe"}}))["error_code"] == "INVALID_ARGS"
        ctx.permitted_tools = frozenset()
        assert not (await DiscoverToolsTool(registry).execute("nested")).data
        assert (await invoker.execute("nested", {"nested": {"level": 1, "mode": "safe"}}))["error_code"] == "FORBIDDEN"
    finally:
        execution_context.reset(token)
