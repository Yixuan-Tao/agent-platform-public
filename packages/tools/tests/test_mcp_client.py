# Test values are synthetic fixtures for software behavior.
"""MCP 客户端测试（stdio 假服务器）。"""

from __future__ import annotations

import asyncio
import json
import sys

from agent_platform.tools.base import ToolRegistry
from agent_platform.tools.mcp.client import McpToolClient, register_mcp_servers

_FAKE_SERVER = r"""
import json, sys
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    if method == "initialize":
        out = {"jsonrpc": "2.0", "id": req.get("id"), "result": {"protocolVersion": "2025-03-26", "capabilities": {}, "serverInfo": {"name": "fake", "version": "1"}}}
    elif method == "tools/list":
        out = {"jsonrpc": "2.0", "id": req.get("id"), "result": {"tools": [{"name": "upper", "description": "转大写", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}}
    elif method == "tools/call":
        args = req.get("params", {}).get("arguments", {})
        out = {"jsonrpc": "2.0", "id": req.get("id"), "result": {"content": [{"type": "text", "text": args.get("text", "").upper()}]}}
    else:
        continue
    sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    sys.stdout.flush()
"""


async def test_mcp_client_stdio():
    client = McpToolClient(
        "fake", transport="stdio", command=[sys.executable, "-c", _FAKE_SERVER]
    )
    await client.start()
    assert client.ready
    tools = await client.list_tools()
    assert tools and tools[0]["name"] == "upper"
    result = await client.call_tool("upper", {"text": "hello"})
    assert result == "HELLO"
    await client.close()


async def test_register_mcp_servers():
    registry = ToolRegistry()
    clients = await register_mcp_servers(
        registry,
        [
            {
                "name": "fake",
                "transport": "stdio",
                "command": [sys.executable, "-c", _FAKE_SERVER],
            }
        ],
    )
    assert "mcp__fake__upper" in registry.names()
    tool = registry.get("mcp__fake__upper")
    assert await tool.execute(text="abc") == "ABC"
    for client in clients:
        await client.close()


_SLOW_SERVER = r"""
import json, sys, time
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
for line in sys.stdin:
    req = json.loads(line)
    if "id" not in req:
        continue
    method = req.get("method")
    if method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "回显",
                             "inputSchema": {"type": "object",
                                             "properties": {"text": {"type": "string"}},
                                             "required": ["text"]}}]}
    elif method == "tools/call":
        # 制造交错窗口：若客户端不串行化请求/响应，第二个协程会读走第一个响应
        time.sleep(0.05)
        text = req.get("params", {}).get("arguments", {}).get("text", "")
        result = {"content": [{"type": "text", "text": text}]}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}) + "\n")
    sys.stdout.flush()
"""


async def test_mcp_stdio_serializes_concurrent_calls():
    """stdio 是单管道请求/响应，并发调用必须串行化（历史 bug：读取互相抢行）。"""
    client = McpToolClient(
        "slow", transport="stdio", command=[sys.executable, "-c", _SLOW_SERVER]
    )
    await client.start()
    try:
        outcomes = await asyncio.gather(
            *(client.call_tool("echo", {"text": f"t{i}"}) for i in range(5))
        )
    finally:
        await client.close()
    assert outcomes == [f"t{i}" for i in range(5)]
