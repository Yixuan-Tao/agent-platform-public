"""把 ToolRegistry 包装为轻量 MCP JSON-RPC 2.0 服务。

支持两种用法：
1. 进程内调用 call(registry, name, arguments)
2. stdio 协议服务器 serve_stdio(registry)：逐行读 JSON-RPC 请求，写响应到 stdout
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from agent_platform.tools.base import ToolRegistry
from agent_platform.tools.gateway import ToolGateway


def _rpc_result(req: dict[str, Any], result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req.get("id"), "result": result}


def _rpc_error(req: dict[str, Any], code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": code, "message": message}}


class McpToolServer:
    """进程内 JSON-RPC 处理器。"""

    def __init__(self, registry: ToolRegistry, gateway: ToolGateway | None = None) -> None:
        self.registry = registry
        self.gateway = gateway or ToolGateway(registry=registry)

    async def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        method = request.get("method", "")
        params = request.get("params", {}) or {}
        if method == "tools/list":
            return _rpc_result(request, {"tools": self.registry.manifests()})
        if method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments", {}) or {}
            result = await self.gateway.execute(name, args, principal_role="admin")
            if result.get("ok"):
                return _rpc_result(
                    request, {"content": [{"type": "text", "text": result["result"]}]}
                )
            return _rpc_error(
                request,
                -32000,
                f"{result.get('error_code', 'UNKNOWN')}: {result.get('message', '')}",
            )
        if method == "tools/call_many":
            calls = params.get("calls", []) or []
            results = await self.gateway.execute_many(calls, principal_role="admin")
            return _rpc_result(
                request, {"content": [{"type": "text", "text": json.dumps(results, ensure_ascii=False)}]}
            )
        return _rpc_error(request, -32601, f"未知方法: {method}")

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        resp = await self.handle({"id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        items = resp.get("result", {}).get("content", [])
        return "".join(i.get("text", "") for i in items if i.get("type") == "text")


async def serve_stdio(registry: ToolRegistry) -> None:
    """stdio 协议：逐行读入 JSON-RPC，写响应到 stdout。"""
    server = McpToolServer(registry)
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = await server.handle(req)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        await loop.run_in_executor(None, sys.stdout.flush)
