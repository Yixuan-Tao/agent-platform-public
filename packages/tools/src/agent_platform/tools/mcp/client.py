"""MCP 客户端：stdio / Streamable HTTP 两种传输，把外部 MCP 服务器工具接入本地注册表。

原生实现（仅依赖 httpx / asyncio），不引入 mcp SDK，便于内网环境离线可用。
远程工具以 `mcp__<server>__<tool>` 命名注册进 ToolRegistry，可像本地工具一样被 Agent 调用。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import httpx
from loguru import logger

from agent_platform.tools.base import BaseTool, ToolRegistry

_DEFAULT_PROTOCOL_VERSION = "2025-03-26"


def _parse_response(text: str) -> dict[str, Any]:
    """解析 MCP 响应：兼容纯 JSON 与 SSE（取最后一个 data: 行）。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("MCP 响应为空")
    data_lines = [ln[5:].strip() for ln in text.splitlines() if ln.startswith("data:")]
    if data_lines:
        return json.loads(data_lines[-1])
    return json.loads(text)


class McpToolClient:
    """单个外部 MCP 服务器的客户端。"""

    def __init__(
        self,
        server_name: str,
        *,
        transport: str = "stdio",
        url: str | None = None,
        command: list[str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.server_name = server_name
        self.transport = transport
        self.url = url
        self.command = command or [sys.executable, "-c", "raise SystemExit('no command')"]
        self.timeout = timeout
        self._process: asyncio.subprocess.Process | None = None
        self._http: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._next_id = 1
        self._ready = False
        # stdio 传输是单管道请求/响应模型：并发写请求会让两个协程同时等待
        # 同一行 stdout，读出彼此的消息（已复现 "readuntil() called while
        # another coroutine is already waiting"）。用锁把每次请求-响应串起来。
        self._request_lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._ready

    async def start(self) -> None:
        """启动传输并完成 initialize 握手。"""
        if self.transport == "stdio":
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        elif self.transport in ("http", "streamable-http"):
            if not self.url:
                raise ValueError("streamable-http 传输需要 url")
            self._http = httpx.AsyncClient(timeout=self.timeout)
        else:
            raise ValueError(f"不支持的 MCP 传输: {self.transport}")
        await self._request(
            "initialize",
            {
                "protocolVersion": _DEFAULT_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agent-platform", "version": "0.1.0"},
            },
        )
        self._ready = True
        await self._notify("notifications/initialized", {})

    async def _write_http(
        self, payload: dict[str, Any], expect_response: bool = True
    ) -> dict[str, Any] | None:
        if self._http is None:
            raise RuntimeError("HTTP 客户端未初始化")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        resp = await self._http.post(
            self.url, content=json.dumps(payload, ensure_ascii=False), headers=headers
        )
        session_id = resp.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id
        if not expect_response:
            return None
        if resp.status_code >= 400:
            raise RuntimeError(f"MCP HTTP {resp.status_code}: {resp.text[:200]}")
        return _parse_response(resp.text)

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.transport == "stdio":
            async with self._request_lock:
                return await self._request_locked(method, params)
        return await self._request_locked(method, params)

    async def _request_locked(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        if self.transport == "stdio":
            if self._process is None or self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("stdio 子进程未启动")
            self._process.stdin.write(
                (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            )
            await self._process.stdin.drain()
            raw = await asyncio.wait_for(self._process.stdout.readline(), timeout=self.timeout)
            resp = json.loads(raw.decode("utf-8"))
        else:
            resp = await self._write_http(payload)
            assert resp is not None
        if "error" in resp:
            raise RuntimeError(f"MCP {method} 错误: {resp['error']}")
        return resp.get("result", {})

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        if self.transport == "stdio":
            if self._process is None or self._process.stdin is None:
                return
            self._process.stdin.write(
                (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            )
            await self._process.stdin.drain()
        else:
            await self._write_http(payload, expect_response=False)

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._request("tools/list", {})
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._request("tools/call", {"name": name, "arguments": arguments or {}})
        content = result.get("content", []) or []
        return "".join(i.get("text", "") for i in content if i.get("type") == "text")

    async def close(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=3)
            except Exception:  # noqa: BLE001
                pass
            self._process = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None


class RemoteTool(BaseTool):
    read_only = False
    idempotent = False
    """把外部 MCP 工具包装成本地 BaseTool。"""

    def __init__(self, client: McpToolClient, server_name: str, tool_spec: dict[str, Any]) -> None:
        remote_name = tool_spec.get("name", "")
        self.name = f"mcp__{server_name}__{remote_name}"
        self.required_scopes = (f"mcp:{server_name}:{remote_name}",)
        self.description = tool_spec.get("description", f"外部 MCP 工具 {remote_name}")
        self.args_schema = (
            tool_spec.get("inputSchema")
            or tool_spec.get("args_schema")
            or {"type": "object", "properties": {}, "required": []}
        )
        self._client = client
        self._remote_name = remote_name

    async def execute(self, **kwargs: Any) -> str:
        return await self._client.call_tool(self._remote_name, kwargs)


async def register_mcp_servers(
    registry: ToolRegistry,
    servers: list[dict[str, Any]],
) -> list[McpToolClient]:
    """按配置连接外部 MCP 服务器并注册远程工具；单个服务器失败不影响其余。"""
    clients: list[McpToolClient] = []
    for cfg in servers or []:
        name = cfg.get("name", "")
        if not name:
            continue
        client = McpToolClient(
            name,
            transport=cfg.get("transport", "stdio"),
            url=cfg.get("url"),
            command=cfg.get("command"),
        )
        try:
            await client.start()
            specs = await client.list_tools()
            for spec in specs:
                if not spec.get("name"):
                    continue
                try:
                    registry.register(RemoteTool(client, name, spec))
                except ValueError as exc:
                    logger.warning("跳过重复 MCP 工具: {}", exc)
            clients.append(client)
            logger.info("MCP 服务器 {} 就绪，注册 {} 个工具", name, len(specs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP 服务器 {} 连接失败，跳过: {}", name, exc)
            await client.close()
    return clients


def parse_mcp_servers(env_value: str) -> list[dict[str, Any]]:
    """解析 MCP_SERVERS 环境变量（JSON 数组）。"""
    if not env_value:
        return []
    try:
        data = json.loads(env_value)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        logger.warning("MCP_SERVERS 不是合法 JSON，忽略")
        return []
