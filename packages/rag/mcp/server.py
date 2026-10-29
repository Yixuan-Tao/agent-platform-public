"""
MCP Server — 基于 FastAPI 的标准 MCP HTTP+SSE 服务

实现 MCP 规范的核心协议：
- JSON-RPC 2.0 消息格式
- tools/list：列出可用工具
- tools/call：调用指定工具
- 通过 HTTP POST + SSE 进行通信
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mcp.handlers import handle_compare, handle_plan, handle_search
from mcp.tools import MCP_TOOLS, TOOL_COMPARE, TOOL_PLAN, TOOL_SEARCH
from pipeline.query import QueryPipeline
from utils.logger import log

# MCP 协议常量
JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2024-11-05"

# 工具名 → 处理函数映射
_HANDLER_MAP = {
    TOOL_PLAN: handle_plan,
    TOOL_SEARCH: handle_search,
    TOOL_COMPARE: handle_compare,
}

# SSE 会话管理（每个 client 一个 session）
_sessions: dict[str, asyncio.Queue] = {}

router = APIRouter(prefix="/mcp", tags=["MCP"])


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 请求体。"""
    jsonrpc: str = JSONRPC_VERSION
    id: Optional[str] = None
    method: str
    params: Optional[dict[str, Any]] = None


def _make_response(req_id: Optional[str], result: Any) -> dict:
    """构建 JSON-RPC 成功响应。"""
    return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": result}


def _make_error(req_id: Optional[str], code: int, message: str) -> dict:
    """构建 JSON-RPC 错误响应。"""
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _get_pipeline(request: Request) -> QueryPipeline:
    """从 app.state 获取 QueryPipeline，未注入时兜底构造。"""
    pipeline = getattr(request.app.state, "query_pipeline", None)
    if pipeline is None:
        pipeline = QueryPipeline()
        request.app.state.query_pipeline = pipeline
    return pipeline


@router.get("/health")
async def mcp_health():
    """MCP 服务健康检查。"""
    return {
        "status": "ok",
        "protocol": "mcp",
        "version": MCP_PROTOCOL_VERSION,
        "tools": len(MCP_TOOLS),
    }


@router.get("/sse")
async def mcp_sse():
    """SSE 事件流 — 客户端通过此端点建立长连接接收服务端推送。"""
    session_id = uuid4().hex[:12]
    queue: asyncio.Queue = asyncio.Queue()
    _sessions[session_id] = queue

    async def event_stream():
        """SSE 事件生成器，包含心跳保活机制。"""
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    # 30s 无消息时发送心跳，防止连接被代理断开
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _sessions.pop(session_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/messages")
async def mcp_messages(req: JsonRpcRequest, request: Request):
    """接收 JSON-RPC 请求，处理 MCP 协议方法。"""
    method = req.method
    params = req.params or {}
    req_id = req.id

    try:
        if method == "initialize":
            return _make_response(req_id, {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "rag-mcp-server",
                    "version": "0.1.0",
                },
            })

        elif method == "tools/list":
            return _make_response(req_id, {"tools": MCP_TOOLS})

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            pipeline = _get_pipeline(request)
            handler = _HANDLER_MAP.get(tool_name)

            if handler is None:
                return _make_error(req_id, -32601, f"未知工具: {tool_name}")

            result = handler(arguments, pipeline)

            return _make_response(req_id, {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, ensure_ascii=False, indent=2),
                    }
                ],
                "isError": "error" in result,
            })

        elif method == "notifications/initialized":
            return _make_response(req_id, {})

        else:
            return _make_error(req_id, -32601, f"未知方法: {method}")

    except Exception as e:
        log.exception("MCP 消息处理失败: %s", e)
        return _make_error(req_id, -32603, str(e))


__all__ = ["router"]
