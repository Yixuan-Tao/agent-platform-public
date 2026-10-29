"""RAG 知识服务客户端：统一认证头、超时与错误映射。

RAG 服务的写操作鉴权有两套语义：
    - 本机模式：``X-Local-Token``（``require_local_token``）
    - LAN 模式：``Authorization: Bearer <admin>``（``require_admin``）

网关持有服务凭据（``RAG_SERVICE_TOKEN``），两个头同时发送即可覆盖两种模式，
不必在网关侧再次判断 RAG 的运行模式。
"""

from __future__ import annotations

from typing import Any

import httpx

from agent_platform.core.config import Settings, get_settings


class RagServiceError(RuntimeError):
    """RAG 服务不可达或返回错误时抛出，携带上游状态码便于网关透传。"""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class RagClient:
    """轻量异步客户端；每次请求新建连接，避免跨事件循环复用问题。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def auth_headers(self) -> dict[str, str]:
        token = self.settings.rag_service_token
        if not token:
            return {}
        return {
            "X-Local-Token": token,
            "Authorization": f"Bearer {token}",
        }

    async def request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> tuple[int, Any]:
        """执行请求并返回 ``(status_code, body)``，上游错误码原样返回。"""
        url = f"{self.settings.rag_service_url.rstrip('/')}{path}"
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.update(self.auth_headers())
        try:
            async with httpx.AsyncClient(
                timeout=timeout or self.settings.rag_request_timeout
            ) as client:
                resp = await client.request(method, url, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise RagServiceError(f"RAG 服务不可达: {exc}", status_code=502) from exc
        content_type = resp.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            try:
                body: Any = resp.json()
            except ValueError:
                body = resp.text
        else:
            body = resp.text
        return resp.status_code, body

    async def request_or_raise(
        self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any
    ) -> Any:
        """同上，但上游非 2xx 时抛出 :class:`RagServiceError`。"""
        status, body = await self.request(method, path, timeout=timeout, **kwargs)
        if status >= 400:
            raise RagServiceError(f"RAG 返回 {status}: {body}", status_code=status)
        return body
