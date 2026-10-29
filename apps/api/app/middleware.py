"""网关鉴权中间件：统一识别服务密钥与用户会话，并记录访问/审计。"""

from __future__ import annotations

import time

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from agent_platform.core.security import resolve_identity

# 免鉴权路径：健康检查、交互式文档、登录接口，以及浏览器自动请求的 favicon。
_PUBLIC_PREFIXES = ("/health", "/docs", "/openapi.json", "/redoc", "/api/v1/auth", "/favicon.ico")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """请求鉴权：需要 ``X-API-Key``、``ADMIN_API_KEY`` 或有效 Bearer 会话之一。

    解析成功的身份写入 ``request.state.identity``，供路由依赖与审计复用，
    避免各处重复解析导致语义不一致。``/health``、文档与登录接口放行。
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # CORS 预检（OPTIONS）放行：预检不带凭据，由 CORSMiddleware 处理
        if request.method == "OPTIONS":
            return await call_next(request)
        # 根路径单独判断：不能把 "/" 放进前缀列表（那会放行所有请求）
        if path == "/" or any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)

        identity = resolve_identity(request)
        if identity is None:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "缺少或无效的身份凭据",
                    "hint": "登录获取 Bearer 令牌：POST /api/v1/auth/login；"
                    "自动化可用 X-API-Key。前端页面见 / 根路径索引。",
                },
                # 标准做法：告知客户端需要 Bearer 凭据
                headers={"WWW-Authenticate": 'Bearer realm="agent-platform"'},
            )
        request.state.identity = identity
        return await call_next(request)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """访问日志与耗时统计；对写操作追加用户归属审计记录。"""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        cost_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "{} {} -> {} ({:.0f}ms)",
            request.method,
            request.url.path,
            response.status_code,
            cost_ms,
        )
        # 审计写操作：归属到 Bearer 会话用户 / ADMIN_API_KEY / X-API-Key 自动化
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            try:
                audit_store = getattr(request.app.state, "audit_store", None)
                if audit_store is not None:
                    # 中间件顺序可能使本中间件先于鉴权中间件执行，这里兜底再解析一次。
                    identity = getattr(request.state, "identity", None) or resolve_identity(
                        request
                    )
                    username = identity["username"] if identity else "anonymous"
                    audit_store.record(
                        username=username,
                        source_ip=request.client.host if request.client else "",
                        action=f"{request.method} {request.url.path}",
                        status=str(response.status_code),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("审计写入失败: {}", exc)
        return response
