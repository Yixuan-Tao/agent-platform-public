"""健康检查。"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Request

from agent_platform.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    settings = get_settings()
    rag_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.rag_service_url}/health")
            rag_status = "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
    except Exception:
        rag_status = "unreachable"
    return {
        "status": "ok",
        "app": settings.app_name,
        "rag_service": rag_status,
        "model_gateway": getattr(
            request.app.state, "gateway_status", {"ok": None, "reason": "unknown"}
        ),
    }
