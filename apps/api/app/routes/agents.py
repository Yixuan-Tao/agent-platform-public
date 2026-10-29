"""Agent 注册表查看。"""

from __future__ import annotations

from fastapi import APIRouter

from agent_platform.orchestrator.registry import get_agent_registry

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


@router.get("")
async def list_agents() -> dict:
    registry = get_agent_registry()
    manifests = [m.model_dump() for m in registry.list_manifests()]
    return {"agents": manifests, "count": len(manifests)}
