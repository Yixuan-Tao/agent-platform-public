"""FastAPI 依赖：从 app.state 获取共享服务。"""

from __future__ import annotations

from fastapi import Request

from agent_platform.memory.service import MemoryService
from agent_platform.orchestrator.service import OrchestratorService


def get_orchestrator(request: Request) -> OrchestratorService:
    return request.app.state.orchestrator


def get_memory(request: Request) -> MemoryService:
    return request.app.state.memory
