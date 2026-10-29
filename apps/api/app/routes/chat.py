"""SSE 流式对话。"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from agent_platform.core.db import get_session, session_scope
from agent_platform.core.models import ChatMessage, ChatRequest
from agent_platform.core.security import require_login
from agent_platform.memory.service import MemoryService
from agent_platform.orchestrator.service import OrchestratorService
from agent_platform.tools.identity import reset_principal_role, set_principal_role
from app.dependencies import get_memory, get_orchestrator
from app.routes.sessions import append_message, ensure_session, new_session_id
from agent_platform.core.execution import ExecutionContext, execution_context

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _sse(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False)


@router.post("")
async def chat(
    req: ChatRequest,
    identity: dict = Depends(require_login),
    orchestrator: OrchestratorService = Depends(get_orchestrator),
    memory: MemoryService = Depends(get_memory),
    db: AsyncSession = Depends(get_session),
    request: Request = None,
) -> EventSourceResponse:
    if request and request.app.state.settings.chat_via_runs:
        from app.routes.runs import runtime, scoped, checked
        import asyncio
        from agent_platform.orchestrator.runtime_store import TERMINAL
        service = runtime(request)
        run = await checked(service.create(scoped(identity, request), req.message, req.session_id,
                    request.app.state.settings.default_run_profile, completion={"agent_hint": req.agent_hint}))
        async def durable_stream():
            cursor = 0
            yield {"data": _sse({"type": "session", "session_id": run["session_id"], "run_id": run["id"]})}
            while not await request.is_disconnected():
                for event in await service.store.events(run["id"], identity["username"], cursor):
                    cursor = event["seq"]
                    if event["type"] == "answer_delta":
                        yield {"data": _sse({"type": "token", "content": event["content"]})}
                    elif event["type"] == "answer":
                        yield {"data": _sse({"type": "done", "answer": event["answer"], "agent": event["agent"], "run_id": run["id"]})}
                    elif event["type"] in {"approval", "input", "status"}:
                        yield {"data": _sse({**event, "run_id": run["id"]})}
                current = await service.store.get(run["id"], identity["username"])
                if current["status"] in TERMINAL and cursor >= current["seq"]:
                    if current["result"].get("error"):
                        yield {"data": _sse({"type": "error", "message": current["result"]["error"]})}
                    yield {"data": _sse({"type": "session", "session_id": run["session_id"], "run_id": run["id"], "message_id": run["id"] + "-answer" if current["result"].get("answer") else ""})}
                    return
                await asyncio.sleep(.3)
        return EventSourceResponse(durable_stream())
    user_id = identity["username"]
    session_id = req.session_id or new_session_id()
    await ensure_session(session_id, user_id, db)
    await append_message(
        db, session_id=session_id, user_id=user_id, role="user", content=req.message
    )

    # 写入短期记忆
    await memory.short_term.append_message(
        session_id, ChatMessage(role="user", content=req.message)
    )

    async def event_gen():
        answer_parts: list[str] = []
        agent_name = ""
        # Service keys retain explicit tool scopes and never inherit administrator role.
        role_token = set_principal_role(_tool_role(identity))
        from agent_platform.core.config import get_settings
        settings = request.app.state.settings if request else get_settings()
        scopes = settings.service_tool_scopes if identity.get("role") == "service" else settings.user_tool_scopes
        identity_token = execution_context.set(ExecutionContext(actor=identity["username"], role=_tool_role(identity), tool_scopes=frozenset(scopes)))
        try:
            runner = (
                orchestrator.route_and_run_react
                if getattr(orchestrator, "use_router_react", False)
                else orchestrator.route_and_run
            )
            async for event in runner(
                session_id=session_id,
                user_message=req.message,
                agent_hint=req.agent_hint,
            ):
                if event["type"] == "token":
                    answer_parts.append(event["content"])
                elif event["type"] == "agent_switch":
                    agent_name = event.get("agent", "") or agent_name
                elif event["type"] == "done":
                    agent_name = event.get("agent", "") or agent_name
                yield {"data": _sse(event)}
        except Exception as exc:  # noqa: BLE001
            yield {"data": _sse({"type": "error", "message": str(exc)})}
            return
        finally:
            reset_principal_role(role_token)
            execution_context.reset(identity_token)

        answer = "".join(answer_parts)
        message_id = ""
        if answer:
            await memory.short_term.append_message(
                session_id, ChatMessage(role="assistant", content=answer)
            )
            # FastAPI 已关闭请求级会话，这里另开一个写入助手消息。
            async with session_scope() as scoped:
                message_id = await append_message(
                    scoped,
                    session_id=session_id,
                    user_id=user_id,
                    role="assistant",
                    content=answer,
                    agent=agent_name,
                )
        yield {
            "data": _sse(
                {
                    "type": "session",
                    "session_id": session_id,
                    "message_id": message_id,
                }
            )
        }

    return EventSourceResponse(event_gen())


def _tool_role(identity: dict) -> str:
    """把接入层身份映射为工具 ACL 角色。

    仅 ``admin`` 映射为管理员，服务身份仍需具体工具权限；
    普通登录用户映射为 user（可执行常规工具，但不能调用管理级工具）。
    """
    role = (identity or {}).get("role", "")
    return "admin" if role == "admin" else "user"
