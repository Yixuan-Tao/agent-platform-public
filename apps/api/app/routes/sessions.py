"""会话管理。"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agent_platform.core.db import ChatMessageRecord, SessionRecord, get_session
from agent_platform.core.models import ChatMessageInfo, SessionInfo
from agent_platform.core.security import require_login

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionInfo])
async def list_sessions(
    identity: dict = Depends(require_login),
    db: AsyncSession = Depends(get_session),
) -> list[SessionInfo]:
    """列出当前身份名下的会话；不接受客户端传入的 user_id。"""
    result = await db.execute(
        select(SessionRecord)
        .where(SessionRecord.user_id == identity["username"])
        .order_by(SessionRecord.updated_at.desc())
        .limit(50)
    )
    return [
        SessionInfo(
            session_id=r.id,
            user_id=r.user_id,
            title=r.title,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in result.scalars()
    ]


@router.get("/{session_id}/messages", response_model=list[ChatMessageInfo])
async def list_messages(
    session_id: str,
    identity: dict = Depends(require_login),
    db: AsyncSession = Depends(get_session),
) -> list[ChatMessageInfo]:
    """回放会话历史；只有会话归属者可以读取。"""
    await require_session_owner(session_id, identity["username"], db)
    result = await db.execute(
        select(ChatMessageRecord)
        .where(ChatMessageRecord.session_id == session_id)
        .order_by(ChatMessageRecord.id.asc())
        .limit(500)
    )
    return [
        ChatMessageInfo(
            message_id=r.message_id,
            session_id=r.session_id,
            role=r.role,  # type: ignore[arg-type]
            content=r.content,
            agent=r.agent or "",
            created_at=r.created_at,
        )
        for r in result.scalars()
    ]


async def require_session_owner(
    session_id: str, user_id: str, db: AsyncSession
) -> SessionRecord:
    """校验会话存在且属于当前身份；不存在 404，越权 403。"""
    result = await db.execute(select(SessionRecord).where(SessionRecord.id == session_id))
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if record.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")
    return record


async def ensure_session(session_id: str, user_id: str, db: AsyncSession) -> None:
    """不存在则创建会话记录（幂等）；已存在时校验归属，防止会话劫持。"""
    result = await db.execute(
        select(SessionRecord).where(SessionRecord.id == session_id)
    )
    existing = result.scalar_one_or_none()
    if existing is None:
        db.add(SessionRecord(id=session_id, user_id=user_id))
        await db.commit()
        return
    if existing.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")


async def append_message(
    db: AsyncSession,
    *,
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    agent: str = "",
    message_id: str | None = None,
) -> str:
    """持久化一条消息并返回 message_id。"""
    mid = message_id or uuid.uuid4().hex
    db.add(
        ChatMessageRecord(
            message_id=mid,
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=content,
            agent=agent,
        )
    )
    # 同步会话活跃时间；首条用户消息顺带生成标题，便于侧边栏识别。
    values: dict = {"updated_at": func.now()}
    if role == "user":
        values["title"] = func.coalesce(
            func.nullif(SessionRecord.title, "新会话"),
            content[:30],
        )
    await db.execute(
        update(SessionRecord).where(SessionRecord.id == session_id).values(**values)
    )
    await db.commit()
    return mid


def new_session_id() -> str:
    return uuid.uuid4().hex
