"""反馈闭环接口：点赞/点踩/重试，并溯源到原始问答。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_platform.core.db import ChatMessageRecord, FeedbackRecord, get_session
from agent_platform.core.models import Feedback
from agent_platform.core.security import require_login
from agent_platform.memory.service import MemoryService
from app.dependencies import get_memory
from app.routes.sessions import require_session_owner

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


async def _resolve_qa(
    session_id: str, message_id: str, db: AsyncSession
) -> tuple[str, str]:
    """按 message_id 找到助手回答，并回溯同一会话中它之前的最后一条用户提问。"""
    result = await db.execute(
        select(ChatMessageRecord).where(
            ChatMessageRecord.message_id == message_id,
            ChatMessageRecord.session_id == session_id,
        )
    )
    answer_msg = result.scalar_one_or_none()
    if answer_msg is None:
        raise HTTPException(status_code=404, detail="消息不存在或不属于该会话")

    result = await db.execute(
        select(ChatMessageRecord)
        .where(
            ChatMessageRecord.session_id == session_id,
            ChatMessageRecord.role == "user",
            ChatMessageRecord.id < answer_msg.id,
        )
        .order_by(ChatMessageRecord.id.desc())
        .limit(1)
    )
    question_msg = result.scalar_one_or_none()
    question = question_msg.content if question_msg else ""
    return question, answer_msg.content or ""


@router.post("", dependencies=[Depends(require_login)])
async def submit_feedback(
    feedback: Feedback,
    identity: dict = Depends(require_login),
    memory: MemoryService = Depends(get_memory),
    db: AsyncSession = Depends(get_session),
) -> dict:
    await require_session_owner(feedback.session_id, identity["username"], db)
    question, answer = await _resolve_qa(feedback.session_id, feedback.message_id, db)
    message = await db.scalar(select(ChatMessageRecord).where(ChatMessageRecord.message_id == feedback.message_id))
    if feedback.run_id and message.run_id != feedback.run_id:
        raise HTTPException(409, "run_id 与服务端消息记录不匹配")

    # 修正内容优先作为答案入库；普通点赞使用服务端记录的原答案。
    effective_answer = feedback.correction or answer
    from agent_platform.orchestrator.runtime_store import MemoryRecord
    from agent_platform.core.execution import fingerprint
    if feedback.feedback_type == "up":
        candidate_id = fingerprint({"owner": identity["username"], "session": feedback.session_id, "message": feedback.message_id})
        candidate = await db.get(MemoryRecord, candidate_id)
        payload = {"owner": identity["username"], "scope": "private", "state": "candidate",
                   "question": question, "answer": effective_answer, "session_id": feedback.session_id,
                   "message_id": feedback.message_id, "run_id": message.run_id,
                   "corrected": bool(feedback.correction), "source": "feedback"}
        if candidate:
            candidate.payload = payload
        else:
            db.add(MemoryRecord(id=candidate_id, run_id=message.run_id or "", payload=payload))
    db.add(
        FeedbackRecord(
            session_id=feedback.session_id,
            message_id=feedback.message_id,
            feedback_type=feedback.feedback_type,
            reason=feedback.reason,
            correction=feedback.correction,
            run_id=message.run_id,
        )
    )
    await db.commit()

    await memory.feedback.record_with_qa(
        feedback,
        question=question,
        answer=effective_answer,
        owner=identity["username"],
    )
    return {"status": "ok", "feedback_type": feedback.feedback_type}


@router.post("/undo", dependencies=[Depends(require_login)])
async def undo_feedback(
    body: dict,
    identity: dict = Depends(require_login),
    memory: MemoryService = Depends(get_memory),
    db: AsyncSession = Depends(get_session),
) -> dict:
    session_id = body.get("session_id", "")
    message_id = body.get("message_id", "")
    await require_session_owner(session_id, identity["username"], db)
    ok = await memory.feedback.undo(session_id, message_id)
    memory.long_term.revoke_source(identity["username"], session_id, message_id)
    from agent_platform.orchestrator.runtime_store import MemoryRecord
    from agent_platform.core.execution import fingerprint
    candidate = await db.get(MemoryRecord, fingerprint({"owner": identity["username"], "session": session_id, "message": message_id}))
    if candidate:
        candidate.payload = {**candidate.payload, "state": "revoked"}
        await db.commit()
        ok = True
    return {"status": "ok" if ok else "noop"}
