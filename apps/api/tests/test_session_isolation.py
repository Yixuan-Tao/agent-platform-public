# Test values are synthetic fixtures for software behavior.
"""会话归属隔离、消息持久化与反馈溯源测试。"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agent_platform.core.db import Base
from app.routes.feedback import _resolve_qa
from app.routes.sessions import (
    append_message,
    ensure_session,
    list_messages,
    require_session_owner,
)


@pytest.fixture
async def db(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'platform.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


async def test_ensure_session_creates_record_for_owner(db):
    await ensure_session("s1", "alice", db)
    record = await require_session_owner("s1", "alice", db)
    assert record.user_id == "alice"


async def test_ensure_session_rejects_hijack(db):
    """已存在的会话不能被另一个用户重新绑定。"""
    await ensure_session("s1", "alice", db)
    with pytest.raises(HTTPException) as exc:
        await ensure_session("s1", "bob", db)
    assert exc.value.status_code == 403


async def test_require_session_owner_distinguishes_missing_and_foreign(db):
    with pytest.raises(HTTPException) as missing:
        await require_session_owner("nope", "alice", db)
    assert missing.value.status_code == 404

    await ensure_session("s2", "alice", db)
    with pytest.raises(HTTPException) as foreign:
        await require_session_owner("s2", "bob", db)
    assert foreign.value.status_code == 403


async def test_messages_persist_in_order_and_skip_foreign_reads(db):
    await ensure_session("s3", "alice", db)
    await append_message(
        db, session_id="s3", user_id="alice", role="user", content="VPN 连不上"
    )
    await append_message(
        db,
        session_id="s3",
        user_id="alice",
        role="assistant",
        content="先重启客户端",
        agent="it_support",
    )

    messages = await list_messages("s3", {"username": "alice"}, db)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].agent == "it_support"

    with pytest.raises(HTTPException) as exc:
        await list_messages("s3", {"username": "bob"}, db)
    assert exc.value.status_code == 403


async def test_resolve_qa_uses_recorded_question_and_answer(db):
    """反馈溯源必须拿到真实问题与答案，而不是 session_id 或 correction。"""
    await ensure_session("s4", "alice", db)
    await append_message(
        db, session_id="s4", user_id="alice", role="user", content="打印机卡纸怎么办"
    )
    answer_id = await append_message(
        db,
        session_id="s4",
        user_id="alice",
        role="assistant",
        content="打开后盖取出卡纸",
        agent="it_support",
    )

    question, answer = await _resolve_qa("s4", answer_id, db)
    assert question == "打印机卡纸怎么办"
    assert answer == "打开后盖取出卡纸"


async def test_resolve_qa_uses_latest_preceding_user_message(db):
    """多轮对话应配对到紧邻该回答之前的那次提问。"""
    await ensure_session("s5", "alice", db)
    await append_message(db, session_id="s5", user_id="alice", role="user", content="第一问")
    await append_message(db, session_id="s5", user_id="alice", role="assistant", content="第一答")
    await append_message(db, session_id="s5", user_id="alice", role="user", content="第二问")
    second_answer = await append_message(
        db, session_id="s5", user_id="alice", role="assistant", content="第二答"
    )

    question, answer = await _resolve_qa("s5", second_answer, db)
    assert question == "第二问"
    assert answer == "第二答"


async def test_resolve_qa_rejects_message_from_other_session(db):
    await ensure_session("s6", "alice", db)
    await ensure_session("s7", "alice", db)
    answer_id = await append_message(
        db, session_id="s6", user_id="alice", role="assistant", content="答"
    )
    with pytest.raises(HTTPException) as exc:
        await _resolve_qa("s7", answer_id, db)
    assert exc.value.status_code == 404
