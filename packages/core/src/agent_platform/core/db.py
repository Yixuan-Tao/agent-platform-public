"""async SQLAlchemy 引擎与会话管理，以及核心表模型。"""

from __future__ import annotations

from datetime import datetime
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255), default="新会话")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    message_id: Mapped[str] = mapped_column(String(64))
    feedback_type: Mapped[str] = mapped_column(String(16))
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    correction: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ChatMessageRecord(Base):
    """持久化对话消息，供历史回放与反馈溯源使用。

    短期记忆（Redis/内存）会随 TTL 过期，反馈点赞也需要按 message_id 找回
    原始问题与答案，因此这里做一份关系型持久化。

    自增 ``id`` 用作稳定的写入顺序号：``created_at`` 在 SQLite 等后端只有秒级
    精度，仅靠时间戳排序会在同一秒内的多条消息上出现歧义。
    """

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="")
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text, default="")
    agent: Mapped[str] = mapped_column(String(64), default="")
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class AuditRecord(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    agent: Mapped[str] = mapped_column(String(64), default="unknown")
    action: Mapped[str] = mapped_column(String(64))
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def init_db(database_url: str) -> None:
    """初始化全局 async 引擎。进程启动时调用一次。"""
    global _engine, _session_factory
    _engine = create_async_engine(database_url, pool_pre_ping=True)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def create_all() -> None:
    """建表（开发环境用；生产建议迁移工具）。"""
    if _engine is None:
        raise RuntimeError("init_db() 未调用")
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：提供 AsyncSession。"""
    if _session_factory is None:
        raise RuntimeError("init_db() 未调用")
    async with _session_factory() as session:
        yield session


def session_factory():
    if _session_factory is None:
        raise RuntimeError("init_db() 未调用")
    return _session_factory


class MigrationRequired(RuntimeError):
    pass


async def verify_schema_compatibility():
    """Read-only check before startup create_all touches an existing database."""
    from sqlalchemy import inspect
    async with _engine.connect() as connection:
        def check(db):
            schema = inspect(db)
            names = set(schema.get_table_names())
            for table in ("chat_messages", "feedback"):
                if table in names and "run_id" not in {column["name"] for column in schema.get_columns(table)}:
                    raise MigrationRequired("现有数据库需要先备份，再执行 python -m alembic upgrade head；启动不会自动改写历史表")
        await connection.run_sync(check)


async def close_db():
    if _engine is not None:
        await _engine.dispose()


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """在响应外层（如 SSE 生成器、后台任务）使用的新会话。

    FastAPI 会在返回响应对象后立即关闭 ``Depends(get_session)`` 的会话，
    流式生成器运行时该会话已失效，因此需要在生成器内部单独开一个。
    """
    if _session_factory is None:
        raise RuntimeError("init_db() 未调用")
    async with _session_factory() as session:
        yield session
