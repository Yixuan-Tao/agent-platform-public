"""短期记忆：会话级消息历史（Redis/fakeredis，失败降级纯内存）。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from agent_platform.core.models import ChatMessage


class ShortTermMemory:
    """按 session_id 保存最近消息。"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        use_fakeredis: bool = True,
        ttl_seconds: int = 3600,
    ) -> None:
        self.ttl = ttl_seconds
        self._client = None
        self._fallback: dict[str, list[ChatMessage]] = {}
        self._lock = asyncio.Lock()
        try:
            if use_fakeredis:
                import fakeredis.aioredis as fr

                self._client = fr.FakeRedis(decode_responses=True)
            else:
                import redis.asyncio as aioredis

                self._client = aioredis.from_url(redis_url, decode_responses=True)
            logger.debug("ShortTermMemory 使用 Redis 后端")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 初始化失败，短期记忆降级为纯内存: {}", exc)
            self._client = None

    def _key(self, session_id: str) -> str:
        return f"agent_platform:session:{session_id}"

    async def get_messages(self, session_id: str) -> list[ChatMessage]:
        async with self._lock:
            return await self._load(session_id)

    async def append_message(self, session_id: str, message: ChatMessage) -> None:
        async with self._lock:
            messages = await self._load(session_id)
            messages.append(message)
            await self._save(session_id, messages)

    async def _load(self, session_id: str) -> list[ChatMessage]:
        """无锁读取（调用方需持有 self._lock）。"""
        if self._client is None:
            return list(self._fallback.get(session_id, []))
        raw = await self._client.get(self._key(session_id))
        if not raw:
            return []
        return [ChatMessage(**m) for m in json.loads(raw)]

    async def _save(self, session_id: str, messages: list[ChatMessage]) -> None:
        data = json.dumps([m.model_dump() for m in messages], ensure_ascii=False)
        if self._client is None:
            self._fallback[session_id] = messages
            return
        await self._client.set(self._key(session_id), data, ex=self.ttl)

    async def recent(self, session_id: str, k: int = 20) -> list[ChatMessage]:
        messages = await self.get_messages(session_id)
        return messages[-k:]

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            if self._client is None:
                self._fallback.pop(session_id, None)
            else:
                await self._client.delete(self._key(session_id))
