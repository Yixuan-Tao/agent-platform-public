"""OpenAI 兼容 LLM 客户端，带 tenacity 重试与熔断器。"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Sequence

from openai import APITimeoutError, AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent_platform.core.config import Settings, get_settings
from agent_platform.core.models import ChatMessage


class CircuitBreaker:
    """简单熔断器：连续失败达到阈值后开启，冷却后半开。"""

    def __init__(self, threshold: int = 5, reset_seconds: int = 60) -> None:
        self.threshold = threshold
        self.reset_seconds = reset_seconds
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    async def check(self) -> bool:
        """返回 True 表示允许调用。"""
        async with self._lock:
            if self._opened_at is None:
                return True
            if time.time() - self._opened_at >= self.reset_seconds:
                self._opened_at = None
                self._failures = 0
                return True
            return False

    async def record_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._opened_at = None

    async def record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            if self._failures >= self.threshold:
                self._opened_at = time.time()


class LLMRouter:
    """统一 LLM 入口：按 Agent 覆盖模型、重试、熔断。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._clients: dict[str, AsyncOpenAI] = {}
        self._breaker = CircuitBreaker(
            threshold=self.settings.circuit_breaker_threshold,
            reset_seconds=self.settings.circuit_breaker_reset_seconds,
        )

    def resolve_model(self, agent_name: str | None = None) -> str:
        from agent_platform.core.execution import execution_context
        ctx = execution_context.get()
        if ctx and ctx.profile.get("model"):
            return ctx.profile["model"]
        return self.settings.resolve_model(agent_name)

    def _client(self) -> AsyncOpenAI:
        key = f"{self.settings.llm_base_url}|{self.settings.llm_api_key}"
        if key not in self._clients:
            self._clients[key] = AsyncOpenAI(**self.settings.llm_kwargs(), max_retries=0)
        return self._clients[key]

    @asynccontextmanager
    async def breaker_guard(self):
        """熔断守卫：允许调用则进入，成功/失败分别记账。

        供绕过 :meth:`chat` 的路径（专家 Agent / Router 图内部直接使用
        ChatOpenAI）复用同一熔断状态。
        """
        if not await self._breaker.check():
            raise RuntimeError("circuit breaker open: 暂缓调用 LLM")
        try:
            yield
        except Exception:
            await self._breaker.record_failure()
            raise
        else:
            await self._breaker.record_success()

    async def chat(
        self,
        messages: Sequence[ChatMessage | dict[str, str]],
        agent_name: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """单轮补全，返回文本。失败自动重试。"""
        if not await self._breaker.check():
            raise RuntimeError("circuit breaker open: 暂缓调用 LLM")
        model = self.resolve_model(agent_name)
        try:
            resp = await self._retry_chat(messages, model, temperature, kwargs)
            await self._breaker.record_success()
            return resp
        except Exception:
            await self._breaker.record_failure()
            raise

    @retry(
        retry=retry_if_exception_type((APITimeoutError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _retry_chat(
        self,
        messages: Sequence[ChatMessage | dict[str, str]],
        model: str,
        temperature: float,
        kwargs: dict[str, Any],
    ) -> str:
        payload = [
            m.model_dump() if isinstance(m, ChatMessage) else m for m in messages
        ]
        from agent_platform.core.execution import execution_context
        from agent_platform.core.llm.telemetry import begin, end
        ctx = execution_context.get()
        if ctx:
            temperature = ctx.profile.get("temperature", temperature)
            kwargs.setdefault("max_tokens", 4096)
        ticket = await begin(payload, model, kwargs.get("max_tokens", 4096))
        try:
            resp = await self._client().chat.completions.create(
                model=model, messages=payload, temperature=temperature, **kwargs
            )
        except Exception as exc:
            await end(ticket, model, error=exc)
            raise
        await end(ticket, model, resp.usage.model_dump() if resp.usage else None)
        return resp.choices[0].message.content or ""

    async def stream(
        self,
        messages: Sequence[ChatMessage | dict[str, str]],
        agent_name: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """流式补全，逐 token 产出文本。"""
        if not await self._breaker.check():
            raise RuntimeError("circuit breaker open: 暂缓调用 LLM")
        model = self.resolve_model(agent_name)
        payload = [
            m.model_dump() if isinstance(m, ChatMessage) else m for m in messages
        ]
        from agent_platform.core.execution import execution_context
        from agent_platform.core.llm.telemetry import begin, end
        ctx = execution_context.get()
        if ctx:
            temperature = ctx.profile.get("temperature", temperature)
            kwargs.setdefault("max_tokens", 4096)
        ticket = await begin(payload, model, kwargs.get("max_tokens", 4096))
        usage, error = None, None
        try:
            stream = await self._client().chat.completions.create(model=model, messages=payload,
                    temperature=temperature, stream=True, **kwargs)
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage.model_dump()
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            await self._breaker.record_success()
        except Exception as exc:
            error = exc
            await self._breaker.record_failure()
            raise
        finally:
            await end(ticket, model, usage, error)


_router: LLMRouter | None = None


def get_llm_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
