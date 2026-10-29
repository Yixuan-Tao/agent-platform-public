"""Request-local identity and durable execution hooks shared by all model/tool paths."""
from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode()).hexdigest()


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ExecutionContext:
    actor: str
    role: str = "user"
    run_id: str = ""
    profile: dict[str, Any] = field(default_factory=dict)
    graph_config: dict[str, Any] = field(default_factory=dict)
    checkpointer: Any = None
    reserve: Callable[..., Awaitable[Any]] | None = None
    emit: Callable[..., Awaitable[Any]] | None = None
    settle: Callable[..., Awaitable[Any]] | None = None
    resources: Any = None
    tool_before: Callable[..., Awaitable[Any]] | None = None
    tool_after: Callable[..., Awaitable[Any]] | None = None
    tool_start: Callable[..., Awaitable[Any]] | None = None
    tool_scopes: frozenset[str] = frozenset()
    permitted_tools: frozenset[str] | None = None


execution_context: ContextVar[ExecutionContext | None] = ContextVar("execution_context", default=None)
step_context: ContextVar[str] = ContextVar("step_context", default="")


async def reserve(kind: str, amount: int = 1) -> None:
    ctx = execution_context.get()
    if ctx and ctx.resources and ctx.resources.get("rag_invalidation"):
        raise RuntimeError(ctx.resources["rag_invalidation"])
    if ctx and ctx.reserve:
        await ctx.reserve(kind, amount)


async def emit(kind: str, **payload: Any) -> None:
    ctx = execution_context.get()
    if ctx and ctx.emit:
        payload.setdefault("step_id", step_context.get())
        await ctx.emit(kind, payload)


def is_graph_interrupt(exc: BaseException) -> bool:
    # Imported lazily to keep core usable without LangGraph.
    from langgraph.errors import GraphInterrupt
    return isinstance(exc, GraphInterrupt)


def estimate_tokens(text: str) -> int:
    # Conservative Unicode-aware fallback; flagged as estimated in telemetry.
    return sum(1 if ord(c) > 127 else 0.25 for c in text).__ceil__()
