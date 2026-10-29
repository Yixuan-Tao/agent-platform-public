# Test values are synthetic fixtures for software behavior.
from types import SimpleNamespace
import pytest
from agent_platform.core.config import Settings
from agent_platform.core.llm.router import LLMRouter
from agent_platform.core.execution import ExecutionContext, execution_context, BudgetExceeded

async def test_llm_router_actual_usage_budget_and_frozen_model(monkeypatch):
    settings = Settings(_env_file=None, CHAT_MODEL="outside-model")
    router = LLMRouter(settings)
    requests, reservations, events = [], [], []
    class Completions:
        async def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
                usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}))
    monkeypatch.setattr(router, "_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    async def reserve(kind, amount=1):
        reservations.append((kind, amount))
    async def emit(kind, payload):
        events.append((kind, payload))
    ctx = ExecutionContext(actor="a", profile={"model": "frozen-model", "temperature": 0}, reserve=reserve, settle=reserve, emit=emit)
    token = execution_context.set(ctx)
    try:
        assert await router.chat([{"role": "user", "content": "question"}]) == "answer"
    finally:
        execution_context.reset(token)
    assert requests[0]["model"] == "frozen-model" and requests[0]["temperature"] == 0
    assert sum(amount for kind, amount in reservations if kind == "tokens") == 14
    assert events[-1][1]["usage_source"] == "actual" and events[-1][1]["cost"] is None

async def test_model_budget_exhausted_before_client_call(monkeypatch):
    router = LLMRouter(Settings(_env_file=None))
    monkeypatch.setattr(router, "_client", lambda: (_ for _ in ()).throw(AssertionError("No paid call permitted")))
    async def reserve(kind, amount=1):
        raise BudgetExceeded("budget exhausted")
    token = execution_context.set(ExecutionContext(actor="a", reserve=reserve))
    try:
        with pytest.raises(BudgetExceeded):
            await router.chat([{"role": "user", "content": "q"}])
    finally:
        execution_context.reset(token)

async def test_each_retry_reserves_model_call_and_emits_unknown_usage(monkeypatch):
    router = LLMRouter(Settings(_env_file=None))
    attempts, reservations, events = [], [], []
    class Completions:
        async def create(self, **kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise OSError("temporary failure")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))], usage=None)
    monkeypatch.setattr(router, "_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    async def reserve(kind, amount=1):
        reservations.append((kind, amount))
    async def emit(kind, payload):
        events.append(payload)
    token = execution_context.set(ExecutionContext(actor="alice", reserve=reserve, emit=emit))
    try:
        assert await router.chat([{"role": "user", "content": "q"}]) == "answer"
    finally:
        execution_context.reset(token)
    assert sum(n for kind, n in reservations if kind == "model_requests") == 2
    assert len([e for e in events if e["phase"] == "end"]) == 2
    assert all(e["usage_source"] == "estimated" and e["cost"] is None for e in events if e["phase"] == "end")
