"""Request-scoped delegated budget; no platform imports or credentials in telemetry."""
from __future__ import annotations

from contextvars import ContextVar, copy_context
import json
import math
import threading
import time
import uuid

import httpx


class RemoteBudgetExceeded(RuntimeError):
    pass


current_budget: ContextVar["RequestBudget | None"] = ContextVar("rag_request_budget", default=None)


def estimated_tokens(value):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return math.ceil(sum(1 if ord(c) > 127 else .25 for c in text))


def normalized_usage(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        return None
    total = value.get("total_tokens")
    if total is None and value.get("input_tokens") is not None and value.get("output_tokens") is not None:
        total = value["input_tokens"] + value["output_tokens"]
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        return {"total_tokens": total}
    return None


class RequestBudget:
    def __init__(self, grant):
        if not isinstance(grant, dict) or grant.get("version") != 1:
            raise ValueError("unsupported runtime contract")
        for key, cap in (("model_requests", 32), ("tokens", 100000)):
            if type(grant.get(key)) is not int or not 1 <= grant[key] <= cap:
                raise ValueError(f"invalid {key} grant")
        if not isinstance(grant.get("request_id"), str) or len(grant["request_id"]) != 32 or any(c not in "0123456789abcdef" for c in grant["request_id"]):
            raise ValueError("invalid request id")
        if not isinstance(grant.get("corpus_hash"), str) or len(grant["corpus_hash"]) != 64:
            raise ValueError("missing frozen corpus")
        if not isinstance(grant.get("run_id"),str) or len(grant["run_id"]) != 32 or any(c not in "0123456789abcdef" for c in grant["run_id"]):
            raise ValueError("invalid run id")
        deadline = grant.get("deadline")
        if type(deadline) not in (int, float) or not math.isfinite(deadline) or not time.time() < deadline <= time.time() + 660:
            raise ValueError("invalid or expired deadline")
        self.grant = grant
        self.lock = threading.Lock()
        self.calls = []
        self.tokens = 0
        self.error = None

    def begin(self, payload, output_limit=0):
        reserved = estimated_tokens(payload) + output_limit
        with self.lock:
            if self.error or time.time() >= self.grant["deadline"] or len(self.calls) >= self.grant["model_requests"] or self.tokens + reserved > self.grant["tokens"]:
                self.error = "RAG_BUDGET_EXHAUSTED"
                raise RemoteBudgetExceeded("delegated RAG budget exhausted")
            ticket = {"model_call_id": uuid.uuid4().hex, "model": str(payload.get("model", "unknown")),
                      "tokens_reserved": reserved, "usage": {"total_tokens": reserved}, "usage_source": "estimated",
                      "started": time.monotonic(), "error": "result_unknown"}
            self.calls.append(ticket)
            self.tokens += reserved
            return ticket

    def end(self, ticket, usage=None, error=None):
        actual = normalized_usage(usage)
        with self.lock:
            if actual is not None:
                self.tokens += actual["total_tokens"] - ticket["tokens_reserved"]
                ticket.update(usage=actual, usage_source="actual")
            ticket.update(duration=max(0, time.monotonic() - ticket["started"]), error=error)
            if self.tokens > self.grant["tokens"]:
                self.error = "RAG_BUDGET_EXHAUSTED"

    def report(self, corpus_hash):
        with self.lock:
            return {"version": 1, "request_id": self.grant["request_id"], "corpus_hash": corpus_hash,
                    "model_requests": len(self.calls), "tokens": self.tokens, "error": self.error,
                    "calls": [{k: v for k, v in call.items() if k != "started"} for call in self.calls]}


def model_post(url, **kwargs):
    budget = current_budget.get()
    if budget is None:
        return httpx.post(url, **kwargs)
    ticket = budget.begin(kwargs.get("json") or {})
    kwargs["timeout"] = min(float(kwargs.get("timeout", 60)), max(.01, budget.grant["deadline"] - time.time()))
    try:
        response = httpx.post(url, **kwargs)
        try:
            usage = response.json().get("usage")
        except (ValueError, AttributeError):
            usage = None
    except BaseException as exc:
        budget.end(ticket, error=type(exc).__name__)
        raise
    budget.end(ticket, usage, error=f"HTTP_{response.status_code}" if response.status_code >= 400 else None)
    return response


def model_chat(client, **kwargs):
    budget = current_budget.get()
    if budget is None:
        return client.chat.completions.create(**kwargs)
    if kwargs.get("stream"):
        budget.error = "RAG_STREAM_CONTRACT_UNSUPPORTED"
        raise RemoteBudgetExceeded("streaming is not a delegated tool contract")
    output = min(int(kwargs.get("max_tokens") or 4096), 4096)
    kwargs["max_tokens"] = output
    ticket = budget.begin(kwargs, output)
    try:
        # Retries belong to the existing explicit retry loops, each with a new reservation.
        response = client.with_options(max_retries=0, timeout=max(.01, min(60, budget.grant["deadline"]-time.time()))).chat.completions.create(**kwargs)
    except BaseException as exc:
        budget.end(ticket, error=type(exc).__name__)
        raise
    budget.end(ticket, getattr(response, "usage", None))
    return response


def submit_context(pool, fn, *args, **kwargs):
    # Each submission gets its own Context; the lock-protected budget object is shared.
    return pool.submit(copy_context().run, fn, *args, **kwargs)


def cache_namespace():
    budget = current_budget.get()
    return f"{budget.grant.get('run_id', budget.grant['request_id'])}|{budget.grant['corpus_hash']}|" if budget else ""
