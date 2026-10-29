"""Fail-closed delegated RAG RPC, with actual-vs-estimated durable usage receipts."""
from __future__ import annotations

import asyncio
import json
import math
import uuid

import httpx

from agent_platform.core.execution import execution_context, step_context, BudgetExceeded


class RagContractError(RuntimeError):
    pass


def headers(settings):
    token = getattr(settings, "rag_service_token", "")
    return {"X-Local-Token": token, "Authorization": f"Bearer {token}"} if token else {}


async def fetch_snapshot(settings):
    try:
        async with httpx.AsyncClient(timeout=settings.rag_request_timeout) as client:
            response = await client.get(settings.rag_service_url.rstrip("/") + "/runtime/snapshot", headers=headers(settings))
            response.raise_for_status()
            value = response.json()
    except (httpx.HTTPError,ValueError) as exc:
        raise ValueError("RAG 语料快照暂时不可用；请完成服务验收后重试") from exc
    if value.get("version") != 1 or not isinstance(value.get("corpus_hash"), str) or len(value["corpus_hash"]) != 64:
        raise RagContractError("RAG 服务缺少版本化语料快照")
    return value


def validated_report(data, grant):
    report = data.get("_runtime") if isinstance(data, dict) else None
    if not isinstance(report, dict) or report.get("version") != 1 or report.get("request_id") != grant["request_id"]:
        raise RagContractError("RAG 未返回本次额度的可信回执")
    calls = report.get("calls")
    if not isinstance(calls, list) or type(report.get("model_requests")) is not int or report["model_requests"] != len(calls) or len(calls) > grant["model_requests"]:
        raise RagContractError("RAG 调用数量回执不一致")
    ids, total = set(), 0
    sanitized_calls = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("model_call_id"), str) or call["model_call_id"] in ids or not isinstance(call.get("model"), str):
            raise RagContractError("RAG 模型调用回执无效")
        ids.add(call["model_call_id"])
        amount = (call.get("usage") or {}).get("total_tokens")
        reserved = call.get("tokens_reserved")
        if type(amount) is not int or amount < 0 or type(reserved) is not int or not 0 <= reserved <= grant["tokens"] or call.get("usage_source") not in {"actual", "estimated"}:
            raise RagContractError("RAG token 回执无效")
        if call["usage_source"] == "estimated" and amount != reserved:
            raise RagContractError("RAG 估算回执与预留不一致")
        duration = call.get("duration", 0)
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise RagContractError("RAG 调用耗时回执无效")
        total += amount
        sanitized_calls.append({"model_call_id": call["model_call_id"], "model": call["model"][:128],
                                "tokens_reserved": reserved, "usage": {"total_tokens": amount},
                                "usage_source": call["usage_source"], "duration": duration, "error": str(call["error"])[:256] if call.get("error") else None})
    if type(report.get("tokens")) is not int or report["tokens"] != total or not isinstance(report.get("corpus_hash"), str):
        raise RagContractError("RAG token 总量回执不一致")
    return {"version": 1, "request_id": report["request_id"], "corpus_hash": report["corpus_hash"],
            "model_requests": len(calls), "tokens": total, "calls": sanitized_calls, "error": report.get("error")}


async def post(client, settings, path, payload, timeout=None):
    ctx = execution_context.get()
    active = ctx and ctx.run_id and ctx.resources and ctx.resources.get("rag_snapshot")
    grant, report = None, None
    if active:
        grant = await ctx.resources["store"].rag_grant(ctx.run_id, ctx.resources["fence"], uuid.uuid4().hex,
                    getattr(settings, "rag_delegate_model_requests", 8), getattr(settings, "rag_delegate_tokens", 32000))
    request_headers = headers(settings)
    if grant:
        request_headers["X-Agent-Runtime"] = json.dumps(grant, separators=(",", ":"))
    try:
        response = await client.post(settings.rag_service_url.rstrip("/") + path, json=payload, headers=request_headers,
                                     **({"timeout": timeout} if timeout else {}))
        data = response.json()
        if grant:
            report = validated_report(data, grant)
            # Settlement occurs even when corpus validation or remote execution failed.
            if report["corpus_hash"] != grant["corpus_hash"] or data.get("error_code") == "RAG_CORPUS_CHANGED":
                ctx.resources["rag_invalidation"] = "RAG 语料或索引已变化，请重建基线"
                raise RagContractError(ctx.resources["rag_invalidation"])
            if report.get("error"):
                raise BudgetExceeded(report["error"])
        response.raise_for_status()
        if grant:
            data.pop("_runtime", None)  # Telemetry is recorded in SQL, not repeated inside model evidence.
        return data
    except RagContractError as exc:
        if grant:
            ctx.resources["rag_invalidation"] = str(exc)
        raise
    except ValueError as exc:
        if grant:
            ctx.resources["rag_invalidation"] = "RAG 返回非 JSON 回执，外部消耗未知"
            raise RagContractError(ctx.resources["rag_invalidation"]) from exc
        raise
    finally:
        if grant:
            # Cancellation does not release a grant whose external outcome is unknown.
            imported = await asyncio.shield(ctx.resources["store"].rag_settle(ctx.run_id, ctx.resources["fence"], grant["request_id"], report, step_context.get()))
            exporter = ctx.resources.get("exporter")
            if exporter:
                for payload in imported or []:
                    try:
                        exporter.submit(ctx.run_id, "model", payload)
                    except Exception:
                        pass  # SQL receipt is authoritative; optional export cannot fail the RPC.
