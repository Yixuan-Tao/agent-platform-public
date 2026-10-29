"""Versioned, opt-in runtime contract for non-streaming platform RAG tools."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import time

from fastapi import APIRouter, Request, HTTPException
from starlette.responses import JSONResponse

from config import config
from db.chroma_client import ChromaClientSingleton
from utils.corpus_snapshot import runtime_snapshot
from utils.runtime_budget import RequestBudget, current_budget

router = APIRouter(prefix="/runtime")
CONTRACT_PATHS = {"/tools/retrieve", "/compare/emc", "/compare/component", "/compare/visualize"}


async def snapshot(app):
    provider = getattr(app.state, "runtime_snapshot_provider", None)
    if provider is not None:
        return await asyncio.to_thread(provider)
    return await asyncio.to_thread(runtime_snapshot, ChromaClientSingleton._instance, getattr(app.state, "evidence_store", None))


@router.get("/snapshot")
async def get_snapshot(request: Request):
    try:
        return await snapshot(request.app)
    except Exception as exc:
        raise HTTPException(503, "RAG 语料快照不可用") from exc


def register_request(grant):
    path = Path(config.RUNTIME_DIR) / "rag-runtime-contract.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=10) as db:
        db.execute("CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, deadline REAL NOT NULL, grant TEXT NOT NULL, report TEXT)")
        db.execute("DELETE FROM requests WHERE deadline < ?", (time.time()-86400,))
        try:
            db.execute("INSERT INTO requests(id,deadline,grant) VALUES (?,?,?)", (grant["request_id"], grant["deadline"], json.dumps(grant)))
        except sqlite3.IntegrityError:
            return False
    return True


def save_report(report):
    with sqlite3.connect(Path(config.RUNTIME_DIR) / "rag-runtime-contract.db", timeout=10) as db:
        db.execute("UPDATE requests SET report=? WHERE id=?", (json.dumps(report), report["request_id"]))


class RuntimeContractMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        header = dict(scope.get("headers", [])).get(b"x-agent-runtime")
        if scope["type"] != "http" or not header:
            return await self.app(scope, receive, send)
        budget = None
        try:
            if len(header) > 4096 or scope["method"] != "POST" or scope["path"] not in CONTRACT_PATHS:
                raise ValueError("unsupported runtime tool endpoint")
            budget = RequestBudget(json.loads(header))
            if not await asyncio.to_thread(register_request, budget.grant):
                await JSONResponse({"error_code": "RAG_REQUEST_REPLAY", "detail": "额度请求已领取，禁止重复执行"}, 409)(scope, receive, send)
                return
        except (ValueError, TypeError, UnicodeError):
            await JSONResponse({"error_code": "RAG_INVALID_GRANT"}, 400)(scope, receive, send)
            return
        except Exception:
            await JSONResponse({"error_code": "RAG_CONTRACT_STORAGE_UNAVAILABLE"}, 503)(scope, receive, send)
            return
        token = current_budget.set(budget)
        messages, body = [], bytearray()
        status, data = 500, {"error_code": "RAG_CONTRACT_FAILED"}
        corpus_hash = budget.grant["corpus_hash"]
        try:
            before = await snapshot(scope["app"])
            corpus_hash = before["corpus_hash"]
            if corpus_hash != budget.grant["corpus_hash"]:
                status, data = 409, {"error_code": "RAG_CORPUS_CHANGED"}
            else:
                async def capture(message):
                    if message["type"] == "http.response.start":
                        messages.append(message)
                    elif message["type"] == "http.response.body":
                        body.extend(message.get("body", b""))
                        if len(body) > 10*1024*1024:
                            raise ValueError("RAG tool response exceeds artifact limit")
                await self.app(scope, receive, capture)
                status = messages[0]["status"]
                data = json.loads(body)
                if not isinstance(data, dict):
                    data = {"results": data}
                after = await snapshot(scope["app"])
                corpus_hash = after["corpus_hash"]
                if before != after:
                    status, data = 409, {"error_code": "RAG_CORPUS_CHANGED"}
            if budget.error:
                status, data = 429, {"error_code": budget.error}
        except Exception:
            status, data = 503, {"error_code": "RAG_CONTRACT_FAILED"}
        finally:
            current_budget.reset(token)
        report = budget.report(corpus_hash)
        data["_runtime"] = report
        try:
            await asyncio.to_thread(save_report, report)
        except Exception:
            status, data = 503, {"error_code": "RAG_CONTRACT_STORAGE_UNAVAILABLE", "_runtime": report}
        await JSONResponse(data, status, headers={"X-Run-Id": str(budget.grant.get("run_id", ""))})(scope, receive, send)
