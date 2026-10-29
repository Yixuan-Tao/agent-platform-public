"""Durable run ledger; SQL is authoritative, every worker mutation is lease-fenced."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import JSON, Float, Integer, String, Text, UniqueConstraint, select, update
from sqlalchemy.orm import Mapped, mapped_column

from agent_platform.core.db import Base, SessionRecord, ChatMessageRecord
from agent_platform.core.execution import BudgetExceeded, fingerprint

TERMINAL = {"completed", "failed", "cancelled"}


def uid() -> str:
    return uuid.uuid4().hex


class LeaseLost(RuntimeError):
    pass


class RunRecord(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    owner: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True, default="queued")
    goal: Mapped[str] = mapped_column(Text)
    snapshot: Mapped[dict] = mapped_column(JSON)
    budget: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    lease_owner: Mapped[str] = mapped_column(String(64), default="")
    lease_until: Mapped[float] = mapped_column(Float, default=0)
    epoch: Mapped[int] = mapped_column(Integer, default=0)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)


class EntryMixin:
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class StepRecord(EntryMixin, Base):
    __tablename__ = "run_steps"


class EventRecord(EntryMixin, Base):
    __tablename__ = "run_events"
    __table_args__ = (UniqueConstraint("run_id", "seq", name="uq_run_event_sequence"),)
    seq: Mapped[int] = mapped_column(Integer, index=True)


class ToolCallRecord(EntryMixin, Base):
    __tablename__ = "run_tool_calls"


class ApprovalRecord(EntryMixin, Base):
    __tablename__ = "run_approvals"


class ArtifactRecord(EntryMixin, Base):
    __tablename__ = "run_artifacts"


class MemoryRecord(EntryMixin, Base):
    __tablename__ = "evidence_memories"


class ProfileRecord(EntryMixin, Base):
    __tablename__ = "experiment_profiles"


class ExperimentRecord(EntryMixin, Base):
    __tablename__ = "experiment_batches"


class RunStore:
    def __init__(self, session_factory):
        self.factory = session_factory
        self._sqlite_lock = asyncio.Lock()

    @asynccontextmanager
    async def transaction(self):
        # SQLite has no SELECT FOR UPDATE; serialize this test/development adapter.
        async with self.factory() as db:
            sqlite = db.bind.dialect.name == "sqlite"
            if sqlite:
                await self._sqlite_lock.acquire()
            try:
                async with db.begin():
                    yield db
            finally:
                if sqlite:
                    self._sqlite_lock.release()

    async def _locked(self, db, run_id, fence=None, owner=None):
        run = await db.scalar(select(RunRecord).where(RunRecord.id == run_id).with_for_update())
        if run is None or (owner is not None and run.owner != owner):
            raise KeyError("run not found")
        if fence is not None:
            worker, epoch = fence
            if run.status != "running" or run.lease_owner != worker or run.epoch != epoch or run.lease_until <= time.time():
                raise LeaseLost("Worker 租约或执行代次已失效")
        return run

    @staticmethod
    def view(run):
        return {name: getattr(run, name) for name in ("id", "session_id", "owner", "status", "goal", "snapshot", "budget", "result", "epoch", "seq", "created_at", "updated_at")}

    async def _event(self, db, run, kind, payload, dedup_key=None):
        event_id = fingerprint({"run": run.id, "event": dedup_key}) if dedup_key else uid()
        if dedup_key and await db.get(EventRecord, event_id):
            return
        run.seq += 1
        run.updated_at = time.time()
        db.add(EventRecord(id=event_id, run_id=run.id, seq=run.seq,
                           payload={"type": kind, **payload}))

    async def create(self, *, owner, goal, snapshot, session_id=None):
        async with self.transaction() as db:
            session_id = session_id or uid()
            session = await db.scalar(select(SessionRecord).where(SessionRecord.id == session_id).with_for_update())
            if session and session.user_id != owner:
                raise KeyError("session not found")
            if session is None:
                db.add(SessionRecord(id=session_id, user_id=owner, title=goal[:100]))
            run = RunRecord(id=uid(), session_id=session_id, owner=owner, status="queued", goal=goal,
                            snapshot=snapshot, budget={}, result={}, epoch=0, seq=0,
                            lease_owner="", lease_until=0, created_at=time.time())
            db.add(run)
            db.add(ChatMessageRecord(message_id=run.id + "-user", session_id=session_id,
                                     user_id=owner, role="user", content=goal, run_id=run.id))
            await self._event(db, run, "status", {"status": "queued"})
            await db.flush()
            return self.view(run)

    async def get(self, run_id, owner=None):
        async with self.factory() as db:
            run = await db.get(RunRecord, run_id)
            if run is None or (owner is not None and run.owner != owner):
                raise KeyError("run not found")
            value = self.view(run)
            for name, model in (("steps", StepRecord), ("approvals", ApprovalRecord), ("artifacts", ArtifactRecord), ("tool_calls", ToolCallRecord)):
                rows = (await db.scalars(select(model).where(model.run_id == run_id).order_by(model.created_at))).all()
                value[name] = [{"id": row.id, **{k: v for k, v in row.payload.items() if k not in {"content_base64", "arguments"}}} for row in rows]
            return value

    async def list(self, owner, limit=100):
        async with self.factory() as db:
            rows = (await db.scalars(select(RunRecord).where(RunRecord.owner == owner).order_by(RunRecord.created_at.desc()).limit(limit))).all()
            return [self.view(row) for row in rows]

    async def events(self, run_id, owner, after=0):
        await self.get(run_id, owner)
        async with self.factory() as db:
            rows = (await db.scalars(select(EventRecord).where(EventRecord.run_id == run_id, EventRecord.seq > after).order_by(EventRecord.seq).limit(500))).all()
            return [{"seq": row.seq, **row.payload} for row in rows]

    async def claim(self, worker, limit=2, lease=60):
        claimed = []
        now = time.time()
        async with self.transaction() as db:
            rows = (await db.scalars(select(RunRecord).where(
                (RunRecord.status == "queued") | ((RunRecord.status == "running") & (RunRecord.lease_until < now))
            ).order_by(RunRecord.created_at).with_for_update(skip_locked=True).limit(100))).all()
            for run in rows:
                # Lock the owning session across worker processes before checking seriality.
                await db.scalar(select(SessionRecord).where(SessionRecord.id == run.session_id).with_for_update())
                blockers = await db.scalar(select(RunRecord.id).where(
                    RunRecord.session_id == run.session_id, RunRecord.id != run.id,
                    RunRecord.status.not_in(TERMINAL), RunRecord.created_at < run.created_at).limit(1))
                if blockers or run.session_id in {r["session_id"] for r in claimed}:
                    continue
                run.status, run.lease_owner, run.lease_until = "running", worker, now + lease
                run.epoch += 1
                await self._event(db, run, "status", {"status": "running", "epoch": run.epoch})
                claimed.append(self.view(run))
                if len(claimed) >= limit:
                    break
        return claimed

    async def heartbeat(self, run_id, fence, seconds=10):
        async with self.transaction() as db:
            run = await self._locked(db, run_id, fence)
            budget = dict(run.budget)
            budget["active_seconds"] = budget.get("active_seconds", 0) + seconds
            if budget["active_seconds"] > run.snapshot["profile"]["budget"]["active_seconds"]:
                raise BudgetExceeded("活跃执行时间额度耗尽")
            run.budget, run.lease_until = budget, time.time() + 60

    async def reserve(self, run_id, fence, kind, amount=1):
        async with self.transaction() as db:
            run = await self._locked(db, run_id, fence)
            budget = dict(run.budget)
            limits = run.snapshot["profile"]["budget"]
            if any(value > limits.get(key, 10**12) for key, value in budget.items()):
                raise BudgetExceeded("实际消耗已超过任务额度，禁止后续调用")
            value = budget.get(kind, 0) + amount
            if value > run.snapshot["profile"]["budget"].get(kind, 10**12):
                raise BudgetExceeded(f"{kind} 额度耗尽")
            budget[kind] = value
            run.budget = budget

    async def settle(self, run_id, fence, kind, difference):
        # Actual usage must remain visible even when an estimate was too small.
        async with self.transaction() as db:
            run = await self._locked(db, run_id, fence)
            budget = dict(run.budget)
            budget[kind] = max(0, budget.get(kind, 0) + difference)
            run.budget = budget

    async def rag_grant(self, run_id, fence, request_id, maximum_requests=8, maximum_tokens=32000):
        """Atomically reserve both counters and record a durable delegation before HTTP."""
        async with self.transaction() as db:
            run = await self._locked(db, run_id, fence)
            limits, budget = run.snapshot["profile"]["budget"], dict(run.budget)
            if any(value > limits.get(key, 10**12) for key, value in budget.items()):
                raise BudgetExceeded("实际消耗已超过任务额度")
            counts = {"model_requests": min(maximum_requests, limits["model_requests"]-budget.get("model_requests", 0)),
                      "tokens": min(maximum_tokens, limits["tokens"]-budget.get("tokens", 0))}
            if any(v <= 0 for v in counts.values()):
                raise BudgetExceeded("RAG 额度耗尽")
            event_key = "rag-grant-" + request_id
            if await db.get(EventRecord, fingerprint({"run": run_id, "event": event_key})):
                raise ValueError("RAG request id already delegated")
            for key, amount in counts.items():
                budget[key] = budget.get(key, 0) + amount
            run.budget = budget
            grant = {"version": 1, "request_id": request_id, "run_id": run_id, **counts,
                     "corpus_hash": run.snapshot["rag_snapshot"]["corpus_hash"],
                     "deadline": time.time() + max(.01, min(600, limits["active_seconds"]-budget.get("active_seconds", 0)))}
            await self._event(db, run, "rag_budget", {"phase": "reserved", "grant": grant}, event_key)
            return grant

    async def rag_settle(self, run_id, fence, request_id, report=None, step_id=""):
        """Settlement and imported usage are one fenced, idempotent transaction."""
        imported_events = []
        async with self.transaction() as db:
            run = await self._locked(db, run_id, fence)
            settled_key = "rag-settled-" + request_id
            if await db.get(EventRecord, fingerprint({"run": run_id, "event": settled_key})):
                return []
            record = await db.get(EventRecord, fingerprint({"run": run_id, "event": "rag-grant-" + request_id}))
            if record is None:
                raise ValueError("RAG grant not found")
            grant = record.payload["grant"]
            if report is None:
                # No trustworthy receipt: retain the ENTIRE reservation, including potential retries.
                payload = {"phase": "end", "model": "rag-service-unknown", "service": "rag",
                                  "model_call_id": request_id, "usage": {"total_tokens": grant["tokens"]},
                                  "usage_source": "estimated", "cost": None, "step_id": step_id, "error": "result_unknown"}
                await self._event(db, run, "model", payload, "rag-unknown-"+request_id)
                imported_events.append(payload)
            else:
                budget = dict(run.budget)
                for key in ("model_requests", "tokens"):
                    budget[key] = max(0, budget.get(key, 0) + report[key] - grant[key])
                run.budget = budget
                for call in report["calls"]:
                    common = {"model_call_id": call["model_call_id"], "model": call["model"], "service": "rag", "rag_request_id": request_id, "step_id": step_id}
                    start = {**common,"phase":"start","tokens_reserved":call["tokens_reserved"]}
                    end = {**common,"phase":"end","duration":call.get("duration",0),"usage":call["usage"],
                           "usage_source":call["usage_source"],"cost":None,"error":call.get("error")}
                    await self._event(db,run,"model",start)
                    await self._event(db,run,"model",end)
                    imported_events.extend([start,end])
            await self._event(db, run, "rag_budget", {"phase": "settled" if report is not None else "unknown", "request_id": request_id}, settled_key)
        return imported_events

    async def rag_recover_unknown(self, run_id, fence):
        async with self.factory() as db:
            events = (await db.scalars(select(EventRecord).where(EventRecord.run_id == run_id))).all()
            settled = {e.payload["request_id"] for e in events if e.payload.get("type") == "rag_budget" and e.payload.get("phase") in {"settled", "unknown"}}
            pending = [e.payload["grant"]["request_id"] for e in events if e.payload.get("type") == "rag_budget" and e.payload.get("phase") == "reserved" and e.payload["grant"]["request_id"] not in settled]
        for request_id in pending:
            await self.rag_settle(run_id, fence, request_id)

    async def emit(self, run_id, fence, kind, payload, dedup_key=None):
        async with self.transaction() as db:
            run = await self._locked(db, run_id, fence)
            await self._event(db, run, kind, payload, dedup_key)

    async def finish(self, run_id, fence, status, result=None):
        if status not in TERMINAL | {"waiting_input", "waiting_approval"}:
            raise ValueError("Invalid finish state")
        async with self.transaction() as db:
            run = await self._locked(db, run_id, fence)
            run.status = status
            run.result = result or run.result
            run.lease_until = 0
            steps = (await db.scalars(select(StepRecord).where(StepRecord.run_id == run_id))).all()
            for step in steps:
                if step.payload.get("status") == "running":
                    step.payload = {**step.payload, "status": status}
            await self._event(db, run, "status", {"status": status})

    async def cancel(self, run_id, owner):
        async with self.transaction() as db:
            run = await self._locked(db, run_id, owner=owner)
            if run.status not in TERMINAL:
                run.status, run.epoch, run.lease_until = "cancelled", run.epoch + 1, 0
                steps = (await db.scalars(select(StepRecord).where(StepRecord.run_id == run_id))).all()
                for step in steps:
                    if step.payload.get("status") not in TERMINAL:
                        step.payload = {**step.payload, "status": "cancelled"}
                await self._event(db, run, "status", {"status": "cancelled"})
        return await self.get(run_id, owner)

    async def resume(self, run_id, owner, inputs):
        async with self.transaction() as db:
            run = await self._locked(db, run_id, owner=owner)
            if run.status not in {"waiting_input", "waiting_approval", "failed"}:
                raise ValueError("任务当前不可恢复")
            approvals = (await db.scalars(select(ApprovalRecord).where(ApprovalRecord.run_id == run_id))).all()
            if any(a.payload.get("status") == "pending" for a in approvals):
                raise ValueError("请先处理审批卡片")
            unresolved = (await db.scalars(select(ToolCallRecord).where(ToolCallRecord.run_id == run_id))).all()
            if any(r.payload.get("status") in {"started", "unknown"} and not r.payload.get("read_only") for r in unresolved):
                raise ValueError("外部写结果未知，必须先核验，禁止直接重试")
            run.result = {**run.result, "resume_input": inputs, "resume_from_status": run.status}
            run.status = "queued"
            await self._event(db, run, "status", {"status": "queued", "reason": "resume"})
        return await self.get(run_id, owner)

    async def entry(self, model, entry_id):
        async with self.factory() as db:
            row = await db.get(model, entry_id)
            return dict(row.payload) if row else None

    async def put(self, model, run_id, fence, payload, entry_id=None):
        async with self.transaction() as db:
            await self._locked(db, run_id, fence)
            entry_id = entry_id or uid()
            row = await db.get(model, entry_id)
            if row and row.run_id != run_id:
                raise ValueError("Entry ownership mismatch")
            if row:
                row.payload = payload
            else:
                db.add(model(id=entry_id, run_id=run_id, payload=payload))
            return entry_id

    async def request_approval(self, run_id, fence, name, args):
        approval_id = fingerprint({"run": run_id, "tool": name, "args": args})
        async with self.transaction() as db:
            run = await self._locked(db, run_id, fence)
            row = await db.get(ApprovalRecord, approval_id)
            if row and row.payload.get("status") == "approved" and row.payload["expires_at"] > time.time():
                return "approved", approval_id
            if row and row.payload.get("status") == "rejected":
                return "rejected", approval_id
            preauthorized = any(action.get("tool") == name and fingerprint(action.get("parameters")) == fingerprint(args)
                                for action in run.snapshot.get("authorized_actions", [])) and run.created_at + 1800 > time.time()
            payload = dict(owner=run.owner, tool=name, parameters_hash=fingerprint(args),
                           parameters=args, expires_at=run.created_at + 1800 if preauthorized else time.time() + 1800,
                           status="approved" if preauthorized else "pending", source="explicit_run_authorization" if preauthorized else "approval_card")
            if row:
                row.payload = payload
            else:
                db.add(ApprovalRecord(id=approval_id, run_id=run_id, payload=payload))
            await self._event(db, run, "approval", {"approval_id": approval_id, **payload})
            return payload["status"], approval_id

    async def approve(self, run_id, owner, approval_id, approved, parameters_hash):
        async with self.transaction() as db:
            run = await self._locked(db, run_id, owner=owner)
            row = await db.get(ApprovalRecord, approval_id)
            if row is None or row.run_id != run_id or row.payload["owner"] != owner:
                raise KeyError("approval not found")
            payload = dict(row.payload)
            if payload["parameters_hash"] != parameters_hash or payload["expires_at"] <= time.time():
                raise ValueError("审批参数已变化或已过期")
            if payload["status"] == "pending":
                if run.status != "waiting_approval":
                    raise ValueError("任务尚未等待审批")
                payload["status"] = "approved" if approved else "rejected"
                row.payload = payload
                await db.flush()
                approvals = (await db.scalars(select(ApprovalRecord).where(ApprovalRecord.run_id == run_id))).all()
                await self._event(db, run, "approval", {"approval_id": approval_id, "status": payload["status"]})
                if not any(a.payload.get("status") == "pending" for a in approvals):
                    run.status = "queued"
                    run.result = {**run.result, "resume_input": {"approval_id": approval_id}}
                    await self._event(db, run, "status", {"status": "queued", "reason": "approval"})
        return await self.get(run_id, owner)

    async def artifact(self, run_id, fence, name, content, media_type="application/json"):
        raw = content.encode() if isinstance(content, str) else content
        if len(raw) > 10 * 1024 * 1024:
            raise ValueError("产物超过10MiB")
        async with self.transaction() as db:
            run = await self._locked(db, run_id, fence)
            digest = hashlib.sha256(raw).hexdigest()
            artifact_id = fingerprint({"run": run_id, "name": name, "sha256": digest, "media_type": media_type})
            if await db.get(ArtifactRecord, artifact_id):
                return artifact_id
            existing = (await db.scalars(select(ArtifactRecord).where(ArtifactRecord.run_id == run_id))).all()
            if sum(r.payload.get("size", 0) for r in existing) + len(raw) > 10 * 1024 * 1024:
                raise ValueError("任务产物总量超过10MiB")
            db.add(ArtifactRecord(id=artifact_id, run_id=run_id, payload=dict(name=name,
                media_type=media_type, size=len(raw), sha256=digest,
                content_base64=base64.b64encode(raw).decode())))
            await self._event(db, run, "artifact", {"artifact_id": artifact_id, "name": name})
            return artifact_id

    async def read_artifact(self, run_id, owner, artifact_id):
        await self.get(run_id, owner)
        async with self.factory() as db:
            row = await db.get(ArtifactRecord, artifact_id)
            if row is None or row.run_id != run_id:
                raise KeyError("artifact not found")
            return row.payload

    async def save_profile(self, profile):
        async with self.transaction() as db:
            entry_id = profile["version_hash"]
            row = await db.get(ProfileRecord, entry_id)
            if row and row.payload != profile:
                raise ValueError("历史配置不可修改")
            if not row:
                db.add(ProfileRecord(id=entry_id, run_id="", payload=profile))

    async def memory_candidates(self, owner, data_version, limit=100):
        from sqlalchemy import and_, or_
        async with self.factory() as db:
            payload = MemoryRecord.payload
            permitted = or_(and_(payload["owner"].as_string() == owner, payload["scope"].as_string() == "private", payload["state"].as_string() == "verified"),
                            and_(payload["scope"].as_string() == "shared", payload["state"].as_string() == "published"))
            rows = (await db.scalars(select(MemoryRecord).where(permitted, payload["data_version"].as_string() == data_version)
                                    .order_by(MemoryRecord.created_at.desc()).limit(limit))).all()
            return [dict(row.payload) for row in rows]

    async def preferences(self, owner):
        async with self.factory() as db:
            payload = MemoryRecord.payload
            rows = (await db.scalars(select(MemoryRecord).where(payload["owner"].as_string() == owner,
                payload["kind"].as_string() == "preference", payload["scope"].as_string() == "private",
                payload["state"].as_string() == "verified"))).all()
            return {row.payload["key"]: row.payload["value"] for row in rows}

    async def operational_metrics(self):
        from collections import Counter
        async with self.factory() as db:
            runs = (await db.scalars(select(RunRecord))).all()
            calls = (await db.scalars(select(ToolCallRecord))).all()
            approvals = (await db.scalars(select(ApprovalRecord))).all()
            events = (await db.scalars(select(EventRecord))).all()
        states = Counter(run.status for run in runs)
        denominator = sum(states[s] for s in TERMINAL)
        starts = {(e.run_id, e.payload.get("model_call_id")) for e in events if e.payload.get("type") == "model" and e.payload.get("phase") == "start"}
        ends = {(e.run_id, e.payload.get("model_call_id")) for e in events if e.payload.get("type") == "model" and e.payload.get("phase") == "end"}
        settled_rag = {(e.run_id,e.payload.get("request_id")) for e in events if e.payload.get("type") == "rag_budget" and e.payload.get("phase") in {"settled","unknown"}}
        return {"run_states": dict(states), "terminal_runs": denominator,
                "success_rate": states["completed"] / denominator if denominator else None,
                "expired_active_leases": sum(r.status == "running" and r.lease_until <= time.time() for r in runs),
                "pending_approvals": sum(a.payload.get("status") == "pending" for a in approvals),
                "unknown_writes": sum(c.payload.get("status") in {"started", "unknown"} and not c.payload.get("read_only", True) for c in calls),
                "model_errors": sum(e.payload.get("type") == "model" and bool(e.payload.get("error")) for e in events),
                "missing_model_end_events": len(starts - ends), "duplicate_external_effects": None,
                "unsettled_rag_grants": sum(e.payload.get("type") == "rag_budget" and e.payload.get("phase") == "reserved" and (e.run_id,e.payload["grant"]["request_id"]) not in settled_rag for e in events),
                "budget_exhausted": sum("BudgetExceeded" in r.result.get("error", "") for r in runs),
                "scope": "platform_ledger", "cost": None}

    async def reconcile(self, run_id, owner, tool_call_id, result):
        async with self.transaction() as db:
            run = await self._locked(db, run_id, owner=owner)
            call = await db.get(ToolCallRecord, tool_call_id)
            if not call or call.run_id != run_id or run.status not in {"waiting_input", "failed"}:
                raise ValueError("任务当前没有可核验的写动作")
            if result.get("status") != "success" or not any(r.get("verified") for r in result.get("external_refs", [])):
                raise ValueError("核验尚未确认实际外部结果，保持暂停")
            call.payload = {**call.payload, "status": "success", "external_refs": result["external_refs"],
                            "result": {"ok": True, "name": call.payload["name"], "result": json_text(result), **result}}
            run.status = "queued"
            run.result = {**run.result, "resume_input": {"tool_call_id": tool_call_id, "reconciled": True}}
            await self._event(db, run, "reconciliation", {"tool_call_id": tool_call_id, "status": "verified"})
        return await self.get(run_id, owner)

    async def batch(self, payload, batch_id=None):
        async with self.transaction() as db:
            batch_id = batch_id or uid()
            row = await db.get(ExperimentRecord, batch_id)
            if row:
                row.payload = payload
            else:
                db.add(ExperimentRecord(id=batch_id, run_id="", payload=payload))
            return batch_id


def json_text(value):
    import json
    return json.dumps(value, ensure_ascii=False, default=str)
