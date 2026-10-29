"""Verified private examples; only explicit administrator publication creates shared ones."""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from pydantic import BaseModel, Field, model_validator
from typing import Literal
from agent_platform.core.execution import fingerprint
from agent_platform.orchestrator.runtime_store import uid
from agent_platform.core.db import get_session
from agent_platform.core.security import require_login, require_admin
from agent_platform.orchestrator.runtime_store import MemoryRecord, RunRecord
from app.dependencies import get_memory

router = APIRouter(prefix="/api/v1/memories", tags=["memories"])

class Preference(BaseModel):
    key: Literal["reply_language", "detail_level", "report_format"]
    value: str = Field(max_length=32)
    @model_validator(mode="after")
    def valid_value(self):
        values = {"reply_language": {"zh-CN", "en", "auto"}, "detail_level": {"concise", "normal", "detailed"}, "report_format": {"markdown", "plain", "json"}}
        if self.value not in values[self.key]:
            raise ValueError("不支持的偏好值")
        return self

@router.post("/preferences")
async def save_preference(body: Preference, identity=Depends(require_login), db=Depends(get_session)):
    entry_id = fingerprint({"owner": identity["username"], "preference": body.key})
    row = await db.get(MemoryRecord, entry_id)
    payload = {"owner": identity["username"], "scope": "private", "state": "verified", "kind": "preference",
               "key": body.key, "value": body.value, "source": "explicit_user_input", "data_version": "*"}
    if row:
        history = list(row.payload.get("history", [])) + [{"value": row.payload["value"]}]
        row.payload = {**payload, "history": history}
    else:
        db.add(MemoryRecord(id=entry_id, run_id="", payload=payload))
    await db.commit()
    return {"id": entry_id, **payload}

@router.post("/experiences/{run_id}")
async def save_experience(run_id: str, identity=Depends(require_login), db=Depends(get_session), memory=Depends(get_memory)):
    run = await db.get(RunRecord, run_id)
    if run is None or run.owner != identity["username"]:
        raise HTTPException(404, "任务不存在")
    if run.status != "completed" or run.result.get("verification", {}).get("status") != "pass":
        raise HTTPException(409, "任务经验只能来自已完成且核验通过的原始结果")
    entry_id = fingerprint({"owner": identity["username"], "experience": run_id})
    row = await db.get(MemoryRecord, entry_id)
    payload = {"owner": identity["username"], "scope": "private", "state": "verified", "kind": "task_experience",
               "question": run.goal, "answer": run.result["answer"], "data_version": run.snapshot["data_version"],
               "run_id": run_id, "source": "verified_run", "session_id": run.session_id, "message_id": run_id + "-answer"}
    if not row:
        db.add(MemoryRecord(id=entry_id, run_id=run_id, payload=payload))
    await memory.feedback.undo(payload["session_id"], payload["message_id"])
    memory.long_term.add_qa(payload["question"], payload["answer"], owner=run.owner, state="verified", meta=payload)
    await db.commit()
    return {"id": entry_id, **payload}

@router.get("")
async def list_memories(identity=Depends(require_login), db=Depends(get_session)):
    rows = (await db.scalars(select(MemoryRecord).where(MemoryRecord.payload["owner"].as_string() == identity["username"])) ).all()
    return [{"id": r.id, **r.payload} for r in rows if r.payload.get("owner") == identity["username"]]

@router.post("/{memory_id}/verify")
async def verify(memory_id: str, identity=Depends(require_login), db=Depends(get_session), memory=Depends(get_memory)):
    row = await db.get(MemoryRecord, memory_id)
    if row is None or row.payload.get("owner") != identity["username"]:
        raise HTTPException(404, "记忆不存在")
    payload = row.payload
    run = await db.get(RunRecord, row.run_id)
    if payload.get("state") == "revoked" or payload.get("corrected") or not run or run.owner != identity["username"] or run.result.get("verification", {}).get("status") != "pass":
        raise HTTPException(409, "原始问答缺少验证通过的运行证据；修正答案需要重新执行验证")
    payload = {**payload, "state": "verified", "data_version": run.snapshot["data_version"]}
    row.payload = payload
    await memory.feedback.undo(payload["session_id"], payload["message_id"])
    memory.long_term.add_qa(payload["question"], payload["answer"], owner=payload["owner"], state="verified",
                           meta={"session_id": payload["session_id"], "message_id": payload["message_id"], "run_id": row.run_id, "data_version": payload["data_version"]})
    await db.commit()
    return {"id": row.id, **payload}

@router.post("/{memory_id}/publish")
async def publish(memory_id: str, identity=Depends(require_admin), db=Depends(get_session), memory=Depends(get_memory)):
    row = await db.get(MemoryRecord, memory_id)
    if row is None or row.payload.get("state") != "verified" or row.payload.get("kind") == "preference":
        raise HTTPException(409, "只能发布验证通过的私有示例")
    payload = {**row.payload, "state": "published", "scope": "shared", "publisher": identity["username"]}
    memory.long_term.add_qa(payload["question"], payload["answer"], owner=payload["owner"], scope="shared", state="published",
                           meta={"session_id": payload["session_id"], "message_id": payload["message_id"], "run_id": row.run_id, "data_version": payload["data_version"]})
    row.payload = payload
    await db.commit()
    return {"id": row.id, **payload}
