"""Authenticated durable-run and experiment endpoints."""
import asyncio
import base64
import json
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from agent_platform.core.security import require_login, require_admin
from agent_platform.orchestrator.runtime_store import TERMINAL, ExperimentRecord

router = APIRouter(prefix="/api/v1", tags=["runs"])

class CreateRun(BaseModel):
    goal: str = Field(min_length=1, max_length=32000)
    session_id: str | None = None
    profile: str = "B0"
    data_version: str = "unversioned"
    authorized_actions: list[dict[str, Any]] = Field(default_factory=list, max_length=8)

class ResumeRun(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)

class Approval(BaseModel):
    approved: bool
    parameters_hash: str

class Reconciliation(BaseModel):
    external_key: str | None = None

class ResumeExperiment(BaseModel):
    allow_paid_models: bool = False

class Experiment(BaseModel):
    profiles: list[str] = Field(default_factory=lambda: ["B0", "B1", "B2", "C1", "C2", "P1", "M1"], min_length=1, max_length=10)
    split: str = "development"
    repetitions: int = Field(default=3, ge=1, le=3)
    data_version: str
    allow_paid_models: bool = False

def runtime(request: Request):
    service = getattr(request.app.state, "run_service", None)
    if service is None:
        raise HTTPException(503, "任务运行功能未启用；需要配置 RUNS_ENABLED 并完成数据库迁移")
    return service

async def checked(awaitable):
    try:
        return await awaitable
    except KeyError:
        raise HTTPException(404, "记录不存在")
    except (ValueError, PermissionError) as exc:
        raise HTTPException(409, str(exc))

def scoped(identity, request):
    settings = request.app.state.settings
    return {**identity, "scopes": settings.service_tool_scopes if identity["role"] == "service" else settings.user_tool_scopes}

@router.post("/runs", status_code=202)
async def create_run(body: CreateRun, request: Request, identity=Depends(require_login)):
    return await checked(runtime(request).create(scoped(identity, request), body.goal, body.session_id, body.profile, body.data_version,
                        authorized_actions=body.authorized_actions))

@router.get("/runs")
async def list_runs(request: Request, identity=Depends(require_login)):
    return await runtime(request).store.list(identity["username"])

@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request, identity=Depends(require_login)):
    return await checked(runtime(request).store.get(run_id, identity["username"]))

@router.get("/runs/{run_id}/events")
async def events(run_id: str, request: Request, after: int = 0, identity=Depends(require_login)):
    service = runtime(request)
    await checked(service.store.get(run_id, identity["username"]))
    async def stream():
        cursor = max(after, 0)
        while not await request.is_disconnected():
            items = await service.store.events(run_id, identity["username"], cursor)
            for event in items:
                cursor = event["seq"]
                yield {"id": str(cursor), "data": json.dumps(event, ensure_ascii=False, default=str)}
            run = await service.store.get(run_id, identity["username"])
            if run["status"] in TERMINAL and cursor >= run["seq"]:
                return
            await asyncio.sleep(0.3)
    return EventSourceResponse(stream())

@router.post("/runs/{run_id}/resume")
async def resume(run_id: str, body: ResumeRun, request: Request, identity=Depends(require_login)):
    return await checked(runtime(request).store.resume(run_id, identity["username"], body.inputs))

@router.post("/runs/{run_id}/cancel")
async def cancel(run_id: str, request: Request, identity=Depends(require_login)):
    return await checked(runtime(request).store.cancel(run_id, identity["username"]))

@router.post("/runs/{run_id}/approvals/{approval_id}")
async def approve(run_id: str, approval_id: str, body: Approval, request: Request, identity=Depends(require_login)):
    return await checked(runtime(request).store.approve(run_id, identity["username"], approval_id, body.approved, body.parameters_hash))

@router.get("/runs/{run_id}/artifacts/{artifact_id}")
async def artifact(run_id: str, artifact_id: str, request: Request, identity=Depends(require_login)):
    item = await checked(runtime(request).store.read_artifact(run_id, identity["username"], artifact_id))
    return Response(base64.b64decode(item["content_base64"]), media_type=item["media_type"],
                    headers={"Content-Disposition": "attachment; filename*=UTF-8''" + __import__("urllib.parse", fromlist=["quote"]).quote(item["name"])})

@router.post("/runs/{run_id}/tool-calls/{tool_call_id}/reconcile")
async def reconcile(run_id: str, tool_call_id: str, body: Reconciliation, request: Request, identity=Depends(require_login)):
    service = runtime(request)
    run = await checked(service.store.get(run_id, identity["username"]))
    if "jira:read" not in scoped(identity, request)["scopes"]:
        raise HTTPException(403, "缺少工单查询权限")
    from agent_platform.orchestrator.runtime_store import ToolCallRecord
    from agent_platform.tools.builtin.jira_tool import reconcile_jira
    call = await service.store.entry(ToolCallRecord, tool_call_id)
    if not call or tool_call_id not in {c["id"] for c in run["tool_calls"]} or call["name"] != "jira_create_ticket":
        raise HTTPException(404, "工单动作不存在")
    result = await reconcile_jira(run_id, call["arguments"], body.external_key)
    return await checked(service.store.reconcile(run_id, identity["username"], tool_call_id, result.model_dump()))

@router.get("/experiment-profiles")
async def get_profiles(request: Request, identity=Depends(require_login)):
    return runtime(request).profiles()

@router.get("/runtime/metrics")
async def runtime_metrics(request: Request, identity=Depends(require_admin)):
    return await runtime(request).store.operational_metrics()

@router.post("/experiments", status_code=202)
async def experiment(body: Experiment, request: Request, identity=Depends(require_admin)):
    runner = getattr(request.app.state, "experiment_runner", None)
    if runner is None:
        raise HTTPException(503, "实验功能未启用")
    if not body.allow_paid_models:
        raise HTTPException(422, "真实模型实验需手动声明 allow_paid_models=true")
    return await checked(runner.create(scoped(identity, request), body.model_dump()))

@router.get("/experiments")
async def experiments(request: Request, identity=Depends(require_admin)):
    from sqlalchemy import select
    async with runtime(request).store.factory() as db:
        rows = (await db.scalars(select(ExperimentRecord).order_by(ExperimentRecord.created_at.desc()).limit(100))).all()
        return [{"id": row.id, **row.payload} for row in rows if row.payload.get("owner") == identity["username"]]

@router.get("/experiments/{batch_id}")
async def get_experiment(batch_id: str, request: Request, identity=Depends(require_admin)):
    payload = await runtime(request).store.entry(ExperimentRecord, batch_id)
    if payload is None or payload.get("owner") != identity["username"]:
        raise HTTPException(404, "实验不存在")
    return {"id": batch_id, **payload}

@router.post("/experiments/{batch_id}/resume", status_code=202)
async def resume_experiment(batch_id: str, body: ResumeExperiment, request: Request, identity=Depends(require_admin)):
    runner = getattr(request.app.state, "experiment_runner", None)
    if runner is None:
        raise HTTPException(503, "实验功能未启用")
    if not body.allow_paid_models:
        raise HTTPException(422, "恢复付费模型批次需显式声明 allow_paid_models=true")
    return await checked(runner.resume(batch_id, scoped(identity, request)))
