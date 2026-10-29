# Test values are synthetic fixtures for software behavior.
from pathlib import Path
import asyncio
import time
import os
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import update
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START, END
from agent_platform.core.db import Base
from agent_platform.core.execution import BudgetExceeded, execution_context
from agent_platform.core.models import AgentManifest
from agent_platform.orchestrator.runtime_store import RunStore, RunRecord, LeaseLost
from agent_platform.orchestrator.runtime import RunService, Worker
from agent_platform.orchestrator.registry import AgentRegistry
from agent_platform.orchestrator.service import OrchestratorService
from agent_platform.orchestrator.context import ContextBuilder, ContextOverflow
from agent_platform.skills.registry import SkillRegistry
from agent_platform.tools.base import BaseTool, ToolRegistry
from agent_platform.tools.gateway import ToolGateway
from agent_platform.tools.result import ToolResult

ROOT = Path(__file__).resolve().parents[3]

@pytest.fixture
async def store(tmp_path):
    database_url = os.environ.get("PLATFORM_TEST_DATABASE_URL") or f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}"
    if os.environ.get("PLATFORM_TEST_DATABASE_URL"):
        assert database_url.endswith("/agent_platform_test"), "Integration fixture may only reset the dedicated test database"
    engine = create_async_engine(database_url)
    async with engine.begin() as connection:
        if os.environ.get("PLATFORM_TEST_DATABASE_URL"):
            await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    yield RunStore(async_sessionmaker(engine, expire_on_commit=False))
    await engine.dispose()

class LLM:
    settings = None
    async def chat(self, messages, **kwargs):
        return "PASS"
    def resolve_model(self, agent=None):
        return "test-model"

def service(store, graph=None):
    registry = AgentRegistry()
    if graph is None:
        class Graph:
            async def ainvoke(self, state, config=None):
                return {"final_answer": "已完成诊断"}
        graph = Graph()
    registry.register(AgentManifest(name="it_support", description="IT"), lambda *_: graph)
    return RunService(store, OrchestratorService(registry, LLM(), ToolRegistry()), SkillRegistry(ROOT / "skills"), "test-model")

IDENTITY = {"username": "alice", "role": "user", "scopes": ["jira:create", "jira:read"]}

async def test_replayed_produce_does_not_duplicate_artifacts_or_answer_events(store):
    svc = service(store)
    await svc.create(IDENTITY, "生成报告")
    run = (await store.claim("worker"))[0]
    fence = ("worker", run["epoch"])
    first = await store.artifact(run["id"], fence, "report.md", "证据报告", "text/markdown")
    second = await store.artifact(run["id"], fence, "report.md", "证据报告", "text/markdown")
    assert first == second
    for _ in range(2):
        await store.emit(run["id"], fence, "answer_delta", {"content": "证据报告"}, dedup_key="final-answer-0")
    current = await store.get(run["id"])
    events = await store.events(run["id"], "alice")
    assert len(current["artifacts"]) == 1
    assert len([e for e in events if e["type"] == "answer_delta"]) == 1
    assert len({e["seq"] for e in events}) == len(events)

async def test_model_configuration_change_prevents_resume(store, tmp_path):
    svc = service(store)
    await svc.create(IDENTITY, "故障")
    run = (await store.claim("worker"))[0]
    svc.model = "different-model"
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.db")) as saver:
        worker = Worker(svc, saver)
        worker.id = "worker"
        await worker.execute(run)
    current = await store.get(run["id"])
    assert current["status"] == "failed"
    assert "配置已变化" in current["result"]["error"]

async def test_service_cannot_freeze_changed_disk_source_under_old_loaded_code(store,monkeypatch):
    svc = service(store)
    monkeypatch.setattr(svc,"code_version",lambda:"changed-source")
    with pytest.raises(ValueError,match="重启"):
        await svc.create(IDENTITY,"fixture")

async def test_owner_session_seriality_cancel_and_fencing(store):
    svc = service(store)
    first = await svc.create(IDENTITY, "VPN 故障")
    second = await svc.create(IDENTITY, "另一故障", session_id=first["session_id"])
    await svc.create({**IDENTITY, "username": "bob"}, "VPN")
    with pytest.raises(KeyError):
        await store.get(first["id"], "bob")
    claimed = await store.claim("worker")
    assert len(claimed) == 2
    assert second["id"] not in {r["id"] for r in claimed}
    fence = ("worker", next(r["epoch"] for r in claimed if r["id"] == first["id"]))
    await store.cancel(first["id"], "alice")
    with pytest.raises(LeaseLost):
        await store.emit(first["id"], fence, "answer", {})
    assert (await store.claim("worker2"))[0]["id"] == second["id"]

async def test_budget_recovery_never_resets(store):
    svc = service(store)
    run = await svc.create(IDENTITY, "故障")
    run = (await store.claim("one"))[0]
    await store.reserve(run["id"], ("one", run["epoch"]), "tool_attempts", 32)
    await store.finish(run["id"], ("one", run["epoch"]), "failed")
    await store.resume(run["id"], "alice", {})
    restored = (await store.claim("two"))[0]
    with pytest.raises(BudgetExceeded):
        await store.reserve(run["id"], ("two", restored["epoch"]), "tool_attempts")

async def test_actual_usage_overrun_is_recorded_and_blocks_next_effect(store):
    svc = service(store)
    await svc.create(IDENTITY, "故障")
    run = (await store.claim("worker"))[0]
    fence = ("worker", run["epoch"])
    await store.reserve(run["id"], fence, "tokens", 99000)
    await store.settle(run["id"], fence, "tokens", 2000)
    assert (await store.get(run["id"]))["budget"]["tokens"] == 101000
    with pytest.raises(BudgetExceeded):
        await store.reserve(run["id"], fence, "tool_attempts")

async def test_remote_grant_settlement_is_atomic_fenced_and_idempotent(store):
    svc = service(store)
    run = await svc.create(IDENTITY, "fixture")
    # A test-only snapshot for an explicitly delegated external service.
    async with store.transaction() as db:
        row = await db.get(RunRecord,run["id"])
        row.snapshot = {**row.snapshot,"rag_snapshot":{"corpus_hash":"a"*64}}
    run = (await store.claim("worker"))[0]
    fence = ("worker",run["epoch"])
    grant = await store.rag_grant(run["id"],fence,"a"*32)
    assert grant["tokens"] == 32000 and grant["model_requests"] == 8
    assert (await store.get(run["id"]))["budget"]["tokens"] == 32000
    report = {"tokens":100,"model_requests":1,"calls":[{"model_call_id":"m", "model":"embed","tokens_reserved":120,"usage":{"total_tokens":100},"usage_source":"actual"}]}
    await store.rag_settle(run["id"],fence,grant["request_id"],report)
    await store.rag_settle(run["id"],fence,grant["request_id"],report)
    current = await store.get(run["id"])
    assert current["budget"]["tokens"] == 100 and current["budget"]["model_requests"] == 1
    events = await store.events(run["id"],"alice")
    assert sum(e.get("usage",{}).get("total_tokens",0) for e in events if e.get("usage_source") == "actual") == 100
    with pytest.raises(LeaseLost):
        await store.rag_grant(run["id"],("old",run["epoch"]),"b"*32)

async def test_concurrent_grants_and_crash_keep_unknown_consumption_on_recovery(store):
    svc = service(store)
    run = await svc.create(IDENTITY,"fixture")
    async with store.transaction() as db:
        row = await db.get(RunRecord,run["id"])
        row.snapshot = {**row.snapshot,"rag_snapshot":{"corpus_hash":"a"*64}}
    run = (await store.claim("worker"))[0]
    fence = ("worker",run["epoch"])
    grants = await asyncio.gather(*[store.rag_grant(run["id"],fence,str(i)*32,maximum_requests=32,maximum_tokens=100000) for i in (1,2)],return_exceptions=True)
    assert sum(isinstance(g,BudgetExceeded) for g in grants) == 1
    await store.finish(run["id"],fence,"failed")
    await store.resume(run["id"],"alice",{})
    restored = (await store.claim("new"))[0]
    await store.rag_recover_unknown(run["id"],("new",restored["epoch"]))
    await store.rag_recover_unknown(run["id"],("new",restored["epoch"]))
    assert (await store.get(run["id"]))["budget"]["tokens"] == 100000
    events = await store.events(run["id"],"alice")
    assert sum(e.get("usage",{}).get("total_tokens",0) for e in events if e.get("usage_source") == "estimated") == 100000
    with pytest.raises(BudgetExceeded):
        await store.reserve(run["id"],("new",restored["epoch"]),"tokens")

async def test_monitoring_counts_outcomes_and_trace_gaps_without_guessing_cost(store):
    svc = service(store)
    run = await svc.create(IDENTITY,"VPN")
    run = (await store.claim("worker"))[0]
    fence = ("worker",run["epoch"])
    await store.emit(run["id"],fence,"model",{"phase":"start","model_call_id":"unfinished"})
    await store.finish(run["id"],fence,"failed",{"error":"BudgetExceeded: fixture"})
    metrics = await store.operational_metrics()
    assert metrics["terminal_runs"] == 1 and metrics["success_rate"] == 0
    assert metrics["missing_model_end_events"] == 1 and metrics["budget_exhausted"] == 1
    assert metrics["cost"] is None and metrics["duplicate_external_effects"] is None

async def test_expired_lease_reclaimed_old_worker_cannot_publish(store):
    svc = service(store)
    await svc.create(IDENTITY, "故障")
    old = (await store.claim("old"))[0]
    async with store.transaction() as db:
        await db.execute(update(RunRecord).where(RunRecord.id == old["id"]).values(lease_until=time.time()-1))
    new = (await store.claim("new"))[0]
    assert new["epoch"] == old["epoch"]+1
    with pytest.raises(LeaseLost):
        await store.finish(old["id"], ("old", old["epoch"]), "completed")
    await store.finish(new["id"], ("new", new["epoch"]), "completed")

async def test_unknown_write_blocks_resume_and_requires_verified_reconciliation(store):
    from agent_platform.orchestrator.runtime_store import ToolCallRecord
    svc = service(store)
    await svc.create(IDENTITY, "创建工单")
    run = (await store.claim("one"))[0]
    fence = ("one", run["epoch"])
    call_id = await store.put(ToolCallRecord, run["id"], fence, {"name": "jira_create_ticket", "arguments": {"title": "VPN"}, "status": "unknown", "read_only": False})
    await store.finish(run["id"], fence, "waiting_input")
    with pytest.raises(ValueError):
        await store.resume(run["id"], "alice", {"retry": True})
    with pytest.raises(ValueError):
        await store.reconcile(run["id"], "alice", call_id, {"status": "unknown"})
    verified = {"status": "success", "data": {"key": "IT-1"}, "external_refs": [{"id": "IT-1", "verified": True}]}
    await store.reconcile(run["id"], "alice", call_id, verified)
    assert (await store.get(run["id"]))["status"] == "queued"

async def test_explicit_authorization_binds_parameters_owner_and_expiry(store):
    svc = service(store)
    run = await svc.create(IDENTITY, "创建工单", authorized_actions=[{"tool": "jira_create_ticket", "parameters": {"title": "VPN"}}])
    run = (await store.claim("one"))[0]
    fence = ("one", run["epoch"])
    status, approval_id = await store.request_approval(run["id"], fence, "jira_create_ticket", {"title": "VPN"})
    assert status == "approved"
    changed, _ = await store.request_approval(run["id"], fence, "jira_create_ticket", {"title": "Changed"})
    assert changed == "pending"
    from agent_platform.orchestrator.runtime_store import ApprovalRecord
    payload = await store.entry(ApprovalRecord, approval_id)
    with pytest.raises(KeyError):
        await store.approve(run["id"], "bob", approval_id, True, payload["parameters_hash"])
    with pytest.raises(ValueError):
        await store.approve(run["id"], "alice", approval_id, True, "wrong-hash")

async def test_checker_failure_is_unverified_not_completed(store, tmp_path):
    svc = service(store)
    async def broken(*args, **kwargs):
        raise RuntimeError("checker unavailable")
    svc.orchestrator.llm_router.chat = broken
    await svc.create(IDENTITY, "VPN 故障")
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as saver:
        worker = Worker(svc, saver)
        run = (await store.claim(worker.id))[0]
        await worker.execute(run)
    result = await store.get(run["id"])
    assert result["status"] == "failed"
    assert result["result"]["verification"]["status"] == "unverified"

async def test_tracing_export_failure_does_not_erase_local_event(store):
    class Exporter:
        def submit(self, *args):
            raise RuntimeError("offline")
    svc = service(store)
    svc.exporter = Exporter()
    await svc.create(IDENTITY, "VPN 故障")
    run = (await store.claim("one"))[0]
    await svc.emit(run, ("one", run["epoch"]), "model", {"phase": "end"})
    events = await store.events(run["id"], "alice")
    assert [event["type"] for event in events][-2:] == ["model", "tracing_warning"]

async def test_claimed_ticket_without_actual_tool_result_cannot_complete(store, tmp_path):
    class Graph:
        async def ainvoke(self, state, config=None):
            return {"final_answer":"工单已创建 IT-FAKE"}
    svc = service(store, Graph())
    await svc.create(IDENTITY,"创建VPN工单")
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "false-claim.db")) as saver:
        worker = Worker(svc,saver)
        run = (await store.claim(worker.id))[0]
        await worker.execute(run)
    result = await store.get(run["id"])
    assert result["status"] == "failed" and result["result"]["verification"]["status"] == "unverified"

async def test_explicit_resume_retries_completed_failed_graph_with_cumulative_budget(store, tmp_path):
    attempts = []
    class Graph:
        async def ainvoke(self,state,config=None):
            attempts.append(state)
            return {"final_answer":"已完成诊断"}
    svc = service(store,Graph())
    checks = []
    class Checker:
        async def check(self,*_):
            from agent_platform.orchestrator.checker import CheckResult
            checks.append(1)
            return CheckResult(status="unverified" if len(checks) == 1 else "pass", reason="fixture")
    svc.orchestrator.checker = Checker()
    await svc.create(IDENTITY,"VPN诊断")
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "retry.db")) as saver:
        worker = Worker(svc,saver)
        run = (await store.claim(worker.id))[0]
        await worker.execute(run)
        assert (await store.get(run["id"]))["status"] == "failed"
        await store.resume(run["id"],"alice",{"answer":"确认设备在线"})
        await worker.execute((await store.claim(worker.id))[0])
        result = await store.get(run["id"])
    assert result["status"] == "completed" and len(attempts) == 2, result["result"]
    assert result["budget"]["agent_calls"] == 2

async def test_complete_worker_checkpoint_artifacts_and_replay(store, tmp_path):
    svc = service(store)
    await svc.create(IDENTITY, "VPN 故障")
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as saver:
        worker = Worker(svc, saver)
        run = (await store.claim(worker.id))[0]
        await worker.execute(run)
        result = await store.get(run["id"], "alice")
        assert result["status"] == "completed", result
        assert len(result["artifacts"]) == 4
        events = await store.events(run["id"], "alice")
        assert [e["seq"] for e in events] == list(range(1, len(events)+1))
        cursor = events[2]["seq"]
        assert all(e["seq"] > cursor for e in await store.events(run["id"], "alice", cursor))
        with pytest.raises(KeyError):
            await store.read_artifact(run["id"], "bob", result["artifacts"][0]["id"])
        checkpoint = await saver.aget_tuple({"configurable": {"thread_id": run["id"]}})
        assert checkpoint is not None

async def test_approval_resume_duplicate_click_executes_write_once(store, tmp_path):
    effects = []
    class Write(BaseTool):
        name = "jira_create_ticket"
        read_only = False
        idempotent = False
        required_scopes = ("jira:create",)
        args_schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
        async def execute(self, title):
            effects.append(title)
            return ToolResult(status="success", data={"key": "IT-1"}, external_refs=[{"id": "IT-1", "verified": True}])
    gateway = ToolGateway(tools=[Write()])
    async def act(state):
        result = await gateway.execute("jira_create_ticket", {"title": "VPN"})
        return {"final_answer": "IT-1" if result["ok"] else "未建单"}
    from typing import TypedDict
    class State(TypedDict, total=False):
        messages: list
        final_answer: str
    builder = StateGraph(State)
    builder.add_node("act", act)
    builder.add_edge(START, "act")
    builder.add_edge("act", END)
    graph = builder.compile(checkpointer=True)
    svc = service(store, graph)
    await svc.create(IDENTITY, "创建VPN工单")
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as saver:
        worker = Worker(svc, saver)
        run = (await store.claim(worker.id))[0]
        await worker.execute(run)
        pending = await store.get(run["id"], "alice")
        assert pending["status"] == "waiting_approval", pending
        assert not effects
        approval = pending["approvals"][0]
        await store.approve(run["id"], "alice", approval["id"], True, approval["parameters_hash"])
        await store.approve(run["id"], "alice", approval["id"], True, approval["parameters_hash"])
        await worker.execute((await store.claim(worker.id))[0])
        result = await store.get(run["id"])
        assert result["status"] == "completed", result
        assert effects == ["VPN"]

async def test_parallel_approval_cards_wait_until_all_decisions_then_write_once(store, tmp_path):
    from typing import TypedDict, Annotated
    from langchain_core.messages import AIMessage
    from langchain_core.tools import StructuredTool
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    effects = []
    class Write(BaseTool):
        name = "jira_create_ticket"
        read_only = idempotent = False
        required_scopes = ("jira:create",)
        args_schema = {"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}
        async def execute(self, title):
            effects.append(title)
            return ToolResult(status="success", data={"key": title}, external_refs=[{"id": title,"verified":True}])
    gateway = ToolGateway(tools=[Write()])
    async def create(title: str):
        return str(await gateway.execute("jira_create_ticket", {"title": title}))
    class State(TypedDict, total=False):
        messages: Annotated[list, add_messages]
        final_answer: str
    builder = StateGraph(State)
    builder.add_node("calls", lambda state: {"messages":[AIMessage(content="",tool_calls=[{"name":"create","args":{"title":name},"id":name} for name in ("IT-1","IT-2")])]})
    builder.add_node("tools", ToolNode([StructuredTool.from_function(coroutine=create,name="create",description="fixture")]))
    builder.add_node("answer", lambda state: {"final_answer":"已核验 IT-1 与 IT-2"})
    builder.add_edge(START,"calls")
    builder.add_edge("calls","tools")
    builder.add_edge("tools","answer")
    builder.add_edge("answer",END)
    svc = service(store, builder.compile(checkpointer=True))
    await svc.create(IDENTITY, "创建两个测试工单")
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "parallel.db")) as saver:
        worker = Worker(svc,saver)
        run = (await store.claim(worker.id))[0]
        await worker.execute(run)
        pending = await store.get(run["id"])
        assert pending["status"] == "waiting_approval" and len(pending["approvals"]) == 2
        first, second = pending["approvals"]
        await store.approve(run["id"],"alice",first["id"],True,first["parameters_hash"])
        assert (await store.get(run["id"]))["status"] == "waiting_approval" and not effects
        await store.approve(run["id"],"alice",second["id"],True,second["parameters_hash"])
        await worker.execute((await store.claim(worker.id))[0])
        current = await store.get(run["id"])
        assert current["status"] == "completed", current
        assert sorted(effects) == ["IT-1","IT-2"]

def test_skill_path_guard_versions_and_context_preserves_required():
    skills = SkillRegistry(ROOT / "skills")
    assert len(skills.skills) == 2
    skill = skills.skills["emc-standard-compare"]
    with pytest.raises(ValueError):
        skill.resource("../../.env")
    with pytest.raises(ValueError):
        skill.resource("/etc/passwd")
    bundle = ContextBuilder(input_limit=900).build(goal="比较 EMC 版本", authorization={"scopes": ["read"]}, step="比较",
        evidence=[{"value": 30, "unit": "dBµV/m", "doc": "A:2026", "page": 1}], history=[{"role": "user", "content": "旧历史" * 1000}])
    assert bundle.dropped
    text = str(bundle.messages)
    assert "dBµV/m" in text and "A:2026" in text and "scopes" in text
    with pytest.raises(ContextOverflow):
        ContextBuilder(input_limit=10).build(goal="x"*1000, authorization={}, step="x")

def test_old_tool_facts_preserve_numbers_units_versions_and_controlled_source():
    from agent_platform.orchestrator.context import evidence_facts
    projection = evidence_facts({"version":"2026", "condition":{"distance":3,"unit":"m"},
        "comparison_matrix":[{"value":30.25,"unit":"dBµV/m","source":"A:2026"}],
        "text":"长证据段落"*200+" 32 dBµV/m"},"artifact-a")
    assert projection["comparison_matrix"][0]["value"] == 30.25
    assert projection["condition"]["unit"] == "m" and projection["version"] == "2026"
    assert "32 dBµV/m" in projection["text"]["numbers_and_units"]
    assert projection["text"]["artifact_id"] == "artifact-a" and projection["text"]["requires_source_read"]
