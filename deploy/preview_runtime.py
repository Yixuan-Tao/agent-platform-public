"""Loopback-only browser acceptance fixture: real ledger/worker, synthetic tool and model."""
from pathlib import Path
import asyncio
import sys
from contextlib import asynccontextmanager

ROOT = Path(__file__).resolve().parents[1]
for package in ("core", "tools", "memory", "orchestrator", "agents", "skills", "evaluation"):
    sys.path.insert(0, str(ROOT / "packages" / package / "src"))
sys.path.insert(0, str(ROOT / "apps" / "api"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent_platform.core.config import Settings
from agent_platform.core.db import Base
from agent_platform.core.models import AgentManifest
from agent_platform.core.security import require_login, require_admin
from agent_platform.orchestrator.registry import AgentRegistry
from agent_platform.orchestrator.runtime import RunService, Worker
from agent_platform.orchestrator.runtime_store import RunStore
from agent_platform.orchestrator.service import OrchestratorService
from agent_platform.skills.registry import SkillRegistry
from agent_platform.tools.base import BaseTool, ToolRegistry
from agent_platform.tools.gateway import ToolGateway
from agent_platform.tools.result import ToolResult
from app.routes.runs import router


class SyntheticTicket(BaseTool):
    name = "jira_create_ticket"
    description = "浏览器验收合成工单，绝不连接外部系统"
    read_only = idempotent = False
    required_scopes = ("jira:create",)
    args_schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    async def execute(self, title):
        return ToolResult(status="success", data={"key": "PREVIEW-1", "title": title},
                          external_refs=[{"id": "PREVIEW-1", "system": "synthetic", "verified": True}])


class Model:
    settings = None
    def resolve_model(self, agent=None):
        return "synthetic-no-paid-model"
    async def chat(self, *args, **kwargs):
        return "PASS"


def build_graph(*_):
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    class State(TypedDict, total=False):
        messages: list
        final_answer: str
    async def execute(state):
        gateway = ToolGateway(tools=[SyntheticTicket()])
        result = await gateway.execute("jira_create_ticket", {"title": "浏览器验收合成 VPN 工单"})
        return {"final_answer": "合成工单 PREVIEW-1 已创建并核验；未调用真实模型或外部工单系统。" if result["ok"] else "审批未通过，未创建工单。"}
    builder = StateGraph(State)
    builder.add_node("execute", execute)
    builder.add_edge(START, "execute")
    builder.add_edge("execute", END)
    return builder.compile(checkpointer=True)


@asynccontextmanager
async def lifespan(app):
    directory = ROOT / "runtime_data" / "browser_acceptance_20261004"
    directory.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine("sqlite+aiosqlite:///" + (directory / "ledger.db").as_posix())
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    registry, tools = AgentRegistry(), ToolRegistry()
    registry.register(AgentManifest(name="it_support", description="合成 IT"), build_graph)
    tools.register(SyntheticTicket())
    app.state.settings = Settings(_env_file=None, runs_enabled=True)
    app.state.run_service = RunService(RunStore(async_sessionmaker(engine, expire_on_commit=False)),
        OrchestratorService(registry, Model(), tools), SkillRegistry(ROOT / "skills"), "synthetic-no-paid-model")
    async with AsyncSqliteSaver.from_conn_string(str(directory / "checkpoints.db")) as saver:
        await saver.setup()
        worker = Worker(app.state.run_service, saver)
        task = asyncio.create_task(worker.serve())
        try:
            yield
        finally:
            await worker.close()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    await engine.dispose()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3301"], allow_methods=["*"], allow_headers=["*"])
app.dependency_overrides[require_login] = lambda: {"username": "browser-fixture", "role": "user"}
app.dependency_overrides[require_admin] = lambda: {"username": "browser-fixture", "role": "admin"}
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=18800)
