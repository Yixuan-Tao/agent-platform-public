# Test values are synthetic fixtures for software behavior.
"""No paid APIs: authenticated run ownership, SSE replay and optional feature flags."""
from pathlib import Path
import pytest
import httpx
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from agent_platform.core.db import Base
from agent_platform.core.security import require_login, require_admin
from agent_platform.core.config import Settings
from agent_platform.orchestrator.runtime_store import RunStore
from agent_platform.orchestrator.runtime import RunService
from agent_platform.orchestrator.registry import AgentRegistry
from agent_platform.orchestrator.service import OrchestratorService
from agent_platform.skills.registry import SkillRegistry
from agent_platform.tools.base import ToolRegistry
from app.routes.runs import router

@pytest.fixture
async def api(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    class LLM:
        settings = None
    app = FastAPI()
    app.state.settings = Settings(_env_file=None)
    app.state.run_service = RunService(RunStore(async_sessionmaker(engine, expire_on_commit=False)),
        OrchestratorService(AgentRegistry(), LLM(), ToolRegistry()),
        SkillRegistry(Path(__file__).resolve().parents[3] / "skills"), "fixture-model")
    identity = {"username": "alice", "role": "viewer"}
    app.dependency_overrides[require_login] = lambda: identity
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client, app, identity
    await engine.dispose()

async def test_run_api_owner_cancel_cursor_and_features(api):
    client, app, identity = api
    created = await client.post("/api/v1/runs", json={"goal": "VPN 故障"})
    assert created.status_code == 202
    run_id = created.json()["id"]
    assert (await client.get("/api/v1/runs")).json()[0]["id"] == run_id
    identity["username"] = "bob"
    assert (await client.get(f"/api/v1/runs/{run_id}")).status_code == 404
    assert (await client.post(f"/api/v1/runs/{run_id}/cancel")).status_code == 404
    identity["username"] = "alice"
    assert (await client.post(f"/api/v1/runs/{run_id}/cancel")).json()["status"] == "cancelled"
    events = await client.get(f"/api/v1/runs/{run_id}/events?after=1")
    assert '"seq": 2' in events.text and '"seq": 1' not in events.text
    assert [p["name"] for p in (await client.get("/api/v1/experiment-profiles")).json()] == ["B0", "B1", "B2", "C1", "C2", "P1", "M1"]
    app.state.run_service = None
    assert (await client.get("/api/v1/runs")).status_code == 503
