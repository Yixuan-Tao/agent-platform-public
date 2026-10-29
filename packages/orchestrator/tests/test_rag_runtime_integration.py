# Test values are synthetic fixtures for software behavior.
"""Real loopback HTTP -> durable SQL grants -> Worker, without paid providers."""
import asyncio
import os
from pathlib import Path
import socket
import subprocess
import sys

import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent_platform.core.config import Settings
from agent_platform.core.db import Base
from agent_platform.core.models import AgentManifest
from agent_platform.orchestrator.registry import AgentRegistry
from agent_platform.orchestrator.runtime import RunService, Worker
from agent_platform.orchestrator.runtime_store import RunStore
from agent_platform.orchestrator.service import OrchestratorService
from agent_platform.skills.registry import SkillRegistry
from agent_platform.tools.base import ToolRegistry
from agent_platform.tools.builtin.rag_search import RagSearchTool
from agent_platform.tools.gateway import ToolGateway

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
async def rag_server(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        port = sock.getsockname()[1]
    script = ROOT / "packages/tools/tests/fixtures/rag_contract_server.py"
    environment = {**os.environ,"RUNTIME_DIR":str(tmp_path / "rag"),"LAN_ENABLED":"false","PYTHONPATH":str(ROOT / "packages/rag"),
                   "LLM_API_KEY":"","ONEAPI_API_KEY":"","PYTHONIOENCODING":"utf-8"}
    process = await asyncio.create_subprocess_exec(sys.executable,str(script),str(port),cwd=str(ROOT / "packages/rag"),env=environment,
                   stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.PIPE,
                   **({"creationflags":subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
    url = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient(timeout=.5) as client:
            for _ in range(80):
                if process.returncode is not None:
                    raise AssertionError((await process.stderr.read()).decode("utf-8",errors="replace"))
                try:
                    if (await client.get(url+"/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.1)
            else:
                raise AssertionError("fixture RAG server did not become ready")
        yield url
    finally:
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.communicate(),5)
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()


@pytest.mark.parametrize("goal,expected_status",[("retrieve-proof","completed"),("change-index","failed")])
async def test_worker_with_real_rag_rpc_accounts_usage_and_stops_changed_index(tmp_path,rag_server,goal,expected_status):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        store = RunStore(async_sessionmaker(engine,expire_on_commit=False))
        settings = Settings(_env_file=None,rag_service_url=rag_server,rag_runtime_contract_enabled=True,rag_request_timeout=5)
        class LLM:
            async def chat(self,*args,**kwargs):
                return "PASS"
            def resolve_model(self,*args):
                return "fixture"
        llm = LLM()
        llm.settings = settings
        registry, tools = AgentRegistry(), ToolRegistry()
        tool = RagSearchTool(rag_server)
        tool.settings = settings
        tools.register(tool)
        class Graph:
            async def ainvoke(self,state,config=None):
                result = await ToolGateway(tools,max_retries=0).execute("rag_search",{"query":goal})
                return {"final_answer":"证据 30 dBµV/m" if result["ok"] else "无法取得证据"}
        registry.register(AgentManifest(name="emc_compare",description="fixture"),lambda *_:Graph())
        service = RunService(store,OrchestratorService(registry,llm,tools),SkillRegistry(ROOT / "skills"),"fixture")
        run = await service.create({"username":"alice","role":"user","scopes":[]},goal,completion={"agent_hint":"emc_compare"})
        claimed = (await store.claim("worker"))[0]
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.db")) as saver:
            worker = Worker(service,saver)
            worker.id = "worker"
            await worker.execute(claimed)
        current = await store.get(run["id"],"alice")
        assert current["status"] == expected_status, current["result"]
        assert "model_requests" in current["budget"], current["result"]
        assert current["budget"]["model_requests"] == 2 and current["budget"]["tokens"] == 28
        events = await store.events(run["id"],"alice")
        assert sum(e.get("usage",{}).get("total_tokens",0) for e in events if e.get("usage_source") == "actual") == 28
        if goal == "change-index":
            assert "语料" in current["result"]["error"]
            assert not any(e["type"] == "answer_delta" for e in events)
        else:
            assert current["result"]["verification"]["status"] == "pass" and len(current["artifacts"]) >= 4
    finally:
        await engine.dispose()
