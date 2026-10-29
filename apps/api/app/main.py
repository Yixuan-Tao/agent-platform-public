"""agent-platform API 网关入口。"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from agent_platform.core.config import get_settings
from agent_platform.core.db import create_all, init_db, verify_schema_compatibility, MigrationRequired, close_db
from agent_platform.core.llm.gateway import validate_gateway
from agent_platform.core.llm.router import get_llm_router
from agent_platform.core.logging import setup_logging
from agent_platform.core.security import AccessStore, AuditStore

from app.middleware import APIKeyMiddleware, AccessLogMiddleware
from app.routes import agents, auth, chat, documents, feedback, health, sessions, runs, memories


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    setup_logging(settings.log_level, settings.log_dir)
    init_db(settings.database_url)
    try:
        await verify_schema_compatibility()
        await create_all()
    except MigrationRequired:
        await close_db()
        raise
    except Exception as exc:  # noqa: BLE001
        if settings.runs_enabled:
            await close_db()
            raise RuntimeError("持久化任务要求数据库可用，请完成迁移和连接检查") from exc
        logger.warning("建表失败（可能数据库未启动），将降级运行: {}", exc)

    # 组装服务（包接口见各包 README；agents 包由子 Agent 实现 register_all）
    from agent_platform.tools.base import ToolRegistry
    from agent_platform.tools.builtin.escalate_human import EscalateHumanTool
    from agent_platform.tools.builtin.jira_tool import JiraCreateTicketTool, JiraGetTicketTool
    from agent_platform.tools.builtin.rag_search import RagSearchTool
    from agent_platform.tools.builtin.web_collect import RssFetchTool, WebCollectTool

    tool_registry = ToolRegistry()
    for tool in (
        RagSearchTool(),
        JiraCreateTicketTool(),
        JiraGetTicketTool(),
        WebCollectTool(),
        RssFetchTool(),
        EscalateHumanTool(),
    ):
        tool_registry.register(tool)

    # 声明式动态注册（TOOL_CONFIG JSON）
    if settings.tool_config_json:
        try:
            tool_config = json.loads(settings.tool_config_json)
            if isinstance(tool_config, list):
                tool_registry.register_from_config(tool_config)
        except json.JSONDecodeError as exc:
            logger.warning("TOOL_CONFIG 不是合法 JSON，忽略: {}", exc)

    # MCP 客户端（MCP_SERVERS JSON）：外部工具热插拔
    from agent_platform.tools.mcp.client import parse_mcp_servers, register_mcp_servers

    mcp_clients = []
    mcp_servers = parse_mcp_servers(settings.mcp_servers_json)
    if mcp_servers:
        sdk_servers = [s for s in mcp_servers if s.get("adapter", "sdk") == "sdk"]
        legacy_servers = [s for s in mcp_servers if s.get("adapter") == "legacy"]
        if sdk_servers:
            from agent_platform.tools.mcp.sdk_client import register_sdk_servers
            mcp_clients = await register_sdk_servers(tool_registry, sdk_servers)
        if legacy_servers:
            mcp_clients += await register_mcp_servers(tool_registry, legacy_servers)

    # 插件沙箱（plugin_dir）：隔离执行用户插件
    from agent_platform.tools.plugin_tool import scan_plugins

    for plugin in scan_plugins(settings.plugin_dir):
        try:
            tool_registry.register(plugin)
        except ValueError as exc:
            logger.warning("插件注册冲突，跳过: {}", exc)

    from agent_platform.memory.service import build_memory_service
    from agent_platform.orchestrator.registry import get_agent_registry
    from agent_platform.orchestrator.service import OrchestratorService

    registry = get_agent_registry()
    try:
        from agent_platform.agents import register_all

        register_all(registry)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Agent 注册失败（agents 包可能未就绪）: {}", exc)

    memory = build_memory_service(settings)
    llm_router = get_llm_router()

    # ---- 认证 / 审计 / 模型网关自检 ----
    # 相对路径按 apps/api 目录解析，绝对路径（容器内 RUNTIME_DIR）直接使用
    runtime_dir = Path(settings.runtime_dir)
    if not runtime_dir.is_absolute():
        runtime_dir = Path(__file__).resolve().parents[2] / runtime_dir
    runtime_dir.mkdir(parents=True, exist_ok=True)
    app.state.access_store = AccessStore(runtime_dir / "access.db")
    app.state.audit_store = AuditStore(runtime_dir / "audit.db")
    if settings.initial_admin_username and settings.initial_admin_password:
        try:
            app.state.access_store.create_user(
                settings.initial_admin_username, settings.initial_admin_password, "admin"
            )
        except Exception:  # noqa: BLE001
            pass
    gateway_check = validate_gateway(settings)
    app.state.gateway_status = {"ok": gateway_check.ok, "reason": gateway_check.reason}
    if not gateway_check.ok:
        logger.warning("模型网关自检未通过（服务降级，不影响启动）: {}", gateway_check.reason)

    app.state.tool_registry = tool_registry
    app.state.agent_registry = registry
    app.state.memory = memory
    app.state.orchestrator = OrchestratorService(
        registry=registry,
        llm_router=llm_router,
        tool_registry=tool_registry,
        memory_service=memory,
        use_router_react=settings.use_router_react,
        max_agent_calls=settings.max_agent_calls,
        plan_mode=settings.plan_mode,
        max_plan_steps=settings.max_plan_steps,
        answer_streaming=settings.answer_streaming,
        agent_call_budget=settings.agent_call_budget,
    )
    logger.info("{} 启动完成，可用 Agent: {}", settings.app_name, registry.names())
    runtime_stack = __import__("contextlib").AsyncExitStack()
    worker_task = None
    if settings.runs_enabled:
        from agent_platform.core.db import session_factory
        from agent_platform.orchestrator.runtime_store import RunStore
        from agent_platform.orchestrator.runtime import RunService, Worker
        from agent_platform.skills.registry import SkillRegistry
        skills = SkillRegistry(Path(__file__).resolve().parents[3] / "skills")
        if settings.database_url.startswith("sqlite"):
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            db_path = settings.database_url.split("///", 1)[1]
            saver = await runtime_stack.enter_async_context(AsyncSqliteSaver.from_conn_string(db_path + ".checkpoints"))
        else:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            saver = await runtime_stack.enter_async_context(AsyncPostgresSaver.from_conn_string(settings.database_url.replace("postgresql+asyncpg://", "postgresql://")))
        await saver.setup()
        async def reload_identity(saved):
            if saved.get("auth") == "session":
                user = next((u for u in app.state.access_store.list_users() if u["username"] == saved["username"]), None)
                if user is None:
                    return None
            elif saved.get("auth") == "api_key" and not settings.api_key:
                return None
            elif saved.get("auth") == "admin_api_key" and not settings.admin_api_key:
                return None
            else:
                user = saved
            return {**user, "scopes": settings.service_tool_scopes if user["role"] == "service" else settings.user_tool_scopes}
        exporter = None
        if settings.langfuse_enabled:
            from agent_platform.core.llm.exporter import LangfuseExporter
            exporter = LangfuseExporter(settings)
            exporter.start()
        capabilities, python_tool = [], None
        if settings.semantic_memory_enabled:
            capabilities.append("semantic_memory")
        if settings.tool_discovery_enabled:
            capabilities.append("tool_discovery")
        if settings.sandbox_enabled:
            from agent_platform.tools.python_container import ContainerPythonTool, accept_environment
            image_id = await accept_environment()
            python_tool = ContainerPythonTool(runtime_dir / "python_tasks", image=image_id)
            capabilities.append("python_container")
        service = RunService(RunStore(session_factory()), app.state.orchestrator, skills, llm_router.resolve_model(), reload_identity, exporter,
                             capabilities=capabilities, python_tool=python_tool)
        app.state.run_service = service
        worker = Worker(service, saver)
        app.state.run_worker = worker
        import asyncio
        worker_task = asyncio.create_task(worker.serve())
        if settings.experiments_enabled:
            from agent_platform.evaluation.runner import ExperimentRunner
            app.state.experiment_runner = ExperimentRunner(service)
    yield
    # Task/checkpoint workers close before disposing their ledger engine below.
    if worker_task:
        if getattr(app.state, "experiment_runner", None):
            await app.state.experiment_runner.close()
        await app.state.run_worker.close()
        worker_task.cancel()
        await asyncio.gather(worker_task, return_exceptions=True)
        if app.state.run_service.exporter:
            await app.state.run_service.exporter.close()
    await runtime_stack.aclose()
    for client in reversed(mcp_clients):
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass
    await close_db()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="agent-platform API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(AccessLogMiddleware)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(sessions.router)
    app.include_router(feedback.router)
    app.include_router(agents.router)
    app.include_router(documents.router)
    app.include_router(runs.router)
    app.include_router(memories.router)

    @app.get("/", include_in_schema=False)
    async def service_index() -> dict:
        """服务索引（免鉴权）。

        直接访问 API 端口时给出人类可读的说明，避免看到 401 JSON 而困惑：
        网关是接口服务，交互界面在前端地址。
        """
        settings = get_settings()
        return {
            "service": settings.app_name,
            "kind": "api-gateway",
            "version": app.version,
            "health": "/health",
            "docs": "/docs",
            "frontend": settings.frontend_url,
            "endpoints": {
                "login": "POST /api/v1/auth/login",
                "chat": "POST /api/v1/chat (SSE)",
                "sessions": "GET /api/v1/sessions",
                "documents": "GET /api/v1/documents",
            },
            "auth": "业务接口需凭据：Authorization: Bearer <token> 或 X-API-Key",
        }

    return app


app = create_app()
