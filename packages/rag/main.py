"""
多模态 RAG 系统 — FastAPI 入口

启动流程：
    1) lifespan 中预热 ChromaDB 连接，并把 pipeline 注入 app.state
    2) 注册 API 路由
    3) ``/health`` 改为读 app.state，避免冷启动阻塞

安全能力（对齐 intern_rag 新版）：
    - 局域网模式（LAN_ENABLED=true）强制 HTTPS 并启用 Bearer 会话鉴权
    - 人工证据版本库（/evidence/*）与本地账号/审计存储
    - 可信模型网关校验（MODEL_GATEWAY_ALLOWED_HOSTS 非空时启用）
"""
from contextlib import asynccontextmanager
import hmac
import os

from pathlib import Path
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.compare_routes import router as compare_router
from api.evidence_routes import router as evidence_router
from api.auth_routes import require_viewer, router as auth_router
from api.routes import _ingest_executor, router as api_router
from api.runtime_contract import router as runtime_router, RuntimeContractMiddleware
from config import config
from db.chroma_client import ChromaClientSingleton
from core.model_gateway import ModelGateway, GatewayConfigurationError
from pipeline.ingest import IngestPipeline
from pipeline.query import QueryPipeline
from rag.service import RagService
from utils.logger import log
from utils.local_runtime import LocalTokenStore, TaskStore
from utils.access_control import AccessStore
from utils.audit import AuditStore
from utils.tls import certificate_fingerprint, tls_options
from utils.evidence_store import EvidenceStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化 pipeline + 预热 ChromaDB。"""
    log.info("=" * 50)
    log.info("多模态 RAG 系统启动中...")
    log.info("OneAPI 网关: %s", config.ONEAPI_BASE_URL)
    log.info("ChromaDB 数据库: %s", config.CHROMA_DB_PATH)
    log.info("=" * 50)
    # 可信模型网关校验（白名单为空时跳过，开发模式）
    try:
        app.state.model_gateway = ModelGateway(
            config.ONEAPI_BASE_URL, config.ONEAPI_API_KEY,
            config.MODEL_GATEWAY_ALLOWED_HOSTS,
        ) if config.MODEL_GATEWAY_ALLOWED_HOSTS else None
    except GatewayConfigurationError as exc:
        log.warning("模型网关校验未通过（跳过，使用默认客户端）: %s", exc)
        app.state.model_gateway = None
    if config.LAN_ENABLED:
        log.info("局域网 HTTPS 已启用；证书 SHA-256 指纹: %s", certificate_fingerprint(Path(config.TLS_CERT_FILE)))
        log.info("请仅在核验上述指纹后，让团队设备信任该自签名证书。")

    # 1) 预热 ChromaDB：获取 collection
    chroma_ready = True
    try:
        ChromaClientSingleton.get_instance()
        log.info("ChromaDB 连接就绪")
    except Exception as e:  # noqa: BLE001
        log.warning("ChromaDB 预热失败（启动后端点仍可用）: %s", e)
        chroma_ready = False

    # 2) 实例化 pipeline 并注入 app.state（供 Depends 使用）
    app.state.chroma_ready = chroma_ready
    app.state.ingest_pipeline = IngestPipeline()
    app.state.query_pipeline = QueryPipeline()
    # RagService: 工具层消费 RAG 数据的统一接口，不依赖 QueryPipeline
    try:
        chroma_client = ChromaClientSingleton.get_instance()
        app.state.rag_service = RagService(
            chroma=chroma_client,
            embedder=app.state.query_pipeline.embedder,
            retriever=app.state.query_pipeline.retriever,
        )
    except Exception as e:
        log.warning("RagService 初始化失败（后续端点可能不可用）: %s", e)
        app.state.rag_service = None
    runtime_dir = Path(config.RUNTIME_DIR)
    # 显式配置 LOCAL_API_TOKEN 时使用固定值，便于网关联调（本机模式写操作鉴权）；
    # 未配置则沿用随机生成本机令牌的既有行为。
    app.state.local_api_token = (
        os.getenv("LOCAL_API_TOKEN", "").strip()
        or LocalTokenStore(runtime_dir / "local-token").get_or_create()
    )
    app.state.task_store = TaskStore(runtime_dir / "runtime.db")
    app.state.access_store = AccessStore(runtime_dir / "access.db")
    app.state.audit_store = AuditStore(runtime_dir / "audit.db")
    app.state.evidence_store = EvidenceStore(runtime_dir / "evidence.db")
    if config.INITIAL_ADMIN_USERNAME and config.INITIAL_ADMIN_PASSWORD:
        try:
            app.state.access_store.create_user(
                config.INITIAL_ADMIN_USERNAME, config.INITIAL_ADMIN_PASSWORD, "admin"
            )
        except Exception:
            pass
    # 启动时清理一次过期任务，防止 runtime.db 无限增长
    interrupted = app.state.task_store.recover_interrupted()
    if interrupted:
        log.warning("已标记 %d 个重启前中断的摄入任务", interrupted)
    app.state.task_store.prune()

    log.info("Pipeline 已就绪 (chroma_ready=%s)", chroma_ready)
    yield

    # 应用关闭时清理资源
    _ingest_executor.shutdown(wait=True)
    log.info("多模态 RAG 系统已关闭")


app = FastAPI(
    title="多模态 RAG 系统",
    description="面向硬件行业文档的多模态检索增强生成系统，支持 PDF/DOCX/图片，中英德三语",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 跨域：仅放行本机常用源，避免完全开放
# - "null" 兼容 file:// 协议打开的 console.html
# - http://localhost / http://127.0.0.1 覆盖本机浏览器开发场景
app.add_middleware(
    CORSMiddleware,
    # LAN 控制台与 API 同源，无需 CORS；仅本机模式兼容本地文件控制台。
    allow_origins=[] if config.LAN_ENABLED else ["null", "http://localhost", "http://127.0.0.1"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.add_middleware(RuntimeContractMiddleware)


@app.middleware("http")
async def audit_lan_request(request: Request, call_next):
    response = await call_next(request)
    if config.LAN_ENABLED and not request.url.path.startswith("/auth/"):
        try:
            username = "anonymous"
            authorization = request.headers.get("authorization", "")
            if authorization.startswith("Bearer "):
                try:
                    token = authorization[7:]
                    username = (
                        "automation" if config.ADMIN_API_KEY and hmac.compare_digest(token, config.ADMIN_API_KEY)
                        else app.state.access_store.identity(token)["username"]
                    )
                except Exception:
                    pass
            app.state.audit_store.record(
                username=username, source_ip=request.client.host if request.client else "",
                action=f"{request.method} {request.url.path}", status=str(response.status_code),
                run_id=response.headers.get("X-Run-Id", ""),
            )
        except Exception as exc:
            log.warning("审计写入失败: %s", exc)
    return response

# 挂载业务路由
app.include_router(api_router, prefix="", dependencies=[Depends(require_viewer)])
app.include_router(runtime_router, dependencies=[Depends(require_viewer)])
app.include_router(auth_router)
# 挂载对比/可视化路由
app.include_router(compare_router, prefix="", dependencies=[Depends(require_viewer)])
app.include_router(evidence_router)

# 挂载 MCP 服务路由
from mcp.server import router as mcp_router  # noqa: E402
app.include_router(mcp_router, dependencies=[Depends(require_viewer)])

# 根路径直接返回控制台 HTML
@app.get("/")
async def serve_console(request: Request):
    html = Path("console.html").read_text(encoding="utf-8")
    # 局域网模式下，写权限由管理员 Bearer 会话控制，不能把本机令牌交给浏览器。
    local_token = "" if config.LAN_ENABLED else request.app.state.local_api_token
    html = html.replace("__LOCAL_API_TOKEN__", local_token)
    return HTMLResponse(html.replace("__LAN_ENABLED__", "true" if config.LAN_ENABLED else "false"))


if __name__ == "__main__":
    import uvicorn
    # 即使误用开发入口，局域网模式也不能回退到明文 HTTP。
    uvicorn.run(
        "main:app", host=config.HOST, port=config.PORT,
        reload=not config.LAN_ENABLED, **tls_options(),
    )
