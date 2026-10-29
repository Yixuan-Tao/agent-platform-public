"""
多模态 RAG 系统 — REST API 路由 (Task 7)

提供端点：
    - GET  /health                健康检查
    - POST /v1/chat/completions   OpenAI 兼容的 chat completions（含 tool_calls 入口）
    - POST /tools/search          简化版检索工具
    - POST /ingest                文件上传摄入（multipart/form-data，异步模式）
    - GET  /ingest/status/{id}    查询异步摄入任务状态
    - GET  /documents             列出已索引文档
    - DELETE /documents/{name}    按文件名删除

设计要点：
    - Pipeline 实例通过 ``Depends`` 注入（lifespan 中放入 app.state）
    - 所有异常用 ``HTTPException(500)`` 包裹
    - tool_calls 路径：仅暴露"接收 tool_call 请求→返回检索结果"，
      不实现完整 multi-turn 自动循环
    - /ingest 采用异步模式：立即返回 task_id，后台线程处理耗时操作
"""
from __future__ import annotations

import json
import secrets
import mimetypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile, Header,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    DeleteResponse,
    DocumentListItem,
    DocumentListResponse,
    HealthResponse,
    IngestAcceptedResponse,
    IngestResponse,
    IngestStatusResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    Usage,
)
from api.tool_definitions import handle_tool_call
from api.auth_routes import require_admin
from config import config
from db.chroma_client import ChromaClientSingleton
from pipeline.ingest import IngestPipeline
from pipeline.query import QueryPipeline
from utils.logger import log
from utils.local_runtime import TaskStore

# 临时上传目录（存放 multipart 文件）
UPLOAD_DIR: Path = Path(__file__).resolve().parent.parent / "logs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 后台线程池（处理耗时摄入操作，避免阻塞 HTTP 连接）
_ingest_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")

# ============================================================
# Router
# ============================================================
router = APIRouter()

def require_local_token(request: Request, x_local_token: str = Header(default=""), authorization: str = Header(default="")) -> None:
    if config.LAN_ENABLED:
        require_admin(request, authorization)
        return
    token = getattr(request.app.state, "local_api_token", "")
    if not token or not secrets.compare_digest(x_local_token, token):
        raise HTTPException(status_code=401, detail="缺少或无效的本机写操作令牌")

# ============================================================
# 依赖注入
# ============================================================
def get_ingest_pipeline(request: Request) -> IngestPipeline:
    """从 app.state 取 IngestPipeline；未注入时兜底即时构造。"""
    pipeline = getattr(request.app.state, "ingest_pipeline", None)
    if pipeline is None:
        pipeline = IngestPipeline()
        request.app.state.ingest_pipeline = pipeline
    return pipeline


def get_query_pipeline(request: Request) -> QueryPipeline:
    """从 app.state 取 QueryPipeline。"""
    pipeline = getattr(request.app.state, "query_pipeline", None)
    if pipeline is None:
        pipeline = QueryPipeline()
        request.app.state.query_pipeline = pipeline
    return pipeline


def get_chroma() -> Optional[ChromaClientSingleton]:
    """直接取 ChromaDB 单例（健康检查/统计用）；连接失败时返回 None。"""
    try:
        return ChromaClientSingleton.get_instance()
    except Exception as e:  # noqa: BLE001
        log.warning("ChromaDB 连接失败，依赖此实例的端点将降级: %s", e)
        return None


# ============================================================
# /health
# ============================================================
@router.get("/health", response_model=HealthResponse)
async def health(chroma: Optional[ChromaClientSingleton] = Depends(get_chroma)) -> dict:
    """健康检查：服务状态 + ChromaDB 连接 + 文档数。"""
    try:
        chroma_connected = chroma is not None
        chunk_count = 0
        document_count = 0
        if chroma_connected:
            try:
                stats = chroma.get_collection_stats()
                chunk_count = int(stats.get("row_count", 0))
                document_count = len(chroma.list_documents())
            except Exception as e:  # noqa: BLE001
                log.warning("统计信息获取失败: %s", e)
                chroma_connected = False

        return {
            "status": "ok" if chroma_connected else "degraded",
            "version": "0.1.0",
            "chroma_connected": chroma_connected,
            "document_count": document_count,
            "chunk_count": chunk_count,
        }
    except Exception as e:  # noqa: BLE001
        log.exception("/health 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================
# /images/{chunk_id}  — 图片服务
# ============================================================
@router.get("/images/{chunk_id}")
async def get_image(chunk_id: str, chroma: Optional[ChromaClientSingleton] = Depends(get_chroma)) -> Any:
    """按 chunk_id 返回原始图片 bytes。

    图片数据存在磁盘 chroma_data/images/{chunk_id}.{ext}，metadata 中存 image_path。
    """
    if chroma is None:
        raise HTTPException(status_code=503, detail="ChromaDB 不可用")
    entity = chroma.get_by_id(chunk_id)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"chunk 不存在: {chunk_id}")
    meta = entity.get("metadata", {}) or {}
    img_path = meta.get("image_path")
    if not img_path or not Path(img_path).is_file():
        raise HTTPException(status_code=404, detail="该 chunk 不包含图片数据")
    content_type, _ = mimetypes.guess_type(img_path)
    return Response(content=Path(img_path).read_bytes(), media_type=content_type or "image/png")


# ============================================================
# /v1/chat/completions
# ============================================================

def _make_response_id() -> str:
    return f"chatcmpl-{uuid4().hex[:24]}"


def _sse_format(data: str) -> str:
    """SSE 单行格式：以 ``data:`` 开头，结尾两个换行。"""
    # content 中若含换行，需逐行加前缀
    lines = data.split("\n")
    return "".join(f"data: {line}\n" for line in lines) + "\n"


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    payload: dict,
    request: Request,
    query_pipeline: QueryPipeline = Depends(get_query_pipeline),
) -> Any:
    """OpenAI 兼容的 chat completions。

    路径分支：
        1) ``stream=true``：返回 SSE 流（仅普通 chat，不支持 tool_call 流）
        2) ``messages`` 含 ``tool_calls`` 或 ``role=tool``：执行 tool 并以 assistant 文本回包
        3) 普通 chat：调 ``QueryPipeline.query``
    """
    # ---- 1) 解析请求体（Pydantic 校验失败 → 400） ----
    try:
        req = ChatCompletionRequest.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"请求体校验失败: {e}") from e

    response_id = _make_response_id()
    created_ts = int(time.time())

    # ---- 2) 取出最后一条 user 消息（作为查询） ----
    last_user_msg: Optional[ChatMessage] = None
    for m in req.messages:
        if m.role == "user":
            last_user_msg = m

    user_query = (last_user_msg.content if last_user_msg else "") or ""

    # ---- 3) tool_call 分支：消息中包含 tool_calls 字段 ----
    tool_calls = _extract_tool_calls(payload)
    if tool_calls:
        # 执行所有 tool_call，汇总结果
        parts: list[str] = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            arguments = tc.get("arguments", {}) or {}
            if not isinstance(arguments, dict):
                try:
                    arguments = json.loads(arguments)
                except Exception:  # noqa: BLE001
                    arguments = {}

            tool_result = handle_tool_call(tool_name, arguments, query_pipeline)
            parts.append(
                f"工具 {tool_name} 执行结果：\n"
                f"```json\n{json.dumps(tool_result, ensure_ascii=False, indent=2)}\n```"
            )
        answer_text = "\n\n".join(parts)
        return ChatCompletionResponse(
            id=response_id,
            created=created_ts,
            model=req.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=answer_text),
                    finish_reason="tool_calls",
                )
            ],
            usage=Usage(),
        )

    # ---- 4) 流式分支 ----
    if req.stream:
        def _event_source():
            # 立即发送心跳，避免浏览器在检索/重排阶段因无数据而超时取消
            yield ": heartbeat\n\n"
            for piece in query_pipeline.query_stream(
                user_query, top_k=req.top_k,
                use_rerank=req.use_rerank if req.use_rerank is not None else True,
                language=req.language,
            ):
                if piece.startswith("__IMAGE_CHUNKS__:"):
                    # 图片引用事件：附加 type=images，前端据此渲染图片
                    payload = piece[len("__IMAGE_CHUNKS__:"):]
                    yield _sse_format(json.dumps({
                        "type": "images",
                        "chunks": json.loads(payload),
                    }, ensure_ascii=False))
                else:
                    chunk = {
                        "id": response_id, "object": "chat.completion.chunk",
                        "created": created_ts, "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield _sse_format(json.dumps(chunk, ensure_ascii=False))
            yield _sse_format("[DONE]")

        return StreamingResponse(_event_source(), media_type="text/event-stream")

    # ---- 5) 普通 chat：调 query pipeline ----
    try:
        result = await run_in_threadpool(
            lambda: query_pipeline.query(
                user_query, top_k=req.top_k,
                use_rerank=req.use_rerank if req.use_rerank is not None else True,
                language=req.language,
            )
        )
        # image_chunks 透传到响应（extra="allow" 允许扩展字段）
        # 前端据此渲染检索到的引用图片
        image_chunks = result.get("image_chunks", [])
        return ChatCompletionResponse(
            id=response_id, created=created_ts, model=req.model,
            choices=[Choice(
                index=0,
                message=ChatMessage(role="assistant", content=result.get("answer", "")),
                finish_reason="stop",
            )],
            usage=Usage(),
            image_chunks=image_chunks,
            run_id=result.get("run_id", ""),
            answer_status=result.get("answer_status", "ok"),
            citations=result.get("sources", []),
        )
    except Exception as e:  # noqa: BLE001
        log.exception("chat_completions 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


def _extract_tool_calls(payload: dict) -> list[dict]:
    """从请求体中抽取所有 tool_call 信息。返回列表，每个元素含 name / arguments。"""
    try:
        messages = payload.get("messages") or []
        for m in messages:
            tc = m.get("tool_calls") if isinstance(m, dict) else None
            if tc and isinstance(tc, list) and tc:
                result: list[dict] = []
                for item in tc:
                    fn = item.get("function") or {}
                    name = fn.get("name") or item.get("name") or ""
                    args = fn.get("arguments") or item.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:  # noqa: BLE001
                            args = {}
                    if name:
                        result.append({"name": name, "arguments": args})
                if result:
                    return result
        return []
    except Exception as e:  # noqa: BLE001
        log.warning("提取 tool_calls 失败: %s", e)
        return []


# ============================================================
# /tools/search
# ============================================================

@router.post("/tools/search", response_model=SearchResponse)
async def tool_search(
    req: SearchRequest,
    query_pipeline: QueryPipeline = Depends(get_query_pipeline),
) -> dict:
    """工具版检索：调 QueryPipeline.query，返回结构化结果。"""
    try:
        # 检索+生成是同步阻塞流程（含 embed/LLM 网络调用），放到线程池执行，
        # 否则会阻塞事件循环，导致同一进程内其他请求排队。
        result = await run_in_threadpool(
            lambda: query_pipeline.query(
                req.query,
                top_k=req.top_k,
                use_rerank=req.use_rerank,
                language=req.language,
            )
        )
        # 把 dict 结果转 SearchResponse
        sources = result.get("sources", [])
        return {
            "query": req.query,
            "results": [SearchResultItem(**s) for s in sources],
            "answer": result.get("answer", ""),
            "language": result.get("language", "en"),
            "used_rerank": result.get("used_rerank", False),
        }
    except Exception as e:  # noqa: BLE001
        log.exception("/tools/search 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/tools/retrieve", response_model=SearchResponse)
async def tool_retrieve(
    req: SearchRequest,
    query_pipeline: QueryPipeline = Depends(get_query_pipeline),
) -> dict:
    """纯检索端点：只做召回 + 重排，不调用生成模型。

    上层 Agent 需要的是证据片段而不是另一份答案；走这个端点可以避免
    "RAG 生成一次、Agent 再生成一次"的重复延迟与费用。
    """
    try:
        result = await run_in_threadpool(
            lambda: query_pipeline.query(
                req.query,
                top_k=req.top_k,
                use_rerank=req.use_rerank,
                language=req.language,
                filter_expr=req.filter,
                generate=False,
            )
        )
        sources = result.get("sources", [])
        return {
            "query": req.query,
            "results": [SearchResultItem(**s) for s in sources],
            "answer": None,
            "language": result.get("language", "en"),
            "used_rerank": result.get("used_rerank", False),
        }
    except Exception as e:  # noqa: BLE001
        log.exception("/tools/retrieve 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================
# /ingest (异步模式)
# ============================================================

def _ingest_worker(
    save_path: Path,
    file_name: str,
    task_id: str,
    task_store: TaskStore,
    ingest_pipeline: IngestPipeline,
) -> None:
    """
    后台线程执行的摄入任务。

    流程：调用 IngestPipeline.ingest_file → 更新 task_store 状态。
    异常时记录错误信息到 task_store，不抛出。
    """
    try:
        # 更新进度：解析中
        # 用用户原始文件名入库，避免 task_id 前缀出现在文档列表与删除接口
        result = ingest_pipeline.ingest_file(str(save_path), file_name=file_name)

        if result.get("status") == "ok":
            # 保存 doc_id / chunk_count，状态接口与前端依赖它定位文档
            task_store.complete(task_id, result=result)
        else:
            task_store.fail(task_id, result.get("error", "摄入失败"))
    except Exception as e:  # noqa: BLE001
        log.exception("后台摄入任务失败 %s: %s", file_name, e)
        task_store.fail(task_id, str(e))
    finally:
        # 清理临时文件（失败不阻塞）
        try:
            save_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


@router.post("/ingest", response_model=IngestAcceptedResponse)
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    _: None = Depends(require_local_token),
    ingest_pipeline: IngestPipeline = Depends(get_ingest_pipeline),
) -> dict:
    """文件上传摄入（异步模式）。

    - 立即保存文件并返回 ``task_id``
    - 后台线程执行解析 → 向量化 → 入库
    - 客户端通过 ``GET /ingest/status/{task_id}`` 轮询进度
    """
    # 1) 取文件名
    file_name = file.filename or "upload.bin"
    task_id = uuid4().hex[:12]

    # 2) 获取任务存储并排队（受并发上限约束，超限返回 429）
    task_store: TaskStore = request.app.state.task_store
    if not task_store.create(task_id, file_name, max_active=config.MAX_INGEST_TASKS):
        raise HTTPException(status_code=429, detail="摄入任务队列已满，请稍后重试")

    # 3) 保存到临时目录
    save_path = UPLOAD_DIR / f"{task_id}_{file_name}"
    try:
        written = 0
        with save_path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > config.MAX_UPLOAD_BYTES:
                    target.close(); save_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="文件超过上传大小限制")
                target.write(chunk)
    except HTTPException:
        task_store.fail(task_id, "上传文件超过限制")
        raise
    except Exception as e:  # noqa: BLE001
        task_store.fail(task_id, f"保存上传文件失败: {e}")
        log.error("写入临时文件失败: %s", e)
        raise HTTPException(status_code=500, detail=f"保存上传文件失败: {e}") from e
    finally:
        await file.close()

    # 6) 提交后台线程执行
    _ingest_executor.submit(
        _ingest_worker,
        save_path=save_path,
        file_name=file_name,
        task_id=task_id,
        task_store=task_store,
        ingest_pipeline=ingest_pipeline,
    )

    log.info("摄入任务已提交: task_id=%s, file=%s", task_id, file_name)
    return IngestAcceptedResponse(
        task_id=task_id,
        status="processing",
        file_name=file_name,
    )


@router.get("/ingest/status/{task_id}", response_model=IngestStatusResponse)
async def ingest_status(
    task_id: str,
    request: Request,
) -> dict:
    """查询异步摄入任务状态。

    返回：
        - ``status="processing"`` 时，``progress`` 字段含当前阶段描述
        - ``status="done"`` 时，``result`` 字段含 ``doc_id`` / ``chunk_count``
        - ``status="error"`` 时，``error`` 字段含错误信息
    """
    try:
        task_info = request.app.state.task_store.get(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    return IngestStatusResponse(
        task_id=task_id,
        status=task_info.get("status", "unknown"),
        file_name=task_info.get("file_name", ""),
        progress=task_info.get("progress", ""),
        result=task_info.get("result"),
        error=task_info.get("error"),
    )


# ============================================================
# /documents
# ============================================================
@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(chroma: Optional[ChromaClientSingleton] = Depends(get_chroma)) -> dict:
    """列出已索引文档。"""
    # 区分"无单例" vs "有单例但未连接" 两种情况
    if chroma is None or not chroma.is_connected:
        return {"total": 0, "documents": [], "chroma_connected": False}
    try:
        docs = await run_in_threadpool(chroma.list_documents)
        items = [DocumentListItem(**d) for d in docs]
        return {"total": len(items), "documents": items, "chroma_connected": True}
    except Exception as e:  # noqa: BLE001
        log.exception("list_documents 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/documents", response_model=DeleteResponse)
async def delete_document(
    file_name: str = Query(..., description="要删除的文件名"),
    _: None = Depends(require_local_token),
    ingest_pipeline: IngestPipeline = Depends(get_ingest_pipeline),
) -> dict:
    """按文件名删除（通过 query 参数传递，兼容含 []、中文等特殊字符的文件名）。"""
    try:
        result = await run_in_threadpool(ingest_pipeline.delete_document, file_name)
        if result.get("status") != "ok":
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "删除失败"),
            )
        return DeleteResponse(
            status=result.get("status", "ok"),
            deleted_count=result.get("deleted_count", 0),
            file_name=file_name,
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("delete_document 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


__all__ = ["router"]
