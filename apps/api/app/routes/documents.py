"""文档管理：代理到 RAG 知识服务（:8010）。

统一通过 :class:`RagClient` 携带服务凭据，并把上游错误码透传为网关 HTTP 状态，
避免下游 4xx/5xx 被包装成 200 而掩盖失败。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from agent_platform.core.security import require_login
from app.services.rag_client import RagClient, RagServiceError

router = APIRouter(prefix="/api/v1", tags=["documents"])


def get_rag_client() -> RagClient:
    return RagClient()


async def _proxy(client: RagClient, method: str, path: str, **kwargs: Any) -> Any:
    """转发请求；上游非 2xx 时以相同状态码抛 HTTPException。"""
    try:
        status, body = await client.request(method, path, **kwargs)
    except RagServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return body


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    _: dict = Depends(require_login),
    client: RagClient = Depends(get_rag_client),
) -> Any:
    content = await file.read()
    return await _proxy(
        client,
        "POST",
        "/ingest",
        files={"file": (file.filename, content, file.content_type)},
    )


@router.get("/ingest/status/{task_id}")
async def ingest_status(
    task_id: str, client: RagClient = Depends(get_rag_client)
) -> Any:
    return await _proxy(client, "GET", f"/ingest/status/{task_id}")


@router.get("/documents")
async def list_documents(
    _: dict = Depends(require_login),
    client: RagClient = Depends(get_rag_client),
) -> Any:
    return await _proxy(client, "GET", "/documents")


@router.delete("/documents")
async def delete_documents(
    file_name: str = Query(..., description="要删除的文件名（RAG 侧按文件名删除）"),
    _: dict = Depends(require_login),
    client: RagClient = Depends(get_rag_client),
) -> Any:
    return await _proxy(client, "DELETE", "/documents", params={"file_name": file_name})
