"""人工证据模板的管理员导入与发布接口。"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile

from api.auth_routes import require_admin, require_viewer
from config import config
from utils.evidence_excel import import_workbook
from utils.evidence_store import EvidenceStore

router = APIRouter(prefix="/evidence", tags=["evidence"])


@router.get("/versions", dependencies=[Depends(require_viewer)])
def list_versions(request: Request, domain: str | None = None) -> dict:
    store: EvidenceStore = request.app.state.evidence_store
    return {"versions": store.list_versions(domain)}


@router.post("/import", dependencies=[Depends(require_admin)])
async def import_evidence(
    request: Request,
    name: str = Query(..., min_length=1),
    file: UploadFile = File(...),
) -> dict:
    if not (file.filename or "").lower().endswith(".xlsx"):
        raise HTTPException(status_code=422, detail="仅支持 .xlsx 证据模板")
    content = await file.read(config.MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="模板超过上传大小限制")
    try:
        with TemporaryDirectory(prefix="evidence-") as directory:
            path = Path(directory) / "evidence.xlsx"
            path.write_bytes(content)
            return import_workbook(path, request.app.state.evidence_store, name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/versions/{version_id}/publish", dependencies=[Depends(require_admin)])
def publish_version(version_id: str, request: Request) -> dict:
    try:
        request.app.state.evidence_store.publish(version_id)
        return request.app.state.evidence_store.get_version(version_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/versions/{version_id}/activate", dependencies=[Depends(require_admin)])
def activate_version(version_id: str, request: Request) -> dict:
    try:
        request.app.state.evidence_store.activate(version_id)
        return request.app.state.evidence_store.get_version(version_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
