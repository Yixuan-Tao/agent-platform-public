# Test values are synthetic fixtures for software behavior.
"""网关 → RAG 文档代理测试：凭据透传、参数映射、错误状态透传。"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_platform.core.config import Settings
from agent_platform.core.security import AccessStore
from app.middleware import APIKeyMiddleware
from app.routes import documents
from app.services.rag_client import RagClient, RagServiceError


class _FakeClient(RagClient):
    """记录调用参数并按预设返回，替代真实 RAG 服务。"""

    def __init__(self, settings, result=None, error=None):
        super().__init__(settings)
        self._result = result
        self._error = error
        self.calls: list[dict] = []

    async def request(self, method, path, *, timeout=None, **kwargs):
        self.calls.append({"method": method, "path": path, **kwargs})
        if self._error is not None:
            raise self._error
        return self._result


def _build(tmp_path, fake: _FakeClient):
    settings = Settings(api_key="service-key", _env_file=None)
    app = FastAPI()
    app.state.settings = settings
    app.state.access_store = AccessStore(tmp_path / "access.db")
    app.include_router(documents.router)
    app.add_middleware(APIKeyMiddleware)
    app.dependency_overrides[documents.get_rag_client] = lambda: fake
    return TestClient(app)


def test_upload_sends_credentials_and_returns_upstream_body(tmp_path):
    settings = Settings(api_key="service-key", rag_service_token="svc-token", _env_file=None)
    fake = _FakeClient(settings, result=(200, {"doc_id": "abc", "status": "queued"}))
    client = _build(tmp_path, fake)

    resp = client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": "service-key"},
        files={"file": ("spec.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"doc_id": "abc", "status": "queued"}
    assert fake.calls[0]["method"] == "POST"
    assert fake.calls[0]["path"] == "/ingest"
    assert fake.auth_headers() == {
        "X-Local-Token": "svc-token",
        "Authorization": "Bearer svc-token",
    }


def test_delete_uses_file_name_query_param(tmp_path):
    """RAG 的删除接口按 file_name 查询，网关此前误传 ids。"""
    settings = Settings(api_key="service-key", rag_service_token="svc-token", _env_file=None)
    fake = _FakeClient(settings, result=(200, {"status": "ok", "deleted_count": 3}))
    client = _build(tmp_path, fake)

    resp = client.request(
        "DELETE",
        "/api/v1/documents",
        params={"file_name": "EMC 标准.pdf"},
        headers={"X-API-Key": "service-key"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["deleted_count"] == 3
    assert fake.calls[0]["params"] == {"file_name": "EMC 标准.pdf"}


def test_upstream_error_status_is_preserved(tmp_path):
    """下游 401/404 不能再被包装成 200。"""
    settings = Settings(api_key="service-key", _env_file=None)
    fake = _FakeClient(settings, result=(401, {"detail": "缺少或无效的本机写操作令牌"}))
    client = _build(tmp_path, fake)

    resp = client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": "service-key"},
        files={"file": ("spec.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"]["detail"] == "缺少或无效的本机写操作令牌"


def test_unreachable_rag_maps_to_502(tmp_path):
    settings = Settings(api_key="service-key", _env_file=None)
    fake = _FakeClient(
        settings, error=RagServiceError("RAG 服务不可达: boom", status_code=502)
    )
    client = _build(tmp_path, fake)

    resp = client.get("/api/v1/documents", headers={"X-API-Key": "service-key"})
    assert resp.status_code == 502


def test_documents_require_credentials(tmp_path):
    settings = Settings(api_key="service-key", _env_file=None)
    fake = _FakeClient(settings, result=(200, {"total": 0, "documents": []}))
    client = _build(tmp_path, fake)
    assert client.get("/api/v1/documents").status_code == 401


async def test_rag_client_builds_auth_headers_without_token():
    settings = Settings(api_key="service-key", rag_service_token="", _env_file=None)
    assert RagClient(settings).auth_headers() == {}


async def test_rag_client_raises_on_non_2xx(monkeypatch):
    settings = Settings(
        api_key="service-key", rag_service_token="t", rag_service_url="http://rag:8010",
        _env_file=None,
    )
    client = RagClient(settings)
    sent: dict = {}

    class _Resp:
        status_code = 403
        headers = {"content-type": "application/json"}
        text = '{"detail":"forbidden"}'

        def json(self):
            return {"detail": "forbidden"}

    class _AsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, method, url, headers=None, **kwargs):
            sent["url"] = url
            sent["headers"] = headers
            return _Resp()

    monkeypatch.setattr("app.services.rag_client.httpx.AsyncClient", _AsyncClient)
    with pytest.raises(RagServiceError) as exc:
        await client.request_or_raise("GET", "/documents")
    assert exc.value.status_code == 403
    assert sent["url"] == "http://rag:8010/documents"
    assert sent["headers"]["X-Local-Token"] == "t"


async def test_rag_client_maps_transport_error(monkeypatch):
    settings = Settings(api_key="k", rag_service_url="http://rag:8010", _env_file=None)
    client = RagClient(settings)

    class _AsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, *a, **k):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("app.services.rag_client.httpx.AsyncClient", _AsyncClient)
    with pytest.raises(RagServiceError) as exc:
        await client.request("GET", "/health")
    assert exc.value.status_code == 502
