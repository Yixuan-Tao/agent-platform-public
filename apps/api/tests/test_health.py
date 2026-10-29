# Test values are synthetic fixtures for software behavior.
"""健康检查接口测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest
from agent_platform.core.config import Settings

from app.main import create_app


@pytest.fixture(autouse=True)
def isolated_health_storage(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'health.db'}",
                        runtime_dir=str(tmp_path / "runtime"), chroma_dir=str(tmp_path / "chroma"),
                        log_dir=str(tmp_path / "logs"), api_key="dev-key-change-me")
    monkeypatch.setattr("app.main.get_settings", lambda: settings)


def test_health() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"


def test_api_key_required() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/v1/agents")
        assert resp.status_code == 401


def test_api_key_accepted() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/agents", headers={"X-API-Key": "dev-key-change-me"}
        )
        assert resp.status_code == 200
        assert "agents" in resp.json()


def test_root_index_is_public_and_explains_service() -> None:
    """直接访问 API 端口不应只看到 401 JSON。"""
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["kind"] == "api-gateway"
        assert body["health"] == "/health"
        assert body["docs"] == "/docs"
        assert "frontend" in body


def test_favicon_does_not_trigger_auth_error() -> None:
    """浏览器会自动请求 favicon，不应回 401 污染日志。"""
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/favicon.ico").status_code != 401


def test_unauthorized_response_carries_hint_and_www_authenticate() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/v1/agents")
        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate", "").startswith("Bearer")
        assert "hint" in resp.json()
