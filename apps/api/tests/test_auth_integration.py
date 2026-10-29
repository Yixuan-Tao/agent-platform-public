# Test values are synthetic fixtures for software behavior.
"""身份中间件集成测试：服务密钥 / 管理员密钥 / 用户会话三种凭据。"""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from agent_platform.core.config import Settings
from agent_platform.core.security import AccessStore, require_login
from app.middleware import APIKeyMiddleware
from app.routes import auth


def _build(tmp_path):
    settings = Settings(
        api_key="service-key",
        admin_api_key="rotate-me",
        initial_admin_username="",
        initial_admin_password="",
        _env_file=None,
    )
    app = FastAPI()
    app.state.settings = settings
    app.state.access_store = AccessStore(tmp_path / "access.db")
    app.state.access_store.create_user("alice", "alice-password", "admin")
    app.state.access_store.create_user("bob", "bob-password", "viewer")
    app.include_router(auth.router)

    @app.get("/protected")
    def protected(identity: dict = Depends(require_login)) -> dict:
        return identity

    app.add_middleware(APIKeyMiddleware)
    return TestClient(app)


def _token(client: TestClient, username: str, password: str) -> str:
    resp = client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_login_token_passes_middleware_and_route_dependency(tmp_path):
    """登录签发的 Bearer 会话此前会被 API Key 中间件拦截，这里锁住修复。"""
    client = _build(tmp_path)
    token = _token(client, "alice", "alice-password")

    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["username"] == "alice"
    assert resp.json()["role"] == "admin"


def test_service_key_is_scoped_role(tmp_path):
    client = _build(tmp_path)
    resp = client.get("/protected", headers={"X-API-Key": "service-key"})
    assert resp.status_code == 200
    assert resp.json() == {"username": "api-key", "role": "service", "auth": "api_key"}


def test_service_key_cannot_list_users(tmp_path):
    """浏览器里的共享 API Key 不应再等价于管理员。"""
    client = _build(tmp_path)
    resp = client.get("/api/v1/auth/users", headers={"X-API-Key": "service-key"})
    assert resp.status_code == 403


def test_admin_api_key_can_list_users(tmp_path):
    client = _build(tmp_path)
    resp = client.get(
        "/api/v1/auth/users", headers={"Authorization": "Bearer rotate-me"}
    )
    assert resp.status_code == 200, resp.text
    usernames = {u["username"] for u in resp.json()["users"]}
    assert {"alice", "bob"} <= usernames


def test_invalid_and_missing_credentials_rejected(tmp_path):
    client = _build(tmp_path)
    assert client.get("/protected").status_code == 401
    assert (
        client.get("/protected", headers={"Authorization": "Bearer nope"}).status_code
        == 401
    )
    assert client.get("/protected", headers={"X-API-Key": "wrong"}).status_code == 401


def test_revoked_session_rejected(tmp_path):
    client = _build(tmp_path)
    token = _token(client, "bob", "bob-password")
    assert client.get(
        "/protected", headers={"Authorization": f"Bearer {token}"}
    ).status_code == 200
    client.app.state.access_store.revoke(token)
    assert (
        client.get(
            "/protected", headers={"Authorization": f"Bearer {token}"}
        ).status_code
        == 401
    )
