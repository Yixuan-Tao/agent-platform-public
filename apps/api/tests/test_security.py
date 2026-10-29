# Test values are synthetic fixtures for software behavior.
"""平台 core 安全模块（账号/会话/角色/审计）与模型网关校验测试。"""

from __future__ import annotations

import os

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from agent_platform.core.config import Settings, get_settings
from agent_platform.core.llm.gateway import GatewayStatus, validate_gateway
from agent_platform.core.security import (
    AccessStore,
    AuditStore,
    AuthorizationError,
    current_identity,
    require_admin,
    require_login,
)


# ------------------------------------------------------------
# AccessStore
# ------------------------------------------------------------


def test_access_store_authenticate_and_roles(tmp_path):
    store = AccessStore(tmp_path / "access.db")
    store.create_user("admin", "strong-password", "admin")
    store.create_user("viewer", "viewer-password", "viewer")

    with pytest.raises(AuthorizationError):
        store.authenticate("admin", "wrong-password")

    token = store.authenticate("admin", "strong-password")
    assert store.identity(token, "admin")["username"] == "admin"
    assert store.identity(token, "viewer")["role"] == "admin"  # admin 满足 viewer 权限

    viewer_token = store.authenticate("viewer", "viewer-password")
    assert store.identity(viewer_token)["role"] == "viewer"
    with pytest.raises(AuthorizationError):
        store.identity(viewer_token, "admin")

    store.revoke(viewer_token)
    with pytest.raises(AuthorizationError):
        store.identity(viewer_token)


def test_access_store_delete_last_admin_guard(tmp_path):
    store = AccessStore(tmp_path / "access.db")
    store.create_user("admin", "strong-password", "admin")
    with pytest.raises(AuthorizationError):
        store.delete_user("admin")


def test_access_store_rejects_weak_password(tmp_path):
    store = AccessStore(tmp_path / "access.db")
    with pytest.raises(ValueError):
        store.create_user("admin", "short", "admin")


# ------------------------------------------------------------
# AuditStore
# ------------------------------------------------------------


def test_audit_store_records_and_lists(tmp_path):
    store = AuditStore(tmp_path / "audit.db")
    store.record(username="alice", source_ip="127.0.0.1", action="POST /api/v1/ingest", status="200")
    rows = store.list_recent()
    assert len(rows) == 1
    assert rows[0]["username"] == "alice"
    assert rows[0]["action"] == "POST /api/v1/ingest"


# ------------------------------------------------------------
# FastAPI 依赖
# ------------------------------------------------------------


def _app(tmp_path, settings):
    app = FastAPI()
    app.state.settings = settings
    app.state.access_store = AccessStore(tmp_path / "access.db")
    app.state.access_store.create_user("admin", "strong-password", "admin")
    app.state.access_store.create_user("viewer", "viewer-password", "viewer")

    @app.get("/identity")
    def identity(ident: dict = Depends(current_identity)):
        return ident

    @app.get("/login-required")
    def login_required(ident: dict = Depends(require_login)):
        return ident

    @app.get("/admin-required")
    def admin_required(ident: dict = Depends(require_admin)):
        return ident

    return app


def test_bearer_session_roles_via_dependencies(tmp_path):
    settings = Settings(lan_enabled=False, api_key="service-key")
    app = _app(tmp_path, settings)
    client = TestClient(app)
    store = app.state.access_store

    admin_token = store.authenticate("admin", "strong-password")
    viewer_token = store.authenticate("viewer", "viewer-password")

    headers_admin = {"Authorization": f"Bearer {admin_token}"}
    headers_viewer = {"Authorization": f"Bearer {viewer_token}"}

    assert client.get("/identity", headers=headers_admin).json()["role"] == "admin"
    assert client.get("/login-required", headers=headers_viewer).json()["username"] == "viewer"
    assert client.get("/admin-required", headers=headers_admin).status_code == 200
    assert client.get("/admin-required", headers=headers_viewer).status_code == 403
    assert client.get("/admin-required").status_code == 401


def test_x_api_key_is_service_role_and_cannot_administer(tmp_path):
    """服务 API Key 只用于业务调用，不再默认等价于人类管理员。"""
    settings = Settings(api_key="service-key")
    app = _app(tmp_path, settings)
    client = TestClient(app)

    service = client.get("/login-required", headers={"X-API-Key": "service-key"})
    assert service.status_code == 200
    assert service.json()["role"] == "service"

    denied = client.get("/admin-required", headers={"X-API-Key": "service-key"})
    assert denied.status_code == 403


def test_admin_api_key_acts_as_automation_identity(tmp_path):
    settings = Settings(api_key="service-key", admin_api_key="rotate-me")
    app = _app(tmp_path, settings)
    client = TestClient(app)

    response = client.get("/admin-required", headers={"Authorization": "Bearer rotate-me"})
    assert response.status_code == 200
    assert response.json()["username"] == "automation"


# ------------------------------------------------------------
# 模型网关校验
# ------------------------------------------------------------


def test_validate_gateway_requires_https():
    settings = Settings(LLM_BASE_URL="http://plain-http.example.com/v1", LLM_API_KEY="k", _env_file=None)
    status = validate_gateway(settings)
    assert status.ok is False
    assert "HTTPS" in status.reason


def test_validate_gateway_host_allowlist():
    settings = Settings(
        LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1",
        LLM_API_KEY="k",
        model_gateway_allowed_hosts="dashscope.aliyuncs.com,other.example.com",
        _env_file=None,
    )
    assert validate_gateway(settings).ok is True

    blocked = Settings(
        LLM_BASE_URL="https://evil.example.com/v1",
        LLM_API_KEY="k",
        model_gateway_allowed_hosts="dashscope.aliyuncs.com",
        _env_file=None,
    )
    status = validate_gateway(blocked)
    assert status.ok is False
    assert "不在允许列表" in status.reason


def test_validate_gateway_requires_api_key():
    settings = Settings(LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1", LLM_API_KEY="", _env_file=None)
    status = validate_gateway(settings)
    assert status.ok is False
    assert "API Key" in status.reason


def test_validate_gateway_dev_mode_skips_allowlist():
    settings = Settings(
        LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1",
        LLM_API_KEY="k",
        model_gateway_allowed_hosts="",
        _env_file=None,
    )
    assert validate_gateway(settings).ok is True


def test_settings_load_new_security_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("LAN_ENABLED", "true")
    monkeypatch.setenv("TLS_CERT_FILE", "server.crt")
    monkeypatch.setenv("TLS_KEY_FILE", "server.key")
    monkeypatch.setenv("INITIAL_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("INITIAL_ADMIN_PASSWORD", "strong-password")
    monkeypatch.setenv("ADMIN_API_KEY", "rotate-me")
    monkeypatch.setenv("MODEL_GATEWAY_ALLOWED_HOSTS", "dashscope.aliyuncs.com")
    settings = Settings()
    assert settings.lan_enabled is True
    assert settings.tls_cert_file == "server.crt"
    assert settings.initial_admin_username == "admin"
    assert settings.admin_api_key == "rotate-me"
    assert settings.model_gateway_allowed_hosts == "dashscope.aliyuncs.com"
