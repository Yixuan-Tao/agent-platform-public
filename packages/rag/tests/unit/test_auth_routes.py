# Test values are synthetic fixtures for software behavior.
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api.auth_routes import require_viewer, router
from utils.access_control import AccessStore


def test_login_and_current_user(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.access_store = AccessStore(tmp_path / "access.db")
    app.state.access_store.create_user("admin", "safe password", "admin")
    client = TestClient(app)

    login = client.post("/auth/login", json={"username": "admin", "password": "safe password"})
    assert login.status_code == 200
    token = login.json()["access_token"]
    current = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert current.json() == {"username": "admin", "role": "admin"}


def test_current_user_rejects_missing_bearer_token(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.access_store = AccessStore(tmp_path / "access.db")
    assert TestClient(app).get("/auth/me").status_code == 401


def test_viewer_dependency_requires_session_only_in_lan_mode(tmp_path, monkeypatch):
    app = FastAPI()
    app.state.access_store = AccessStore(tmp_path / "access.db")
    @app.get("/protected", dependencies=[Depends(require_viewer)])
    def protected(): return {"ok": True}
    monkeypatch.setenv("LAN_ENABLED", "true")
    assert TestClient(app).get("/protected").status_code == 401
    monkeypatch.delenv("LAN_ENABLED")
    assert TestClient(app).get("/protected").status_code == 200


def test_admin_api_key_is_accepted_without_persisting_the_key(tmp_path, monkeypatch):
    app = FastAPI()
    app.include_router(router)
    app.state.access_store = AccessStore(tmp_path / "access.db")
    monkeypatch.setenv("ADMIN_API_KEY", "rotatable-key")
    response = TestClient(app).get("/auth/me", headers={"Authorization": "Bearer rotatable-key"})
    assert response.status_code == 200
    assert response.json() == {"username": "automation", "role": "admin"}


def test_admin_can_manage_users_and_logout_revokes_session(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.access_store = AccessStore(tmp_path / "access.db")
    app.state.access_store.create_user("admin", "safe password", "admin")
    client = TestClient(app)
    admin_token = client.post("/auth/login", json={"username": "admin", "password": "safe password"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}

    assert client.post("/auth/users", headers=headers, json={"username": "viewer", "password": "viewer password", "role": "viewer"}).status_code == 200
    assert client.get("/auth/users", headers=headers).json()["users"] == [
        {"username": "admin", "role": "admin"}, {"username": "viewer", "role": "viewer"},
    ]
    assert client.delete("/auth/users/viewer", headers=headers).status_code == 200
    assert client.post("/auth/logout", headers=headers).status_code == 200
    assert client.get("/auth/me", headers=headers).status_code == 401


def test_cannot_delete_last_admin(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.access_store = AccessStore(tmp_path / "access.db")
    app.state.access_store.create_user("admin", "safe password", "admin")
    client = TestClient(app)
    token = client.post("/auth/login", json={"username": "admin", "password": "safe password"}).json()["access_token"]
    assert client.delete("/auth/users/admin", headers={"Authorization": f"Bearer {token}"}).status_code == 422
