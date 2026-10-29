# Test values are synthetic fixtures for software behavior.
from pathlib import Path

import pytest

from utils.access_control import AccessStore, AuthorizationError


def test_authenticates_user_and_issues_expiring_session(tmp_path: Path):
    store = AccessStore(tmp_path / "access.db", session_ttl_seconds=60)
    store.create_user("admin", "correct horse", "admin")

    token = store.authenticate("admin", "correct horse")

    assert store.require_role(token, "viewer") == "admin"
    assert store.require_role(token, "admin") == "admin"


def test_viewer_cannot_use_admin_operation(tmp_path: Path):
    store = AccessStore(tmp_path / "access.db")
    store.create_user("reader", "safe password", "viewer")
    token = store.authenticate("reader", "safe password")

    with pytest.raises(AuthorizationError):
        store.require_role(token, "admin")


def test_rejects_invalid_password_and_revoked_session(tmp_path: Path):
    store = AccessStore(tmp_path / "access.db")
    store.create_user("admin", "safe password", "admin")
    with pytest.raises(AuthorizationError):
        store.authenticate("admin", "wrong")
    token = store.authenticate("admin", "safe password")
    store.revoke(token)
    with pytest.raises(AuthorizationError):
        store.require_role(token, "viewer")
