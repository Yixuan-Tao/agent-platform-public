# Test values are synthetic fixtures for software behavior.
from config import _Config


def test_lan_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LAN_ENABLED", raising=False)
    monkeypatch.delenv("HOST", raising=False)
    assert _Config().HOST == "127.0.0.1"


def test_lan_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("LAN_ENABLED", "true")
    monkeypatch.setenv("HOST", "0.0.0.0")
    assert _Config().HOST == "0.0.0.0"


def test_explicit_host_is_honored_even_when_lan_disabled(monkeypatch):
    """Docker 部署传 HOST=0.0.0.0 时必须生效（api 容器经容器网络访问 rag）。"""
    monkeypatch.delenv("LAN_ENABLED", raising=False)
    monkeypatch.setenv("HOST", "0.0.0.0")
    assert _Config().HOST == "0.0.0.0"
