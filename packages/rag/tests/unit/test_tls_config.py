# Test values are synthetic fixtures for software behavior.
from pathlib import Path

import pytest

from utils.tls import tls_options


def test_lan_requires_certificate(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LAN_ENABLED", "true")
    monkeypatch.setenv("TLS_CERT_FILE", str(tmp_path / "missing.crt"))
    monkeypatch.setenv("TLS_KEY_FILE", str(tmp_path / "missing.key"))
    with pytest.raises(RuntimeError):
        tls_options()


def test_local_mode_does_not_require_certificate(monkeypatch):
    monkeypatch.delenv("LAN_ENABLED", raising=False)
    assert tls_options() == {}
