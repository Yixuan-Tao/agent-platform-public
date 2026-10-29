# Test values are synthetic fixtures for software behavior.
from unittest.mock import patch

from utils.tls import certificate_fingerprint, ensure_self_signed_certificate


def test_certificate_generator_returns_existing_pair(tmp_path):
    cert, key = tmp_path / "server.crt", tmp_path / "server.key"
    cert.write_text("cert")
    key.write_text("key")
    with patch("utils.tls.subprocess.run") as run:
        assert ensure_self_signed_certificate(cert, key) == (cert, key)
        run.assert_not_called()


def test_certificate_fingerprint_is_sha256_display_format(tmp_path):
    cert = tmp_path / "server.crt"
    cert.write_bytes(b"certificate")
    fingerprint = certificate_fingerprint(cert)
    assert len(fingerprint.split(":")) == 32
