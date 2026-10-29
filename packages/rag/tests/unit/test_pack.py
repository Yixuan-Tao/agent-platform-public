# Test values are synthetic fixtures for software behavior.
import hashlib
import json

import pytest

from utils.pack import validate_pack


def test_validate_pack_checks_declared_file_hash(tmp_path):
    document = tmp_path / "standard.pdf"
    document.write_bytes(b"emc")
    (tmp_path / "manifest.json").write_text(json.dumps({"version": "v1", "files": [{"path": "standard.pdf", "sha256": hashlib.sha256(b"emc").hexdigest(), "type": "pdf"}]}), encoding="utf-8")

    assert validate_pack(tmp_path)["version"] == "v1"


def test_validate_pack_rejects_tampered_file(tmp_path):
    (tmp_path / "standard.pdf").write_bytes(b"changed")
    (tmp_path / "manifest.json").write_text(json.dumps({"version": "v1", "files": [{"path": "standard.pdf", "sha256": "0" * 64, "type": "pdf"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        validate_pack(tmp_path)
