# Test values are synthetic fixtures for software behavior.
import json
import subprocess
import sys
from pathlib import Path

_RAG_ROOT = Path(__file__).resolve().parents[2]


def test_pack_validate_cli_reports_version(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"version": "v1", "files": []}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/pack.py", "validate", str(tmp_path)],
        capture_output=True, text=True, check=False, cwd=_RAG_ROOT,
    )
    assert result.returncode == 0
    assert "v1" in result.stdout


def test_pack_import_dry_run_only_validates(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"version": "v1", "files": []}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/pack.py", "import", str(tmp_path), "--dry-run"],
        capture_output=True, text=True, check=False, cwd=_RAG_ROOT,
    )
    assert result.returncode == 0
    assert '"status": "validated"' in result.stdout
