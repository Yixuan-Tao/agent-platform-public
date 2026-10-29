# Test values are synthetic fixtures for software behavior.
from utils.audit import AuditStore


def test_audit_store_persists_only_safe_metadata(tmp_path):
    store = AuditStore(tmp_path / "audit.db")
    store.record(username="admin", source_ip="192.0.2.2", action="compare.emc", status="ok", run_id="run-1", pack_version="v1")

    record = store.list_recent(limit=1)[0]
    assert record["username"] == "admin"
    assert record["run_id"] == "run-1"
    assert "question" not in record
    assert "output" not in record
