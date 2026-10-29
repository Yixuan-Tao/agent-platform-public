# Test values are synthetic fixtures for software behavior.
"""摄入任务状态必须能回读 doc_id / chunk_count。"""

import json

from utils.local_runtime import TaskStore


def test_complete_stores_and_reads_result(tmp_path):
    store = TaskStore(tmp_path / "runtime.db")
    assert store.create("t1", "spec.pdf") is True
    assert store.get("t1")["result"] is None

    store.complete("t1", result={"doc_id": "abc123", "chunk_count": 42})
    info = store.get("t1")
    assert info["status"] == "done"
    assert info["result"] == {"doc_id": "abc123", "chunk_count": 42}


def test_complete_without_result_keeps_result_none(tmp_path):
    store = TaskStore(tmp_path / "runtime.db")
    store.create("t2", "a.pdf")
    store.complete("t2")
    assert store.get("t2")["result"] is None


def test_failed_task_has_no_result(tmp_path):
    store = TaskStore(tmp_path / "runtime.db")
    store.create("t3", "b.pdf")
    store.fail("t3", "解析失败")
    info = store.get("t3")
    assert info["status"] == "error"
    assert info["error"] == "解析失败"
    assert info["result"] is None


def test_legacy_database_without_result_column_is_migrated(tmp_path):
    """旧库没有 result 列，升级后应自动补齐而不是报错。"""
    import sqlite3

    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE ingest_tasks (id TEXT PRIMARY KEY, status TEXT, "
            "progress TEXT, file_name TEXT, error TEXT, updated_at REAL)"
        )
        db.execute(
            "INSERT INTO ingest_tasks VALUES (?, ?, ?, ?, ?, ?)",
            ("old", "done", "完成", "old.pdf", None, 1.0),
        )

    store = TaskStore(path)
    store.complete("old", result={"doc_id": "x", "chunk_count": 1})
    assert store.get("old")["result"] == {"doc_id": "x", "chunk_count": 1}


def test_result_is_json_serialised_text(tmp_path):
    import sqlite3

    path = tmp_path / "runtime.db"
    store = TaskStore(path)
    store.create("t4", "c.pdf")
    store.complete("t4", result={"doc_id": "d", "chunk_count": 2})
    with sqlite3.connect(path) as db:
        raw = db.execute("SELECT result FROM ingest_tasks WHERE id='t4'").fetchone()[0]
    assert json.loads(raw) == {"doc_id": "d", "chunk_count": 2}
