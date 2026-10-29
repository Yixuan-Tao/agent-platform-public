"""本地脱敏审计记录。"""
import sqlite3
import time
from pathlib import Path


class AuditStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS audit_events (created_at REAL, username TEXT, source_ip TEXT, action TEXT, status TEXT, run_id TEXT, pack_version TEXT)")

    def _db(self):
        return sqlite3.connect(self.path)

    def record(self, *, username: str, source_ip: str, action: str, status: str, run_id: str = "", pack_version: str = "") -> None:
        with self._db() as db:
            db.execute("INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?)", (time.time(), username, source_ip, action, status, run_id, pack_version))

    def list_recent(self, limit: int = 100) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT created_at,username,source_ip,action,status,run_id,pack_version FROM audit_events ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        keys = ("created_at", "username", "source_ip", "action", "status", "run_id", "pack_version")
        return [dict(zip(keys, row)) for row in rows]
