# Test values are synthetic fixtures for software behavior.
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

def test_additive_migration_repeated_preserves_legacy_rows(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE sessions(id VARCHAR(64) PRIMARY KEY,user_id VARCHAR(64),title VARCHAR(255),created_at DATETIME,updated_at DATETIME)")
        db.execute("INSERT INTO sessions VALUES ('legacy','alice','old chat','2026-09-01','2026-09-01')")
        db.execute("CREATE TABLE feedback(id INTEGER PRIMARY KEY,session_id VARCHAR(64),message_id VARCHAR(64),feedback_type VARCHAR(16),reason TEXT,correction TEXT,created_at DATETIME)")
        db.execute("CREATE TABLE chat_messages(id INTEGER PRIMARY KEY,message_id VARCHAR(64),session_id VARCHAR(64),user_id VARCHAR(64),role VARCHAR(16),content TEXT,agent VARCHAR(64),created_at DATETIME)")
    root = Path(__file__).resolve().parents[3]
    environment = {**os.environ, "MIGRATION_DATABASE_URL": "sqlite+aiosqlite:///" + path.as_posix()}
    for _ in range(2):
        result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=root, env=environment, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT user_id,title FROM sessions WHERE id='legacy'").fetchone() == ("alice", "old chat")
        assert "run_id" in {r[1] for r in db.execute("PRAGMA table_info(chat_messages)")}
        assert db.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "20261004_event_sequence"
