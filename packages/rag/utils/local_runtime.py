"""仅本机运行时的令牌与摄入任务持久化。"""
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path

from utils.logger import log


class LocalTokenStore:
    """本地 API 令牌存储，首次运行时生成随机令牌并持久化到文件。"""

    def __init__(self, path: Path):
        self.path = path

    def get_or_create(self) -> str:
        """获取已有令牌，不存在则生成新的随机令牌并写入文件（权限 0o600）。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            return self.path.read_text(encoding="utf-8").strip()
        # 生成新令牌并原子写入
        token = secrets.token_urlsafe(32)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token)
        return token


class TaskStore:
    """摄入任务状态持久化（SQLite），支持创建/完成/失败/查询。"""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS ingest_tasks "
                "(id TEXT PRIMARY KEY, status TEXT, progress TEXT, "
                "file_name TEXT, error TEXT, result TEXT, updated_at REAL)"
            )
            # 兼容旧库：早期版本没有 result 列，导致状态接口拿不到 doc_id/chunk_count
            columns = {row[1] for row in db.execute("PRAGMA table_info(ingest_tasks)")}
            if "result" not in columns:
                db.execute("ALTER TABLE ingest_tasks ADD COLUMN result TEXT")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def create(self, task_id: str, file_name: str, max_active: int | None = None) -> bool:
        """创建任务；设置上限时以 SQLite 事务原子限制活跃任务数，返回是否入队。"""
        with self._connect() as db:
            if max_active is not None:
                active = db.execute(
                    "SELECT COUNT(*) FROM ingest_tasks WHERE status='processing'"
                ).fetchone()[0]
                if active >= max_active:
                    return False
            db.execute(
                "INSERT OR REPLACE INTO ingest_tasks "
                "(id, status, progress, file_name, error, result, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    "processing",
                    "文件已接收，排队处理中...",
                    file_name,
                    None,
                    None,
                    time.time(),
                ),
            )
        return True

    def complete(self, task_id: str, result: dict | None = None) -> None:
        """标记完成并保存结果（doc_id / chunk_count），供状态接口回读。"""
        self._update(
            task_id=task_id,
            status="done",
            progress="完成",
            error=None,
            result=result,
        )

    def fail(self, task_id: str, error: str) -> None:
        self._update(task_id=task_id, status="error", progress="失败", error=error)

    def get(self, task_id: str) -> dict:
        with self._connect() as db:
            row = db.execute(
                "SELECT status, progress, file_name, error, result "
                "FROM ingest_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        if not row:
            raise KeyError(task_id)
        data = dict(zip(("status", "progress", "file_name", "error"), row))
        raw = row[4]
        try:
            data["result"] = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            data["result"] = None
        return data

    def _write(
        self, task_id: str, status: str, progress: str,
        file_name: str, error: str | None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO ingest_tasks "
                "(id, status, progress, file_name, error, result, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task_id, status, progress, file_name, error, None, time.time()),
            )

    def _update(
        self,
        task_id: str,
        status: str,
        progress: str,
        error: str | None,
        result: dict | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE ingest_tasks SET status = ?, progress = ?, "
                "error = ?, result = ?, updated_at = ? WHERE id = ?",
                (
                    status,
                    progress,
                    error,
                    json.dumps(result, ensure_ascii=False) if result else None,
                    time.time(),
                    task_id,
                ),
            )

    def prune(self, ttl_seconds: float = 7 * 24 * 3600) -> int:
        """按 updated_at 清理已结束的过期任务（默认 7 天），返回删除条数。

        说明：仅清理 status IN ('done', 'error') 的终态记录，避免误删处理中任务。
        建议在 lifespan 启动阶段调用一次，或接入后台定时任务。
        """
        threshold = time.time() - ttl_seconds
        with self._connect() as db:
            cur = db.execute(
                "DELETE FROM ingest_tasks "
                "WHERE updated_at < ? AND status IN ('done', 'error')",
                (threshold,),
            )
            deleted = cur.rowcount
        if deleted:
            log.info("TaskStore 已清理 %d 条过期任务（TTL=%.0fs）", deleted, ttl_seconds)
        return deleted

    def recover_interrupted(self) -> int:
        """将重启前未完成任务标记为失败，避免形成半成品索引。"""
        with self._connect() as db:
            cur = db.execute(
                "UPDATE ingest_tasks SET status='error', progress='已中断', "
                "error='服务重启导致任务中断，请重新上传资料包或文件', updated_at=? "
                "WHERE status='processing'", (time.time(),),
            )
        return cur.rowcount
