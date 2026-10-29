"""平台本地账号、会话、角色与审计存储（对齐 intern_rag 的 LAN 安全能力）。

设计：
    - ``AccessStore``：SQLite 持久化的用户/密码哈希（PBKDF2）/角色（viewer/admin）/会话令牌
    - ``AuditStore``：SQLite 持久化的审计事件（用户归属、来源 IP、动作、状态）
    - FastAPI 依赖：``current_identity`` / ``require_login`` / ``require_admin``

凭据语义（2026-09-14 收紧）：
    - ``X-API-Key``：服务/自动化凭据，角色 ``service``。可调用业务接口，
      但不能访问账号管理等管理端点。
    - ``ADMIN_API_KEY``：可轮换的自动化管理员凭据（Bearer 形式）。
    - ``/api/v1/auth/login`` 签发的 Bearer 会话令牌：交互用户身份，
      角色取账号自身角色（viewer/admin）。

审计与归属统一走 :func:`resolve_identity`，避免中间件与路由依赖各写一套判断。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional

from fastapi import Header, HTTPException, Request

from agent_platform.core.config import get_settings


class AuthorizationError(Exception):
    pass


class AccessStore:
    """本地账号与会话存储（SQLite）。"""

    def __init__(self, path: Path, session_ttl_seconds: int = 8 * 3600):
        self.path = path
        self.session_ttl_seconds = session_ttl_seconds
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, username TEXT NOT NULL, expires_at REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
            """)

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    @staticmethod
    def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
        salt = salt or secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000).hex()
        return f"{salt.hex()}${digest}"

    def create_user(self, username: str, password: str, role: str) -> None:
        if role not in {"viewer", "admin"} or not username or len(password) < 8:
            raise ValueError("无效账户信息")
        with self._db() as db:
            db.execute("INSERT INTO users VALUES (?, ?, ?)", (username, self._hash_password(password), role))

    def authenticate(self, username: str, password: str) -> str:
        with self._db() as db:
            row = db.execute("SELECT password_hash FROM users WHERE username=?", (username,)).fetchone()
        if not row:
            raise AuthorizationError("用户名或密码错误")
        salt_hex, expected = row[0].split("$", 1)
        actual = self._hash_password(password, bytes.fromhex(salt_hex)).split("$", 1)[1]
        if not hmac.compare_digest(actual, expected):
            raise AuthorizationError("用户名或密码错误")
        token = secrets.token_urlsafe(32)
        with self._db() as db:
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, 0)",
                (self._token_hash(token), username, time.time() + self.session_ttl_seconds),
            )
        return token

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def revoke(self, token: str) -> None:
        with self._db() as db:
            db.execute("UPDATE sessions SET revoked=1 WHERE token_hash=?", (self._token_hash(token),))

    def list_users(self) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT username, role FROM users ORDER BY username").fetchall()
        return [{"username": row[0], "role": row[1]} for row in rows]

    def delete_user(self, username: str) -> None:
        with self._db() as db:
            row = db.execute("SELECT role FROM users WHERE username=?", (username,)).fetchone()
            if row is None:
                raise AuthorizationError("账号不存在")
            if row[0] == "admin":
                admins = db.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
                if admins <= 1:
                    raise AuthorizationError("不能删除最后一个管理员")
            db.execute("DELETE FROM sessions WHERE username=?", (username,))
            db.execute("DELETE FROM users WHERE username=?", (username,))

    def identity(self, token: str, required: str = "viewer") -> dict:
        with self._db() as db:
            row = db.execute(
                "SELECT u.username,u.role,s.expires_at,s.revoked FROM sessions s "
                "JOIN users u ON u.username=s.username WHERE s.token_hash=?",
                (self._token_hash(token),),
            ).fetchone()
        if not row or row[3] or row[2] <= time.time() or (required == "admin" and row[1] != "admin"):
            raise AuthorizationError("无权限或会话已失效")
        return {"username": row[0], "role": row[1]}


class AuditStore:
    """本地脱敏审计记录（SQLite）。"""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS audit_events (created_at REAL, username TEXT, "
                "source_ip TEXT, action TEXT, status TEXT, run_id TEXT, pack_version TEXT)"
            )

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def record(
        self, *, username: str, source_ip: str, action: str, status: str,
        run_id: str = "", pack_version: str = "",
    ) -> None:
        with self._db() as db:
            db.execute(
                "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), username, source_ip, action, status, run_id, pack_version),
            )

    def list_recent(self, limit: int = 100) -> list[dict]:
        with self._db() as db:
            rows = db.execute(
                "SELECT created_at,username,source_ip,action,status,run_id,pack_version "
                "FROM audit_events ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
        keys = ("created_at", "username", "source_ip", "action", "status", "run_id", "pack_version")
        return [dict(zip(keys, row)) for row in rows]


# ============================================================
# FastAPI 依赖
# ============================================================


def _bearer(authorization: str = Header(default="")) -> str:
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="缺少 Bearer 会话令牌")
    return authorization[len(prefix):]


def automation_identity(token: str, admin_api_key: str) -> dict | None:
    if admin_api_key and hmac.compare_digest(token, admin_api_key):
        return {"username": "automation", "role": "admin", "auth": "admin_api_key"}
    return None


def resolve_identity(request: Request) -> dict | None:
    """解析请求身份；无法识别时返回 ``None``（调用方决定 401/403 语义）。

    优先级：``X-API-Key``（服务）→ ``ADMIN_API_KEY`` → 会话令牌。
    中间件解析成功后会写入 ``request.state.identity``，路由依赖直接复用，
    保证鉴权与审计看到同一份身份。
    """
    settings = getattr(request.app.state, "settings", None) or get_settings()
    api_key = request.headers.get("X-API-Key", "")
    if api_key and settings.api_key and hmac.compare_digest(api_key, settings.api_key):
        return {"username": "api-key", "role": "service", "auth": "api_key"}

    authorization = request.headers.get("authorization", "")
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    token = authorization[len(prefix):]
    identity = automation_identity(token, settings.admin_api_key)
    if identity:
        return identity
    store = getattr(request.app.state, "access_store", None)
    if store is None:
        return None
    try:
        return {**store.identity(token), "auth": "session"}
    except AuthorizationError:
        return None


def _identity_or_none(request: Request) -> dict | None:
    return getattr(request.state, "identity", None) or resolve_identity(request)


def current_identity(request: Request) -> dict:
    identity = _identity_or_none(request)
    if identity is None:
        raise HTTPException(status_code=401, detail="缺少或无效的身份凭据")
    return identity


def require_login(request: Request) -> dict:
    """业务/写操作身份校验：接受服务 API Key、ADMIN_API_KEY 或用户会话。"""
    identity = _identity_or_none(request)
    if identity is None:
        raise HTTPException(status_code=401, detail="缺少或无效的身份凭据")
    return identity


def require_admin(request: Request) -> dict:
    """管理操作校验：仅接受管理员会话或 ADMIN_API_KEY。"""
    identity = _identity_or_none(request)
    if identity is None:
        raise HTTPException(status_code=401, detail="缺少或无效的身份凭据")
    if identity.get("role") != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return identity
