"""局域网服务的本地账户、短期会话与角色校验。"""
import hashlib
import hmac
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional


class AuthorizationError(Exception):
    pass


class AccessStore:
    def __init__(self, path: Path, session_ttl_seconds: int = 8 * 3600):
        self.path, self.session_ttl_seconds = path, session_ttl_seconds
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, username TEXT NOT NULL, expires_at REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
            """)

    def _db(self): return sqlite3.connect(self.path)
    @staticmethod
    def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
        salt = salt or secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000).hex()
        return f"{salt.hex()}${digest}"
    def create_user(self, username: str, password: str, role: str) -> None:
        if role not in {"viewer", "admin"} or not username or len(password) < 8: raise ValueError("无效账户信息")
        with self._db() as db: db.execute("INSERT INTO users VALUES (?, ?, ?)", (username, self._hash_password(password), role))
    def authenticate(self, username: str, password: str) -> str:
        with self._db() as db: row = db.execute("SELECT password_hash FROM users WHERE username=?", (username,)).fetchone()
        if not row: raise AuthorizationError("用户名或密码错误")
        salt_hex, expected = row[0].split("$", 1)
        actual = self._hash_password(password, bytes.fromhex(salt_hex)).split("$", 1)[1]
        if not hmac.compare_digest(actual, expected): raise AuthorizationError("用户名或密码错误")
        token = secrets.token_urlsafe(32)
        with self._db() as db: db.execute("INSERT INTO sessions VALUES (?, ?, ?, 0)", (self._token_hash(token), username, time.time() + self.session_ttl_seconds))
        return token
    @staticmethod
    def _token_hash(token: str) -> str: return hashlib.sha256(token.encode()).hexdigest()
    def revoke(self, token: str) -> None:
        with self._db() as db: db.execute("UPDATE sessions SET revoked=1 WHERE token_hash=?", (self._token_hash(token),))

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
    def require_role(self, token: str, required: str) -> str:
        return self.identity(token, required)["username"]

    def identity(self, token: str, required: str = "viewer") -> dict:
        with self._db() as db:
            row = db.execute("SELECT u.username,u.role,s.expires_at,s.revoked FROM sessions s JOIN users u ON u.username=s.username WHERE s.token_hash=?", (self._token_hash(token),)).fetchone()
        if not row or row[3] or row[2] <= time.time() or (required == "admin" and row[1] != "admin"): raise AuthorizationError("无权限或会话已失效")
        return {"username": row[0], "role": row[1]}
