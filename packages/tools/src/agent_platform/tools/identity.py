"""请求级执行身份：把接入层认证得到的角色传到工具调用。

历史实现把工具调用角色写死为 ``"user"``，导致角色 ACL（``required_role``）
形同虚设。这里用 contextvar 在请求上下文中传递角色，而不必修改所有
Agent 构建函数签名。
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_principal_role: ContextVar[str] = ContextVar("agent_platform_principal_role", default="user")


def set_principal_role(role: str) -> Token:
    """设置当前请求的工具调用角色，返回可用于恢复的 token。"""
    return _principal_role.set(role or "user")


def reset_principal_role(token: Token) -> None:
    _principal_role.reset(token)


def current_principal_role() -> str:
    return _principal_role.get()


__all__ = [
    "current_principal_role",
    "reset_principal_role",
    "set_principal_role",
]
