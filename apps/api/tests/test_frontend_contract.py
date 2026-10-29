# Test values are synthetic fixtures for software behavior.
"""前后端契约测试：锁住 apps/api ↔ apps/web 的接口约定。

以静态读取方式断言：
    - 后端路由路径/方法与前端调用一致（URL、HTTP 方法）
    - 前端请求头（X-API-Key）与 SSE 事件类型（token/agent_switch/session/error）
    - 请求/响应字段名与前端解析一致
"""

from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parents[2] / "web" / "src"


def _api_sources() -> str:
    return "\n".join(
        p.read_text(encoding="utf-8")
        for p in list((API_DIR / "app" / "routes").glob("*.py"))
        + [API_DIR / "app" / "middleware.py"]
    )


def _orchestrator_sources() -> str:
    return "\n".join(
        p.read_text(encoding="utf-8")
        for p in (API_DIR.parents[1] / "packages" / "orchestrator" / "src").rglob("*.py")
        if p.suffix == ".py"
    )


def _security_source() -> str:
    """身份解析集中在 core.security，前端凭据约定以该实现为准。"""
    path = (
        API_DIR.parents[1]
        / "packages"
        / "core"
        / "src"
        / "agent_platform"
        / "core"
        / "security.py"
    )
    return path.read_text(encoding="utf-8")


def _chat_source() -> str:
    return (WEB_DIR / "components" / "Chat.tsx").read_text(encoding="utf-8")


def test_web_chat_calls_existing_api_endpoints():
    """前端引用的每个端点都必须真实存在于后端路由文件。"""
    api_src = _api_sources()
    chat_src = _chat_source()
    for endpoint in (
        "/api/v1/agents",
        "/api/v1/sessions",
        "/api/v1/chat",
        "/api/v1/feedback",
    ):
        assert endpoint in api_src, f"后端缺少前端依赖的端点 {endpoint}"
        assert endpoint in chat_src, f"前端未引用端点 {endpoint}"


def test_chat_request_body_fields_match_backend_model():
    chat_src = _chat_source()
    api_src = _api_sources()
    # 前端发送 body：session_id / message / agent_hint
    for field in ("session_id", "message", "agent_hint"):
        assert f'"{field}"' in chat_src or field in chat_src, f"前端请求体缺少 {field}"
    # 后端 ChatRequest 模型包含相同字段
    models = (API_DIR.parents[1] / "packages" / "core" / "src" / "agent_platform" / "core" / "models.py")
    assert "session_id" in models.read_text(encoding="utf-8")
    assert "agent_hint" in models.read_text(encoding="utf-8")


def test_web_uses_x_api_key_header_for_all_calls():
    chat_src = _chat_source()
    security_src = _security_source()
    assert '"X-API-Key"' in chat_src
    assert "X-API-Key" in security_src
    # 会话令牌同样必须被前端支持，否则登录后仍会被网关拦截
    assert "Authorization" in chat_src


def test_sse_event_types_handled_by_frontend_exist_in_backend():
    chat_src = _chat_source()
    backend_src = _api_sources() + "\n" + _orchestrator_sources()
    for event_type in ("token", "agent_switch", "session", "error"):
        assert f'evt.type === "{event_type}"' in chat_src, f"前端未处理 SSE 事件 {event_type}"
        assert f'"type": "{event_type}"' in backend_src, f"后端未产出 SSE 事件 {event_type}"


def test_agents_list_shape_matches_frontend():
    api_src = _api_sources()
    assert '"agents"' in api_src
    assert '"count"' in api_src
    # 前端读取 data.agents 并 map name/description
    chat_src = _chat_source()
    assert "data.agents" in chat_src
    assert "a.name" in chat_src
    assert "a.description" in chat_src


def test_sessions_list_shape_matches_frontend():
    api_src = _api_sources()
    assert "session_id" in api_src
    assert "title" in api_src
    assert "updated_at" in api_src
    chat_src = _chat_source()
    assert "s.session_id" in chat_src
    assert "s.title" in chat_src
    assert "updated_at" in chat_src


def test_feedback_body_fields_match_backend():
    chat_src = _chat_source()
    api_src = _api_sources()
    for field in ("session_id", "message_id", "feedback_type", "correction"):
        assert field in chat_src, f"前端反馈体缺少 {field}"
        assert field in api_src, f"后端反馈模型缺少 {field}"


def test_write_operations_are_protected():
    """写操作（文档摄入/删除、反馈提交/撤销、账号管理）必须带登录依赖。"""
    api_src = _api_sources()
    documents = (API_DIR / "app" / "routes" / "documents.py").read_text(encoding="utf-8")
    feedback = (API_DIR / "app" / "routes" / "feedback.py").read_text(encoding="utf-8")
    auth = (API_DIR / "app" / "routes" / "auth.py").read_text(encoding="utf-8")
    assert "require_login" in api_src
    assert "require_admin" in api_src
    assert documents.count("Depends(require_login)") >= 2
    assert feedback.count("Depends(require_login)") >= 2
    assert "Depends(require_admin)" in auth


def test_auth_endpoints_exist_for_login_flow():
    api_src = _api_sources()
    for endpoint in ("/login", "/me", "/logout", "/users"):
        assert endpoint in api_src, f"后端缺少认证端点 {endpoint}"


def test_web_and_api_agree_on_auth_path_prefix():
    # 前端将来接入登录时以 /api/v1/auth/login 为准
    api_src = _api_sources()
    assert '"/api/v1/auth"' in api_src
