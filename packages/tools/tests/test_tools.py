# Test values are synthetic fixtures for software behavior.
"""tools 包测试。"""

from __future__ import annotations

import socket

import pytest

from agent_platform.tools.base import BaseTool, ToolRegistry, create_tool
from agent_platform.tools.builtin.rag_search import RagSearchTool
from agent_platform.tools.builtin.web_collect import (
    RssFetchTool,
    UnsafeUrlError,
    WebCollectTool,
    ensure_public_url,
)
from agent_platform.tools.sandbox import SandboxedPlugin, SandboxError


class EchoTool(BaseTool):
    name = "echo"
    description = "回显输入"
    args_schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, text: str, **kwargs):
        return f"echo:{text}"


async def test_registry_register_lookup():
    reg = ToolRegistry()
    reg.register(EchoTool())
    assert reg.names() == ["echo"]
    assert await reg.get("echo").execute(text="hi") == "echo:hi"


def test_registry_duplicate_raises():
    reg = ToolRegistry()
    reg.register(EchoTool())
    with pytest.raises(ValueError):
        reg.register(EchoTool())


async def test_create_tool_decorator():
    async def add(a: int, b: int = 0):
        return a + b

    tool = create_tool("add", "加法", add)
    assert tool.args_schema["required"] == ["a"]
    assert await tool.execute(a=2, b=3) == "5"


async def test_rag_search_failure_returns_friendly(monkeypatch):
    """RAG 服务不可达时应返回说明文本而不是抛异常。"""
    tool = RagSearchTool(base_url="http://127.0.0.1:1")
    result = await tool.execute(query="test")
    assert "失败" in result or "检索" in result


async def test_web_collect_parses_html(monkeypatch):
    class FakeResp:
        status_code = 200
        text = "<html><body><h1>标题</h1><p>正文内容</p></body></html>"
        headers: dict = {}
        is_redirect = False

        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            return FakeResp()

    monkeypatch.setattr("httpx.AsyncClient", FakeClient)
    # 避免单测依赖真实 DNS：显式返回公网地址
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )
    tool = WebCollectTool()
    result = await tool.execute(url="https://example.com")
    assert "正文内容" in result


# ------------------------------------------------------------
# SSRF 防护：内网/保留地址与非法协议必须拒绝
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/api/v1/chat",
        "http://localhost/admin",
        "http://10.0.0.5/internal",
        "http://192.0.2.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "file:///etc/passwd",
        "ftp://example.com/x",
    ],
)
def test_ensure_public_url_rejects_internal_targets(url):
    with pytest.raises(UnsafeUrlError):
        ensure_public_url(url)


def test_ensure_public_url_accepts_public_ip():
    assert ensure_public_url("https://93.184.216.34/x") == "https://93.184.216.34/x"


def test_ensure_public_url_rejects_unresolvable_host(monkeypatch):
    def boom(host, port, *a, **k):
        raise socket.gaierror("name resolution failed")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(UnsafeUrlError):
        ensure_public_url("https://does-not-resolve.invalid/x")


async def test_web_collect_refuses_internal_url_without_fetching(monkeypatch):
    called = False

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            nonlocal called
            called = True
            raise AssertionError("不应发起对内网地址的请求")

    monkeypatch.setattr("httpx.AsyncClient", FakeClient)
    tool = WebCollectTool()
    result = await tool.execute(url="http://169.254.169.254/latest/meta-data/")
    assert "拒绝" in result
    assert called is False


async def test_rss_fetch_refuses_internal_url(monkeypatch):
    tool = RssFetchTool()
    result = await tool.execute(feed_url="http://127.0.0.1:8010/feed")
    assert "拒绝" in result


async def test_web_collect_rejects_redirect_to_internal(monkeypatch):
    """公网地址跳转到内网同样要拦截。"""

    class FakeResp:
        status_code = 302
        text = ""
        headers = {"location": "http://127.0.0.1:8000/admin"}
        is_redirect = True

        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return FakeResp()

    monkeypatch.setattr("httpx.AsyncClient", FakeClient)
    tool = WebCollectTool()
    result = await tool.execute(url="https://93.184.216.34/redirect")
    assert "拒绝" in result


async def test_sandbox_runs_python():
    code = "import json,sys; data=json.load(sys.stdin); print(json.dumps({'ok':True,'echo':data['x']}))"
    plugin = SandboxedPlugin(["python", "-c", code])
    result = await plugin.run({"x": 42})
    assert result == {"ok": True, "echo": 42}


async def test_sandbox_timeout():
    plugin = SandboxedPlugin(["python", "-c", "import time; time.sleep(10)"], timeout=0.2)
    with pytest.raises(SandboxError):
        await plugin.run({})
