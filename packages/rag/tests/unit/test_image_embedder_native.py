# Test values are synthetic fixtures for software behavior.
"""
ImageEmbedder 百炼原生多模态协议单元测试。

覆盖：
    - DashScope 域名 / WorkspaceId / 显式 URL 下的原生协议启用
    - 原生请求体（model / input.contents / parameters.dimension）与响应解析
    - embed_text（文搜图查询侧）
    - OpenAI 兼容网关路径保持不变
    - 禁用/密钥缺失降级
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.embedder import Embedder, ImageEmbedder


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = "mock-response"

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def dashscope_env(monkeypatch):
    """DashScope 域名 + 密钥 + WorkspaceId。"""
    monkeypatch.setenv("ONEAPI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-native-test")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "ws-test-123")
    monkeypatch.setenv("IMAGE_EMBED_MODEL", "tongyi-embedding-vision-plus-2026-03-06")
    monkeypatch.setenv("IMAGE_EMBED_DIM", "1024")
    monkeypatch.delenv("DISABLE_IMAGE_EMBED", raising=False)


class TestNativeModeDetection:
    def test_dashscope_domain_enables_native(self, dashscope_env):
        emb = ImageEmbedder()
        assert emb._native is True
        assert emb._disabled is False

    def test_workspace_id_enables_native_on_custom_base(self, monkeypatch):
        monkeypatch.setenv("ONEAPI_BASE_URL", "https://models.example.com/v1")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-native-test")
        monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "ws-test-123")
        emb = ImageEmbedder()
        assert emb._native is True
        assert "ws-test-123.cn-beijing.maas.aliyuncs.com" in config_url()

    def test_non_dashscope_base_stays_openai(self, monkeypatch):
        monkeypatch.setenv("ONEAPI_BASE_URL", "https://models.example.com/v1")
        monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "")
        emb = ImageEmbedder()
        assert emb._native is False
        assert emb._disabled is False

    def test_disable_flag_disables_native(self, dashscope_env, monkeypatch):
        monkeypatch.setenv("DISABLE_IMAGE_EMBED", "1")
        emb = ImageEmbedder()
        assert emb._disabled is True


def config_url() -> str:
    from config import config
    return config.DASHSCOPE_MULTIMODAL_URL


class TestNativeCall:
    def test_payload_and_parse(self, dashscope_env):
        emb = ImageEmbedder()
        vec = [0.25] * 1024
        resp = _FakeResponse({
            "output": {"embeddings": [{"embedding": vec}]},
            "request_id": "req-1",
        })
        with patch("core.embedder.httpx.post", return_value=resp) as mock_post:
            out = emb._call_native([{"image": "data:image/jpeg;base64,abc"}])
        assert out == [vec]
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        body = kwargs["json"]
        assert body["model"] == "tongyi-embedding-vision-plus-2026-03-06"
        assert body["input"]["contents"] == [{"image": "data:image/jpeg;base64,abc"}]
        assert body["parameters"]["dimension"] == 1024
        assert kwargs["headers"]["Authorization"] == "Bearer sk-native-test"

    def test_dimension_mismatch_returns_none(self, dashscope_env):
        emb = ImageEmbedder()
        resp = _FakeResponse({
            "output": {"embeddings": [{"embedding": [0.1] * 768}]},
        })
        with patch("core.embedder.httpx.post", return_value=resp):
            out = emb._call_native([{"image": "data:image/jpeg;base64,abc"}])
        assert out == [None]

    def test_http_error_returns_none(self, dashscope_env):
        emb = ImageEmbedder()
        resp = _FakeResponse({"code": "InvalidParameter"}, status_code=400)
        with patch("core.embedder.httpx.post", return_value=resp):
            out = emb._call_native([{"image": "data:image/jpeg;base64,abc"}])
        assert out == [None]

    def test_embed_text_uses_native(self, dashscope_env):
        emb = ImageEmbedder()
        vec = [0.5] * 1024
        resp = _FakeResponse({
            "output": {"embeddings": [{"embedding": vec}]},
        })
        with patch("core.embedder.httpx.post", return_value=resp) as mock_post:
            out = emb.embed_text("TVS 钳位电压曲线")
        assert out == vec
        body = mock_post.call_args.kwargs["json"]
        assert body["input"]["contents"] == [{"text": "TVS 钳位电压曲线"}]

    def test_embed_text_requires_native(self):
        emb = ImageEmbedder()
        assert emb.embed_text("TVS 钳位电压曲线") is None

    def test_disabled_returns_none(self, dashscope_env, monkeypatch):
        monkeypatch.setenv("DISABLE_IMAGE_EMBED", "1")
        emb = ImageEmbedder()
        assert emb._call_native([{"image": "x"}]) == [None]
        assert emb.embed_text("query") is None


class TestOpenAICompatPath:
    def test_remote_path_parses_data(self, monkeypatch):
        monkeypatch.setenv("ONEAPI_BASE_URL", "https://models.example.com/v1")
        monkeypatch.setenv("ONEAPI_API_KEY", "mock-key")
        emb = ImageEmbedder()
        assert emb._native is False
        vec = [0.1] * 1024
        resp = _FakeResponse({"data": [{"embedding": vec}]})
        with patch("core.embedder._post_embeddings", return_value=resp):
            out = emb._call_remote(["data:image/jpeg;base64,abc"])
        assert out == [vec]


class TestEmbedderImageQuery:
    def test_embed_image_query_native(self, dashscope_env):
        embedder = Embedder()
        vec = [0.7] * 1024
        resp = _FakeResponse({
            "output": {"embeddings": [{"embedding": vec}]},
        })
        with patch("core.embedder.httpx.post", return_value=resp):
            out = embedder.embed_image_query("ESD 测试波形")
        assert out == vec

    def test_embed_image_query_non_native_returns_none(self):
        embedder = Embedder()
        assert embedder.embed_image_query("ESD 测试波形") is None

    def test_embed_image_query_empty(self, dashscope_env):
        embedder = Embedder()
        assert embedder.embed_image_query("") is None
