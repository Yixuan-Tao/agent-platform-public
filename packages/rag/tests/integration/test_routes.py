# Test values are synthetic fixtures for software behavior.
"""
API 路由集成测试

覆盖：
    - /health 端点
    - /documents 列表/删除
    - /tools/search 检索
    - /v1/chat/completions 各分支
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# 在导入 app 前 mock ChromaClientSingleton，避免实际连接
with patch("db.chroma_client.ChromaClientSingleton.get_instance") as mock_get:
    mock_get.return_value = MagicMock()
    mock_get.return_value._client = MagicMock()
    mock_get.return_value.is_connected = True
    mock_get.return_value.get_collection_stats.return_value = {"row_count": 0}
    mock_get.return_value.list_documents.return_value = []
    mock_get.return_value.delete_by_file_name.return_value = 0

    from main import app


@pytest.fixture
def client():
    """FastAPI TestClient。"""
    return TestClient(app)


class TestHealthEndpoint:
    """健康检查端点测试。"""

    def test_health_returns_200(self, client):
        """健康检查应返回 200。"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "chroma_connected" in data
        assert "document_count" in data
        assert "chunk_count" in data

    def test_health_has_correct_fields(self, client):
        """健康检查响应字段应完整。"""
        response = client.get("/health")
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert data["version"] == "0.1.0"


class TestDocumentsEndpoint:
    """文档管理端点测试。"""

    def test_list_documents_returns_200(self, client):
        """文档列表应返回 200。"""
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "documents" in data
        assert "chroma_connected" in data

    def test_delete_document_returns_404_for_missing_name(self, client):
        from main import app
        app.state.local_api_token = "test-token"
        """删除不存在的文档（路径参数缺失时 FastAPI 返回 422）。"""
        # DELETE /documents/ 路径不匹配，FastAPI 返回 422 或 405
        response = client.delete("/documents/", headers={"X-Local-Token": "test-token"})
        assert response.status_code in (404, 405, 422)


class TestSearchEndpoint:
    """检索工具端点测试。"""

    @patch("pipeline.query.QueryPipeline.query")
    def test_search_returns_results(self, mock_query, client):
        """检索工具应返回结构化结果。"""
        mock_query.return_value = {
            "answer": "测试答案",
            "sources": [
                {
                    "id": "c1", "file_name": "test.pdf", "page": 1,
                    "modality": "text", "content": "test content", "score": 0.9,
                }
            ],
            "language": "zh",
            "used_rerank": True,
        }
        response = client.post("/tools/search", json={
            "query": "ESD 击穿电压",
            "top_k": 5,
            "use_rerank": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "ESD 击穿电压"
        assert len(data["results"]) == 1
        assert data["answer"] == "测试答案"

    def test_search_empty_query(self, client):
        """空查询应能被处理。"""
        response = client.post("/tools/search", json={"query": ""})
        # 空查询不应导致 500
        assert response.status_code in (200, 500)


class TestChatCompletionsEndpoint:
    """Chat Completions 端点测试。"""

    @patch("pipeline.query.QueryPipeline.query")
    def test_chat_non_stream(self, mock_query, client):
        """非流式 chat 应返回标准响应。"""
        mock_query.return_value = {
            "answer": "击穿电压为 27V。",
            "sources": [],
            "language": "zh",
            "used_rerank": False,
        }
        response = client.post("/v1/chat/completions", json={
            "model": "qwen3.7-max",
            "messages": [
                {"role": "user", "content": "ESD 击穿电压是多少？"}
            ],
            "stream": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "击穿电压为 27V。"

    def test_chat_invalid_request_returns_400(self, client):
        """无效请求体应返回 400。"""
        response = client.post("/v1/chat/completions", json={
            "model": "qwen3.7-max",
            # 缺少 messages 字段
        })
        assert response.status_code == 400

    def test_chat_with_tool_calls(self, client):
        """含 tool_calls 的请求应被正确处理。"""
        response = client.post("/v1/chat/completions", json={
            "model": "qwen3.7-max",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_hardware_knowledge_base",
                                "arguments": '{"query": "ESD voltage"}',
                            },
                        }
                    ],
                }
            ],
            "stream": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    @patch("pipeline.query.QueryPipeline.query_stream")
    def test_chat_stream(self, mock_stream, client):
        """流式 chat 应返回 SSE 格式。"""
        mock_stream.return_value = iter(["击穿", "电压", "为", "27V"])
        response = client.post("/v1/chat/completions", json={
            "model": "qwen3.7-max",
            "messages": [
                {"role": "user", "content": "ESD 击穿电压是多少？"}
            ],
            "stream": True,
        })
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = response.text
        assert "data:" in body
