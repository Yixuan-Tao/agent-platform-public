# Test values are synthetic fixtures for software behavior.
"""
FastAPI 对比/可视化路由单元测试

覆盖 3 个端点：
    - POST /compare/emc          — 模拟 EmcCompareTool
    - POST /compare/component    — 模拟 ComponentCheckTool
    - POST /compare/visualize    — 模拟 EmcVisualizeTool

所有用例均通过 TestClient + unittest.mock 隔离外部依赖，
无需真实 RAG 管线或 LLM。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def client():
    """
    创建测试用 FastAPI 应用，注入 Mock query_pipeline 并注册路由。

    注意：在 import router 之前需要 mock 多个第三方模块，
    因为导入链会触发 chromadb / langdetect 等未安装包的导入错误。
    """
    # 预先填充 sys.modules，避免因第三方包未安装而导入失败
    _mock_chromadb = MagicMock()
    _mock_chromadb.PersistentClient = MagicMock
    _mock_chromadb.config = MagicMock()
    _mock_chromadb.config.Settings = MagicMock
    sys.modules["chromadb"] = _mock_chromadb
    sys.modules["chromadb.config"] = _mock_chromadb.config

    _mock_langdetect = MagicMock()
    _mock_langdetect.detect = MagicMock(return_value="zh")
    _mock_langdetect.DetectorFactory = MagicMock()
    sys.modules["langdetect"] = _mock_langdetect

    app = FastAPI()
    app.state.query_pipeline = MagicMock()
    app.state.query_pipeline.retriever = MagicMock()
    app.state.query_pipeline.generator = MagicMock()
    app.state.query_pipeline.embedder = MagicMock()
    app.state.query_pipeline.chroma = MagicMock()

    from api.compare_routes import router
    app.include_router(router)
    return TestClient(app)


# ============================================================
# POST /compare/emc — EMC 标准对比
# ============================================================

@patch("api.compare_routes.EmcCompareTool")
class TestCompareEmcRoute:
    """POST /compare/emc — 模拟 EmcCompareTool"""

    def test_success(self, mock_tool_class, client):
        """正常对比应返回 answer/compare_route 等字段。"""
        mock_tool_class.return_value.compare.return_value = {
            "answer": "对比分析结果：两者击穿电压不同",
            "doc_a": "doc_a.pdf",
            "doc_b": "doc_b.pdf",
            "params_a": {"VBR": "27V"},
            "params_b": {"VBR": "30V"},
            "compare_route": "full",
            "sources": [],
            "image_chunks": [],
        }
        resp = client.post("/compare/emc", json={
            "query": "对比击穿电压",
            "doc_a": "doc_a",
            "doc_b": "doc_b",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "对比分析结果：两者击穿电压不同"
        assert data["compare_route"] == "full"
        assert data["doc_a"]["name"] == "doc_a.pdf"
        assert data["doc_a"]["params"] == {"VBR": "27V"}
        assert data["doc_b"]["name"] == "doc_b.pdf"
        assert data["doc_b"]["params"] == {"VBR": "30V"}
        # 验证工具调用参数
        mock_tool_class.return_value.compare.assert_called_once_with(
            query="对比击穿电压", doc_a="doc_a", doc_b="doc_b", language="zh",
        )

    def test_with_language(self, mock_tool_class, client):
        """显式传入 language 应透传给工具。"""
        mock_tool_class.return_value.compare.return_value = {
            "answer": "EN comparison result",
            "doc_a": "a", "doc_b": "b",
            "params_a": {}, "params_b": {},
            "compare_route": "full", "sources": [], "image_chunks": [],
        }
        resp = client.post("/compare/emc", json={
            "query": "compare", "doc_a": "a", "doc_b": "b", "language": "en",
        })
        assert resp.status_code == 200
        mock_tool_class.return_value.compare.assert_called_once_with(
            query="compare", doc_a="a", doc_b="b", language="en",
        )

    def test_tool_error(self, mock_tool_class, client):
        """工具内部抛出异常应返回 500。"""
        mock_tool_class.return_value.compare.side_effect = Exception("工具内部错误")
        resp = client.post("/compare/emc", json={
            "query": "对比", "doc_a": "a", "doc_b": "b",
        })
        assert resp.status_code == 500
        assert "detail" in resp.json()


# ============================================================
# POST /compare/component — 元器件替换判断
# ============================================================

@patch("api.compare_routes.ComponentCheckTool")
class TestCompareComponentRoute:
    """POST /compare/component — 模拟 ComponentCheckTool"""

    def test_success(self, mock_tool_class, client):
        """正常替换判断应返回 conclusion/conclusion_detail 等字段。"""
        mock_tool_class.return_value.compare_with_template.return_value = {
            "conclusion": "可替换",
            "conclusion_detail": "所有参数匹配",
            "params_comparison": [],
            "failed_params": [],
        }
        resp = client.post("/compare/component", json={
            "doc_a": "doc_a",
            "doc_b": "doc_b",
            "params_template": {"electrical": {"VBR": {"unit": "V"}}},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["conclusion"] == "可替换"
        assert data["conclusion_detail"] == "所有参数匹配"
        assert data["params_comparison"] == []
        assert data["failed_params"] == []

    def test_with_strict_mode_true(self, mock_tool_class, client):
        """strict_mode=True 应透传给工具。"""
        mock_tool_class.return_value.compare_with_template.return_value = {
            "conclusion": "有条件替换",
            "conclusion_detail": "部分参数不匹配",
            "params_comparison": [
                {"param": "VBR", "category": "electrical",
                 "doc_a_value": "27V", "doc_b_value": "30V",
                 "match": False, "within_tolerance": True, "note": ""},
            ],
            "failed_params": [],
        }
        resp = client.post("/compare/component", json={
            "doc_a": "doc_a",
            "doc_b": "doc_b",
            "params_template": {"electrical": {"VBR": {"unit": "V", "tolerance": "10%"}}},
            "strict_mode": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["conclusion"] == "有条件替换"
        assert len(data["params_comparison"]) == 1
        assert data["params_comparison"][0]["match"] is False

    def test_with_strict_mode_false(self, mock_tool_class, client):
        """strict_mode=False 时 strict_mode 参数应正确传递。"""
        mock_tool_class.return_value.compare_with_template.return_value = {
            "conclusion": "可替换", "conclusion_detail": "宽松模式",
            "params_comparison": [], "failed_params": [],
        }
        resp = client.post("/compare/component", json={
            "doc_a": "doc_a", "doc_b": "doc_b",
            "params_template": {"electrical": {"VBR": {"unit": "V"}}},
            "strict_mode": False,
        })
        assert resp.status_code == 200
        assert resp.json()["conclusion"] == "可替换"
        # 验证工具收到 strict_mode=False
        _, kwargs = mock_tool_class.return_value.compare_with_template.call_args
        assert kwargs["strict_mode"] is False

    def test_tool_error(self, mock_tool_class, client):
        """工具内部抛出异常应返回 500。"""
        mock_tool_class.return_value.compare_with_template.side_effect = (
            Exception("文档解析失败")
        )
        resp = client.post("/compare/component", json={
            "doc_a": "doc_a", "doc_b": "doc_b",
            "params_template": {"electrical": {}},
        })
        assert resp.status_code == 500
        assert "detail" in resp.json()


# ============================================================
# POST /compare/visualize — EMC 表格可视化
# ============================================================

@patch("api.compare_routes.EmcVisualizeTool")
class TestVisualizeRoute:
    """POST /compare/visualize — 模拟 EmcVisualizeTool"""

    def test_single_doc(self, mock_tool_class, client):
        """单文档可视化应调用 tool.visualize()。"""
        mock_tool_class.return_value.visualize.return_value = {
            "type": "Type_A",
            "html": "<html><body><div id='chart'>chart</div></body></html>",
            "datasets": [{"name": "RE", "values": []}],
            "sources": [{"doc_id": "doc_123", "table_name": "Table1"}],
        }
        resp = client.post("/compare/visualize", json={
            "query": "辐射发射限值",
            "doc_id": "doc_123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "html_bundle" in data
        assert "chart" in data["html_bundle"]
        assert data["table_type"] == "Type_A"
        assert len(data["datasets"]) == 1
        # 此时走 visualize() 分支（非双文档叠加）
        mock_tool_class.return_value.visualize.assert_called_once_with(
            query="辐射发射限值", doc_id="doc_123", top_k=5,
        )
        mock_tool_class.return_value.visualize_overlay.assert_not_called()

    def test_single_doc_with_table_type(self, mock_tool_class, client):
        """单文档可视化时 table_type 应透传（但 visualize 当前签名不含 table_type）。"""
        mock_tool_class.return_value.visualize.return_value = {
            "type": "CE", "html": "<html>CE</html>",
            "datasets": [], "sources": [],
        }
        resp = client.post("/compare/visualize", json={
            "query": "传导发射", "doc_id": "doc_123", "table_type": "conducted",
        })
        assert resp.status_code == 200
        assert resp.json()["table_type"] == "CE"

    def test_overlay_two_docs(self, mock_tool_class, client):
        """双文档叠加对比应调用 tool.visualize_overlay()。"""
        mock_tool_class.return_value.visualize_overlay.return_value = {
            "type": "Type_A",
            "html": "<html><div>overlay chart</div></html>",
            "datasets": [{"name": "RE_A"}, {"name": "RE_B"}],
            "doc_a_name": "Standard_A",
            "doc_b_name": "Standard_B",
        }
        resp = client.post("/compare/visualize", json={
            "query": "对比限值",
            "doc_a": "doc_a",
            "doc_b": "doc_b",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "html_bundle" in data
        assert "overlay chart" in data["html_bundle"]
        mock_tool_class.return_value.visualize_overlay.assert_called_once_with(
            doc_a="doc_a", doc_b="doc_b",
            query="对比限值", table_type=None, top_k=10,
        )
        mock_tool_class.return_value.visualize.assert_not_called()

    def test_overlay_with_table_type(self, mock_tool_class, client):
        """双文档叠加时 table_type 应透传给 visualize_overlay。"""
        mock_tool_class.return_value.visualize_overlay.return_value = {
            "type": "RE", "html": "<html>RE</html>",
            "datasets": [], "sources": [],
        }
        resp = client.post("/compare/visualize", json={
            "query": "辐射", "doc_a": "a", "doc_b": "b", "table_type": "radiated",
        })
        assert resp.status_code == 200
        _, kwargs = mock_tool_class.return_value.visualize_overlay.call_args
        assert kwargs["table_type"] == "radiated"
        assert kwargs["top_k"] == 10  # top_k * 2

    def test_visualize_result_with_error(self, mock_tool_class, client):
        """工具返回 error 字段时应触发 500。"""
        mock_tool_class.return_value.visualize.return_value = {"error": "无数据"}
        resp = client.post("/compare/visualize", json={
            "query": "限值", "doc_id": "missing",
        })
        assert resp.status_code == 500
        assert resp.json()["detail"] == "无数据"

    def test_overlay_result_with_error(self, mock_tool_class, client):
        """双文档叠加时工具返回 error 字段也应触发 500。"""
        mock_tool_class.return_value.visualize_overlay.return_value = {
            "error": "文档不存在",
        }
        resp = client.post("/compare/visualize", json={
            "query": "对比", "doc_a": "a", "doc_b": "b",
        })
        assert resp.status_code == 500
        assert resp.json()["detail"] == "文档不存在"

    def test_tool_raises_exception(self, mock_tool_class, client):
        """工具抛出异常应返回 500。"""
        mock_tool_class.return_value.visualize.side_effect = RuntimeError("LLM 调用失败")
        resp = client.post("/compare/visualize", json={
            "query": "限值", "doc_id": "doc_123",
        })
        assert resp.status_code == 500
        assert "detail" in resp.json()

    def test_html_fallback_to_bundle(self, mock_tool_class, client):
        """若返回 dict 不含 html，应回退到 bundle 字段。"""
        mock_tool_class.return_value.visualize.return_value = {
            "type": "Type_A",
            "bundle": "<html>bundle fallback</html>",
            "datasets": [], "sources": [],
        }
        resp = client.post("/compare/visualize", json={
            "query": "限值", "doc_id": "doc_123",
        })
        assert resp.status_code == 200
        assert "bundle fallback" in resp.json()["html_bundle"]

    def test_html_is_dict_fallback(self, mock_tool_class, client):
        """若 html 字段是 dict，应 JSON 序列化为字符串。"""
        mock_tool_class.return_value.visualize.return_value = {
            "type": "Type_A",
            "html": {"content": "<p>nested</p>"},
            "datasets": [], "sources": [],
        }
        resp = client.post("/compare/visualize", json={
            "query": "限值", "doc_id": "doc_123",
        })
        assert resp.status_code == 200
        # 应为 JSON 字符串化的 dict
        assert '"content": "<p>nested</p>"' in resp.json()["html_bundle"]
