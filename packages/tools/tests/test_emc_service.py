# Test values are synthetic fixtures for software behavior.
"""平台 EMC 工具的服务适配测试：契约、格式化与降级。"""

from __future__ import annotations

import json

import httpx
import pytest

from agent_platform.core.config import Settings
from agent_platform.tools.builtin.emc_service import (
    EmcCompareService,
    EmcCompareTool,
    EmcComponentCheckTool,
    EmcServiceError,
    EmcVisualizeTool,
)


class _FakeService(EmcCompareService):
    DOCS = [
        {"file_name": "CISPR25_2021.pdf", "doc_id": "aaaa1111bbbb", "chunk_count": 40},
        {"file_name": "ISO_11452-4_2020.pdf", "doc_id": "cccc2222dddd", "chunk_count": 30},
        {"file_name": "EMC 基线.xlsx", "doc_id": "eeee3333ffff", "chunk_count": 5},
    ]

    def __init__(self, payload=None, error=None, docs=None):
        super().__init__(Settings(_env_file=None))
        self.payload = payload
        self.error = error
        self._docs = self.DOCS if docs is None else docs
        self.calls: list[tuple[str, dict]] = []

    async def list_documents(self):
        return list(self._docs)

    async def post(self, path, payload):
        self.calls.append((path, payload))
        if self.error is not None:
            raise self.error
        return self.payload


async def test_compare_tool_formats_matrix_and_citations():
    service = _FakeService(
        payload={
            "answer": "两份标准在辐射发射限值上不同。",
            "comparison_matrix": [
                {"field": "频段", "a": ["30-230 MHz"], "b": ["30-230 MHz"], "status": "same"},
                {"field": "限值", "a": ["40 dBuV/m"], "b": ["30 dBuV/m"], "status": "different"},
            ],
            "citations": [
                {"file_name": "CISPR25.pdf", "section": "5.1", "page": 12},
                {"file_name": "CISPR25.pdf", "section": "5.1", "page": 12},
            ],
            "answer_status": "ok",
        }
    )
    tool = EmcCompareTool(service=service)
    result = await tool.execute(query="对比限值", doc_a="CISPR25", doc_b="ISO_11452-4")

    assert service.calls[0][0] == "/compare/emc"
    assert service.calls[0][1]["comparison_mode"] == "standard"
    # 文件名关键词被解析为 doc_id 后才发送给 RAG
    assert service.calls[0][1]["doc_a"] == "aaaa1111bbbb"
    assert service.calls[0][1]["doc_b"] == "cccc2222dddd"
    assert "两份标准在辐射发射限值上不同。" in result
    assert "| 限值 | 40 dBuV/m | 30 dBuV/m | 存在差异 |" in result
    # 重复引用去重
    assert result.count("CISPR25.pdf / 5.1 / 12") == 1


async def test_compare_tool_surfaces_insufficient_evidence_status():
    service = _FakeService(
        payload={"answer": "资料不足", "comparison_matrix": [], "answer_status": "insufficient_evidence"}
    )
    result = await EmcCompareTool(service=service).execute(
        query="q", doc_a="CISPR25", doc_b="ISO_11452-4"
    )
    assert "答案状态：insufficient_evidence" in result


async def test_compare_tool_degrades_with_clear_message():
    service = _FakeService(error=EmcServiceError("RAG 对比服务不可达: refused"))
    result = await EmcCompareTool(service=service).execute(
        query="q", doc_a="CISPR25", doc_b="ISO_11452-4"
    )
    assert "EMC 对比失败" in result
    assert "RAG 对比服务不可达" in result


async def test_component_check_parses_template_and_reports_failures():
    service = _FakeService(
        payload={
            "conclusion": "不建议替换",
            "conclusion_detail": "钳位电压超出容差",
            "params_comparison": [
                {
                    "param": "VBR",
                    "doc_a_value": "27V",
                    "doc_b_value": "24V",
                    "match": False,
                    "note": "低于下限",
                }
            ],
            "failed_params": ["IPP"],
            "citations": [{"file_name": "a.pdf", "page": 3}],
            "answer_status": "ok",
        }
    )
    tool = EmcComponentCheckTool(service=service)
    result = await tool.execute(
        doc_a="CISPR25",
        doc_b="ISO_11452-4",
        params_template=json.dumps({"VBR": {"min": 25, "max": 30}}),
    )
    assert service.calls[0][0] == "/compare/component"
    assert service.calls[0][1]["params_template"] == {"VBR": {"min": 25, "max": 30}}
    assert "结论：不建议替换" in result
    assert "[不通过] VBR：A=27V / B=24V（低于下限）" in result
    assert "未在文档中找到的参数：IPP" in result


async def test_component_check_rejects_bad_json_without_calling_service():
    service = _FakeService(payload={})
    result = await EmcComponentCheckTool(service=service).execute(
        doc_a="CISPR25", doc_b="ISO_11452-4", params_template="{not json"
    )
    assert "不是合法 JSON" in result
    assert service.calls == []


async def test_visualize_tool_returns_datasets_and_html():
    service = _FakeService(
        payload={
            "html_bundle": "<div>chart</div>",
            "table_type": "type_b",
            "datasets": [{"name": "RE", "values": [1, 2, 3]}],
            "sources": [{"file_name": "x.pdf"}],
        }
    )
    result = await EmcVisualizeTool(service=service).execute(
        query="辐射限值表", doc_id="CISPR25"
    )
    data = json.loads(result)
    assert service.calls[0][0] == "/compare/visualize"
    assert service.calls[0][1]["doc_id"] == "aaaa1111bbbb"
    assert data["table_type"] == "type_b"
    assert data["html_bundle"] == "<div>chart</div>"
    assert data["datasets"][0]["name"] == "RE"


async def test_service_sends_auth_headers_and_maps_errors(monkeypatch):
    settings = Settings(
        rag_service_url="http://rag:8010",
        rag_service_token="svc",
        _env_file=None,
    )
    service = EmcCompareService(settings)
    sent: dict = {}

    class _Resp:
        status_code = 200
        text = "{}"

        def json(self):
            return {"answer": "ok"}

        def raise_for_status(self):
            pass

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            sent["url"] = url
            sent["headers"] = headers
            return _Resp()

    monkeypatch.setattr("agent_platform.tools.builtin.emc_service.httpx.AsyncClient", _Client)
    await service.post("/compare/emc", {"query": "q"})
    assert sent["url"] == "http://rag:8010/compare/emc"
    assert sent["headers"]["X-Local-Token"] == "svc"


async def test_service_maps_transport_error(monkeypatch):
    service = EmcCompareService(Settings(rag_service_url="http://rag:8010", _env_file=None))

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr("agent_platform.tools.builtin.emc_service.httpx.AsyncClient", _Client)
    with pytest.raises(EmcServiceError):
        await service.post("/compare/emc", {})


# ------------------------------------------------------------
# 文件名 → doc_id 解析（/compare/* 只接受 doc_id）
# ------------------------------------------------------------


async def test_resolve_doc_id_passes_through_hex_id():
    service = _FakeService(payload={})
    assert await service.resolve_doc_id("aaaa1111bbbb") == "aaaa1111bbbb"


async def test_resolve_doc_id_matches_exact_then_partial_name():
    service = _FakeService(payload={})
    assert await service.resolve_doc_id("CISPR25_2021.pdf") == "aaaa1111bbbb"
    assert await service.resolve_doc_id("iso_11452") == "cccc2222dddd"


async def test_resolve_doc_id_reports_available_names_when_missing():
    service = _FakeService(payload={})
    with pytest.raises(EmcServiceError) as exc:
        await service.resolve_doc_id("不存在的标准")
    assert "未找到匹配的文档" in str(exc.value)
    assert "CISPR25_2021.pdf" in str(exc.value)


async def test_resolve_doc_id_rejects_empty_value():
    service = _FakeService(payload={})
    with pytest.raises(EmcServiceError):
        await service.resolve_doc_id("   ")


async def test_compare_tool_surfaces_document_not_found():
    service = _FakeService(payload={"answer": "x"})
    result = await EmcCompareTool(service=service).execute(
        query="q", doc_a="CISPR25", doc_b="不存在的标准"
    )
    assert "EMC 对比失败" in result
    assert "未找到匹配的文档" in result
    assert service.calls == [], "解析失败时不应发起对比请求"


async def test_list_documents_maps_http_error(monkeypatch):
    service = EmcCompareService(
        Settings(rag_service_url="http://rag:8010", _env_file=None)
    )

    class _Resp:
        status_code = 500
        text = "boom"

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr("agent_platform.tools.builtin.emc_service.httpx.AsyncClient", _Client)
    with pytest.raises(EmcServiceError):
        await service.list_documents()
