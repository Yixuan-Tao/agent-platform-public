# Test values are synthetic fixtures for software behavior.
"""EmcVisualizeTool 单元测试（MagicMock）

覆盖范围：
    - _retrieve_table_chunks：doc_id / query / 异常分支
    - _convert_chunks_to_dataset：structured_data / json_data / content JSON / 未知类型 / 异常
    - visualize：空数据 / 完整流程 / HTML 失败
    - visualize_overlay：双文档叠加 / 单文档缺失 / 自动类型检测 / 手动指定类型
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.emc_visualize import EmcVisualizeTool


# ============================================================
# 辅助：构建测试用 chunk 工厂
# ============================================================

def _make_chunk(**overrides) -> dict:
    """创建标准 table chunk 字典。"""
    chunk = {
        "id": "t1",
        "modality": "table",
        "content": "频率/MHz  限值/dBµV/m",
        "page": 5,
        "doc_id": "doc_test",
    }
    chunk.update(overrides)
    return chunk


# ============================================================
# _retrieve_table_chunks 单测
# ============================================================

class TestRetrieveTableChunks:
    """测试 _retrieve_table_chunks（Mock rag_service）"""

    def test_with_doc_id_calls_retrieve_all_by_doc(self):
        """有 doc_id → 调用 retrieve_all_by_doc → 返回 chunks"""
        mock_rag = MagicMock()
        expected = [_make_chunk(id="t1")]
        mock_rag.retrieve_all_by_doc.return_value = expected

        tool = EmcVisualizeTool(mock_rag)
        result = tool._retrieve_table_chunks("query", "doc_123", top_k=5)

        mock_rag.retrieve_all_by_doc.assert_called_once_with("doc_123")
        assert result == expected

    def test_with_doc_id_fallback_to_query(self):
        """有 doc_id 但无 retrieve_all_by_doc → 回退到 query + filter_expr"""
        mock_rag = MagicMock(spec=["query"])  # 只有 query，没有 retrieve_all_by_doc
        mock_rag.query.return_value = {
            "sources": [_make_chunk(id="t1")],
        }

        tool = EmcVisualizeTool(mock_rag)
        result = tool._retrieve_table_chunks("辐射发射", "doc_456", top_k=5)

        mock_rag.query.assert_called_once()
        call_kwargs = mock_rag.query.call_args[1]
        assert call_kwargs["top_k"] == 10  # top_k * 2
        assert "doc_456" in call_kwargs["filter_expr"]
        assert result == [_make_chunk(id="t1")]

    def test_with_query_no_doc_id(self):
        """无 doc_id → 调用 retrieve → 直接返回 chunk 列表"""
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = [_make_chunk(id="t1"), _make_chunk(id="t2")]

        tool = EmcVisualizeTool(mock_rag)
        result = tool._retrieve_table_chunks("辐射发射限值", None, top_k=5)

        mock_rag.retrieve.assert_called_once_with("辐射发射限值", top_k=5)
        assert len(result) == 2

    def test_without_query_or_doc(self):
        """query 和 doc_id 都为空 → 返回 []"""
        mock_rag = MagicMock()

        tool = EmcVisualizeTool(mock_rag)
        result = tool._retrieve_table_chunks(None, None, top_k=5)

        # retrieve 和 query 都不会被调用（因为 query 为 None）
        mock_rag.retrieve.assert_not_called()
        mock_rag.query.assert_not_called()
        assert result == []

    def test_rag_raises_exception(self):
        """rag_service 抛出异常 → 返回 []"""
        mock_rag = MagicMock()
        mock_rag.retrieve.side_effect = Exception("Connection error")

        tool = EmcVisualizeTool(mock_rag)
        result = tool._retrieve_table_chunks("query", None, top_k=5)

        assert result == []

    def test_retrieve_all_by_doc_raises(self):
        """retrieve_all_by_doc 抛出异常 → 返回 []"""
        mock_rag = MagicMock()
        mock_rag.retrieve_all_by_doc.side_effect = RuntimeError("DB down")

        tool = EmcVisualizeTool(mock_rag)
        result = tool._retrieve_table_chunks("query", "doc_123", top_k=5)

        assert result == []

    def test_retrieve_returns_empty_list(self):
        """retrieve 返回空列表 → 返回 []"""
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = []

        tool = EmcVisualizeTool(mock_rag)
        result = tool._retrieve_table_chunks("query", None, top_k=5)

        assert result == []


# ============================================================
# _convert_chunks_to_dataset 单测
# ============================================================

class TestConvertChunksToDataset:
    """测试 _convert_chunks_to_dataset（Mock converter）"""

    def test_with_structured_data(self):
        """chunks 有 structured_data → 调用 converter"""
        mock_converter = MagicMock(return_value={"name": "test", "data": [{"freq": 100}]})
        chunks = [_make_chunk(structured_data=[{"freq": 100, "limit": 40}])]

        with patch.dict("tools.emc_visualize.CONVERTERS", {"Type_A": mock_converter}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool._convert_chunks_to_dataset(chunks, "Type_A", "test.pdf")

        assert result == {"name": "test", "data": [{"freq": 100}]}
        mock_converter.assert_called_once()

    def test_with_json_data(self):
        """chunks 有 json_data（fallback 字段）→ 调用 converter"""
        mock_converter = MagicMock(return_value={"name": "test", "data": []})
        chunks = [_make_chunk(json_data=[{"freq": 200, "limit": 50}])]

        with patch.dict("tools.emc_visualize.CONVERTERS", {"Type_B": mock_converter}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool._convert_chunks_to_dataset(chunks, "Type_B", "doc_b.pdf")

        assert result is not None
        mock_converter.assert_called_once()

    def test_structured_data_dict_vs_list(self):
        """structured_data 为 dict 时会被包裹进列表"""
        mock_converter = MagicMock(return_value={"name": "test", "data": []})
        chunks = [_make_chunk(structured_data={"freq": 100, "limit": 40})]

        with patch.dict("tools.emc_visualize.CONVERTERS", {"Type_A": mock_converter}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            tool._convert_chunks_to_dataset(chunks, "Type_A", "test.pdf")

        # converter 收到的 data 应包含该 dict 项
        call_data = mock_converter.call_args[0][0]
        assert {"freq": 100, "limit": 40} in call_data["data"]

    def test_from_content_json(self):
        """content 是 JSON 字符串（dict 格式）→ 被解析并传入 converter"""
        mock_converter = MagicMock(return_value={"name": "parsed", "data": []})
        # 源码守卫检查 content.strip().startswith("{")，因此必须用 dict 格式
        json_content = json.dumps({"freq": 30, "limit": 60})
        chunks = [_make_chunk(content=json_content, structured_data=None)]

        with patch.dict("tools.emc_visualize.CONVERTERS", {"Type_A": mock_converter}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            tool._convert_chunks_to_dataset(chunks, "Type_A", "test.pdf")

        call_data = mock_converter.call_args[0][0]
        assert {"freq": 30, "limit": 60} in call_data["data"]

    def test_no_structured_data(self):
        """chunks 既无 structured_data 也无 json_data 且 content 非 JSON → 返回 None"""
        chunks = [_make_chunk(content="普通文本，非 JSON 格式", structured_data=None)]

        with patch.dict("tools.emc_visualize.CONVERTERS", {}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool._convert_chunks_to_dataset(chunks, "Type_A", "test.pdf")

        assert result is None

    def test_unknown_type(self):
        """table_type 不在 CONVERTERS 中 → 返回 None"""
        chunks = [_make_chunk(structured_data=[{"freq": 100}])]

        with patch.dict("tools.emc_visualize.CONVERTERS", {}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool._convert_chunks_to_dataset(chunks, "NonExistentType", "test.pdf")

        assert result is None

    def test_converter_raises(self):
        """converter 抛出异常 → 返回 None"""
        mock_converter = MagicMock(side_effect=ValueError("convert failed"))
        chunks = [_make_chunk(structured_data=[{"freq": 100}])]

        with patch.dict("tools.emc_visualize.CONVERTERS", {"Type_A": mock_converter}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool._convert_chunks_to_dataset(chunks, "Type_A", "test.pdf")

        assert result is None

    def test_empty_chunks_list(self):
        """空 chunks 列表 → 返回 None"""
        with patch.dict("tools.emc_visualize.CONVERTERS", {}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool._convert_chunks_to_dataset([], "Type_A", "test.pdf")

        assert result is None

    def test_multiple_chunks_data_merged(self):
        """多个 chunk 的数据应合并后传给 converter"""
        mock_converter = MagicMock(return_value={"name": "merged", "data": []})
        chunks = [
            _make_chunk(id="t1", structured_data=[{"freq": 100}]),
            _make_chunk(id="t2", structured_data=[{"freq": 200}]),
        ]

        with patch.dict("tools.emc_visualize.CONVERTERS", {"Type_A": mock_converter}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            tool._convert_chunks_to_dataset(chunks, "Type_A", "test.pdf")

        call_data = mock_converter.call_args[0][0]["data"]
        assert len(call_data) == 2
        assert {"freq": 100} in call_data
        assert {"freq": 200} in call_data


# ============================================================
# visualize 单测
# ============================================================

class TestVisualize:
    """测试 visualize 方法（Mock 内部方法和外部依赖）"""

    def test_no_chunks_returns_error(self):
        """_retrieve_table_chunks 返回空 → 返回 error dict"""
        tool = EmcVisualizeTool(rag_service=MagicMock())
        with patch.object(tool, "_retrieve_table_chunks", return_value=[]):
            result = tool.visualize("辐射发射限值")

        assert result["type"] == "Unknown"
        assert "error" in result
        assert "未检索到相关 table chunks" in result["error"]

    def test_no_table_chunks_fallback_to_all(self):
        """chunks 中无 modality=table → 全部 chunks 作为备选"""
        non_table_chunks = [
            {"id": "t1", "modality": "text", "content": "辐射发射限值 30MHz 40dBµV/m"},
        ]
        # get_type_confidence 能匹配到 Type_A，走正常流程
        mock_converter = MagicMock(return_value={"name": "test", "data": []})

        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", return_value=non_table_chunks), \
                patch.object(EmcVisualizeTool, "_convert_chunks_to_dataset", return_value={"name": "test", "data": []}), \
                patch("tools.emc_visualize.get_type_confidence", return_value={
                    "type": "Type_A", "confidence": 0.95, "scores": {"Type_A": 10}, "total": 10,
                }), \
                patch("tools.emc_visualize.generate_html", return_value="<html>"), \
                patch("tools.emc_visualize.to_inline_bundle", return_value={"html": "<html>"}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize("辐射发射限值")

        assert "error" not in result
        assert result["type"] == "Type_A"

    def test_unknown_type_returns_error(self):
        """类型检测为 Unknown → 返回 error dict"""
        chunks = [_make_chunk(content="无关键词文本")]

        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", return_value=chunks), \
                patch("tools.emc_visualize.get_type_confidence", return_value={
                    "type": "Unknown", "confidence": 0.0, "scores": {}, "total": 0,
                }):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize("无关查询")

        assert result["type"] == "Unknown"
        assert "error" in result

    def test_conversion_failure_returns_error(self):
        """_convert_chunks_to_dataset 返回 None → 返回 error dict"""
        chunks = [_make_chunk(structured_data=[{"freq": 100}])]

        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", return_value=chunks), \
                patch("tools.emc_visualize.get_type_confidence", return_value={
                    "type": "Type_A", "confidence": 0.9, "scores": {"Type_A": 5}, "total": 5,
                }), \
                patch.object(EmcVisualizeTool, "_convert_chunks_to_dataset", return_value=None):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize("辐射发射")

        assert result["type"] == "Type_A"
        assert "error" in result

    def test_success_full_flow(self):
        """完整流程成功 → 返回包含 html/bundle/datasets 的 dict"""
        chunks = [_make_chunk(structured_data=[{"freq": 100, "limit": 40}])]
        dataset = {"name": "test.pdf", "type": "Type_A", "data": [{"freq": 100, "limit": 40}]}

        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", return_value=chunks), \
                patch("tools.emc_visualize.get_type_confidence", return_value={
                    "type": "Type_A", "confidence": 0.95, "scores": {"Type_A": 10}, "total": 10,
                }), \
                patch.object(EmcVisualizeTool, "_convert_chunks_to_dataset", return_value=dataset), \
                patch("tools.emc_visualize.generate_html", return_value="<html>complete</html>"), \
                patch("tools.emc_visualize.to_inline_bundle", return_value={"html": "<html>complete</html>"}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize("辐射发射限值", doc_id="doc_123", top_k=5)

        assert result["type"] == "Type_A"
        assert result["confidence"] == 0.95
        assert result["datasets"] == [dataset]
        assert result["html"] == "<html>complete</html>"
        assert result["bundle"] == {"html": "<html>complete</html>"}
        # 应包含 type_detail
        assert "type_detail" in result

    def test_html_generation_failure(self):
        """generate_html 或 to_inline_bundle 抛出异常 → 返回 error dict + datasets"""
        chunks = [_make_chunk(structured_data=[{"freq": 100}])]
        dataset = {"name": "test.pdf", "type": "Type_A", "data": [{"freq": 100}]}

        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", return_value=chunks), \
                patch("tools.emc_visualize.get_type_confidence", return_value={
                    "type": "Type_A", "confidence": 0.9, "scores": {"Type_A": 5}, "total": 5,
                }), \
                patch.object(EmcVisualizeTool, "_convert_chunks_to_dataset", return_value=dataset), \
                patch("tools.emc_visualize.generate_html", side_effect=Exception("Chart error")):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize("辐射发射")

        assert result["type"] == "Type_A"
        assert "error" in result
        assert "HTML 生成失败" in result["error"]
        # datasets 仍应返回
        assert result["datasets"] == [dataset]


# ============================================================
# visualize_overlay 单测
# ============================================================

class TestVisualizeOverlay:
    """测试 visualize_overlay 方法"""

    def test_both_docs_have_data(self):
        """两份文档均有数据 → 返回两个 dataset 的对比结果"""
        chunks_a = [_make_chunk(id="a1", structured_data=[{"freq": 100}])]
        chunks_b = [_make_chunk(id="b1", structured_data=[{"freq": 200}])]
        dataset_a = {"name": "doc_a.pdf", "data": [{"freq": 100}]}
        dataset_b = {"name": "doc_b.pdf", "data": [{"freq": 200}]}

        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", side_effect=[chunks_a, chunks_b]), \
                patch.object(EmcVisualizeTool, "_convert_chunks_to_dataset", side_effect=[dataset_a, dataset_b]), \
                patch("tools.emc_visualize.get_type_confidence", return_value={
                    "type": "Type_A", "confidence": 0.9, "scores": {"Type_A": 5}, "total": 5,
                }), \
                patch("tools.emc_visualize.generate_html", return_value="<html>overlay</html>"), \
                patch("tools.emc_visualize.to_inline_bundle", return_value={"html": "<html>overlay</html>"}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize_overlay("doc_a", "doc_b", query="辐射发射")

        assert "error" not in result
        assert result["type"] == "Type_A"
        assert result["datasets"] == [dataset_a, dataset_b]
        assert result["html"] == "<html>overlay</html>"

    def test_one_doc_missing(self):
        """一份文档无数据 → 返回 error dict"""
        chunks_a = [_make_chunk(id="a1")]
        chunks_b = []

        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", side_effect=[chunks_a, chunks_b]):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize_overlay("doc_a", "doc_b")

        assert "error" in result
        assert "文档B" in result["error"]

    def test_both_docs_missing(self):
        """两份文档都无数据 → 返回 error dict"""
        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", return_value=[]):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize_overlay("doc_a", "doc_b")

        assert "error" in result
        assert "文档A" in result["error"]
        assert "文档B" in result["error"]

    def test_auto_type_detection(self):
        """未指定 table_type → 从 all_chunks 自动检测"""
        chunks_a = [_make_chunk(id="a1", structured_data=[{"freq": 100}])]
        chunks_b = [_make_chunk(id="b1", structured_data=[{"freq": 200}])]
        dataset_a = {"name": "doc_a.pdf", "data": [{"freq": 100}]}
        dataset_b = {"name": "doc_b.pdf", "data": [{"freq": 200}]}

        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", side_effect=[chunks_a, chunks_b]), \
                patch.object(EmcVisualizeTool, "_convert_chunks_to_dataset", side_effect=[dataset_a, dataset_b]), \
                patch("tools.emc_visualize.get_type_confidence", return_value={
                    "type": "Type_B", "confidence": 0.85, "scores": {"Type_B": 8}, "total": 10,
                }) as mock_detect, \
                patch("tools.emc_visualize.generate_html", return_value="<html>"), \
                patch("tools.emc_visualize.to_inline_bundle", return_value={"html": "<html>"}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize_overlay("doc_a", "doc_b", query="抗扰度")

        # 验证自动检测被调用，且传入的是合并后的 chunks
        assert mock_detect.call_count == 1
        # 应该用 chunks_a + chunks_b
        call_chunks = mock_detect.call_args[0][0]
        assert len(call_chunks) == 2
        assert result["type"] == "Type_B"

    def test_manual_type(self):
        """指定 table_type → 使用指定类型，不自动检测"""
        chunks_a = [_make_chunk(id="a1", structured_data=[{"freq": 100}])]
        chunks_b = [_make_chunk(id="b1", structured_data=[{"freq": 200}])]
        dataset_a = {"name": "doc_a.pdf", "data": [{"freq": 100}]}
        dataset_b = {"name": "doc_b.pdf", "data": [{"freq": 200}]}

        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", side_effect=[chunks_a, chunks_b]), \
                patch.object(EmcVisualizeTool, "_convert_chunks_to_dataset", side_effect=[dataset_a, dataset_b]), \
                patch("tools.emc_visualize.get_type_confidence") as mock_detect, \
                patch("tools.emc_visualize.generate_html", return_value="<html>"), \
                patch("tools.emc_visualize.to_inline_bundle", return_value={"html": "<html>"}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize_overlay("doc_a", "doc_b", table_type="Type_C", top_k=5)

        # 手动指定类型时不应调用 get_type_confidence
        mock_detect.assert_not_called()
        assert result["type"] == "Type_C"

    def test_auto_detection_unknown_returns_error(self):
        """自动检测结果为 Unknown → 返回 error dict"""
        chunks_a = [_make_chunk(id="a1", content="no keywords")]
        chunks_b = [_make_chunk(id="b1", content="no keywords")]

        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", side_effect=[chunks_a, chunks_b]), \
                patch("tools.emc_visualize.get_type_confidence", return_value={
                    "type": "Unknown", "confidence": 0.0, "scores": {}, "total": 0,
                }):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize_overlay("doc_a", "doc_b")

        assert "error" in result
        assert "无法自动检测" in result["error"]

    def test_conversion_failure_returns_error(self):
        """某份文档转换失败 → 返回 error dict"""
        chunks_a = [_make_chunk(id="a1", structured_data=[{"freq": 100}])]
        chunks_b = [_make_chunk(id="b1", structured_data=[{"freq": 200}])]

        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", side_effect=[chunks_a, chunks_b]), \
                patch.object(EmcVisualizeTool, "_convert_chunks_to_dataset", side_effect=[
                    {"name": "doc_a.pdf", "data": [{"freq": 100}]},
                    None,  # doc_b 转换失败
                ]), \
                patch("tools.emc_visualize.get_type_confidence", return_value={
                    "type": "Type_A", "confidence": 0.9, "scores": {"Type_A": 5}, "total": 5,
                }):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize_overlay("doc_a", "doc_b")

        assert "error" in result
        assert "文档B" in result["error"]

    def test_html_generation_failure(self):
        """HTML 生成失败 → 返回 error + datasets"""
        chunks_a = [_make_chunk(id="a1", structured_data=[{"freq": 100}])]
        chunks_b = [_make_chunk(id="b1", structured_data=[{"freq": 200}])]
        dataset_a = {"name": "doc_a.pdf", "data": [{"freq": 100}]}
        dataset_b = {"name": "doc_b.pdf", "data": [{"freq": 200}]}

        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", side_effect=[chunks_a, chunks_b]), \
                patch.object(EmcVisualizeTool, "_convert_chunks_to_dataset", side_effect=[dataset_a, dataset_b]), \
                patch("tools.emc_visualize.get_type_confidence", return_value={
                    "type": "Type_A", "confidence": 0.9, "scores": {"Type_A": 5}, "total": 5,
                }), \
                patch("tools.emc_visualize.generate_html", side_effect=Exception("Render error")):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize_overlay("doc_a", "doc_b")

        assert "error" in result
        assert "HTML 生成失败" in result["error"]
        # datasets 仍应保留
        assert len(result["datasets"]) == 2

    def test_doc_a_name_and_doc_b_name_in_result(self):
        """返回结果中包含 doc_a_name 和 doc_b_name"""
        chunks_a = [_make_chunk(id="a1", structured_data=[{"freq": 100}])]
        chunks_b = [_make_chunk(id="b1", structured_data=[{"freq": 200}])]
        dataset_a = {"name": "标准A.pdf", "data": [{"freq": 100}]}
        dataset_b = {"name": "标准B.pdf", "data": [{"freq": 200}]}

        with patch.object(EmcVisualizeTool, "_retrieve_table_chunks", side_effect=[chunks_a, chunks_b]), \
                patch.object(EmcVisualizeTool, "_convert_chunks_to_dataset", side_effect=[dataset_a, dataset_b]), \
                patch("tools.emc_visualize.get_type_confidence", return_value={
                    "type": "Type_A", "confidence": 0.9, "scores": {"Type_A": 5}, "total": 5,
                }), \
                patch("tools.emc_visualize.generate_html", return_value="<html>"), \
                patch("tools.emc_visualize.to_inline_bundle", return_value={"html": "<html>"}):
            tool = EmcVisualizeTool(rag_service=MagicMock())
            result = tool.visualize_overlay("doc_a", "doc_b")

        assert result["doc_a_name"] == "标准A.pdf"
        assert result["doc_b_name"] == "标准B.pdf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
