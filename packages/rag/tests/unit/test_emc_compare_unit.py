# Test values are synthetic fixtures for software behavior.
"""
EMC 对比工具单元测试 — 使用 MagicMock 隔离 LLM / 检索依赖

覆盖范围：
    - _is_comparison_query：对比意图检测正则
    - _resolve_doc_name：文档名解析静态方法
    - extract_parameters：参数提取（Mock generator）
    - generate_compare_from_params：对比生成（Mock generator）
    - try_comparison：集成流程（空 filter / 单文档跳过 / 多文档触发）
    - compare：主入口（Mock 线程池拉取）
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config import config
from tools.emc_compare import EmcCompareTool


# ============================================================
# _is_comparison_query：对比意图检测（纯方法，正则匹配）
# ============================================================

class TestIsComparisonQuery:
    """测试 _is_comparison_query 纯方法（正则匹配）。"""

    def test_compare_chinese(self):
        """中文对比关键词"对比"应返回 True。"""
        tool = EmcCompareTool(
            rag_service=MagicMock(),
            generator=MagicMock(), config=MagicMock(),
        )
        assert tool._is_comparison_query("对比两颗TVS") is True

    def test_compare_english(self):
        """英文对比关键词"compare"应返回 True。"""
        tool = EmcCompareTool(
            rag_service=MagicMock(),
            generator=MagicMock(), config=MagicMock(),
        )
        assert tool._is_comparison_query("compare these two") is True

    def test_no_compare(self):
        """无对比意图的查询应返回 False。"""
        tool = EmcCompareTool(
            rag_service=MagicMock(),
            generator=MagicMock(), config=MagicMock(),
        )
        assert tool._is_comparison_query("什么是击穿电压") is False

    def test_empty(self):
        """空字符串应返回 False。"""
        tool = EmcCompareTool(
            rag_service=MagicMock(),
            generator=MagicMock(), config=MagicMock(),
        )
        assert tool._is_comparison_query("") is False


# ============================================================
# _resolve_doc_name：文档名解析（静态方法）
# ============================================================

class TestResolveDocName:
    """测试 _resolve_doc_name 静态方法。"""

    def test_from_hits(self):
        """hits 中有 file_name 时返回该名称。"""
        hits = [{"file_name": "datasheet_a.pdf", "content": "test"}]
        result = EmcCompareTool._resolve_doc_name(hits, "fallback_name")
        assert result == "datasheet_a.pdf"

    def test_from_fallback(self):
        """hits 为空时返回 fallback。"""
        result = EmcCompareTool._resolve_doc_name([], "fallback_name")
        assert result == "fallback_name"

    def test_hits_without_name(self):
        """hits 中缺少 file_name 字段时返回 fallback。"""
        hits = [{"content": "no file_name field"}]
        result = EmcCompareTool._resolve_doc_name(hits, "fallback_name")
        assert result == "fallback_name"


# ============================================================
# extract_parameters：参数提取（Mock generator）
# ============================================================

class TestExtractParameters:
    """测试 extract_parameters 方法（Mock generator）。"""

    def test_success(self, mock_generator):
        """正常调用：hits + doc_name → 返回解析后的 JSON 参数字典。"""
        hits = [
            {"content": "反向击穿电压 27V", "page": 1},
            {"content": "钳位电压 43V", "page": 2},
        ]
        tool = EmcCompareTool(
            rag_service=MagicMock(),
            generator=mock_generator, config=config,
        )
        result = tool.extract_parameters(hits, "doc_a")
        # mock_generator 返回固定 JSON，safe_parse_json 解析后得到参数字典
        assert result == {"反向击穿电压": "27V", "钳位电压": "43V", "封装类型": "SOD-323"}

    def test_empty_hits(self, mock_generator):
        """空 hits 应直接返回空字典。"""
        tool = EmcCompareTool(
            rag_service=MagicMock(),
            generator=mock_generator, config=config,
        )
        result = tool.extract_parameters([], "doc_a")
        assert result == {}

    def test_json_parse_fails(self, mock_generator):
        """generator 返回非法 JSON 时 safe_parse_json 失败，返回空字典。"""
        mock_generator.generate.return_value = "不是有效 JSON"
        tool = EmcCompareTool(
            rag_service=MagicMock(),
            generator=mock_generator, config=config,
        )
        result = tool.extract_parameters(
            [{"content": "some data", "page": 1}], "doc_a",
        )
        assert result == {}

    def test_generator_raises(self, mock_generator):
        """generator.generate() 抛出异常时返回空字典。"""
        mock_generator.generate.side_effect = Exception("API 调用失败")
        tool = EmcCompareTool(
            rag_service=MagicMock(),
            generator=mock_generator, config=config,
        )
        result = tool.extract_parameters(
            [{"content": "some data", "page": 1}], "doc_a",
        )
        assert result == {}


# ============================================================
# generate_compare_from_params：对比生成（Mock generator）
# ============================================================

class TestGenerateCompareFromParams:
    """测试 generate_compare_from_params（Mock generator）。"""

    def test_success(self, mock_generator):
        """正常调用：传入有效 params → 返回 generator 输出。"""
        # mock_generator.generate 默认返回固定 JSON 字符串
        tool = EmcCompareTool(
            rag_service=MagicMock(),
            generator=mock_generator, config=config,
        )
        result = tool.generate_compare_from_params(
            query="对比钳位电压",
            params_a={"反向击穿电压": "27V"},
            params_b={"反向击穿电压": "30V"},
            doc_a_name="doc_a",
            doc_b_name="doc_b",
            language="zh",
        )
        # 验证 generator.generate() 被调用且返回原始 JSON 字符串
        assert result == '{"反向击穿电压": "27V", "钳位电压": "43V", "封装类型": "SOD-323"}'

    def test_generator_raises(self, mock_generator):
        """generator.generate() 抛出异常时返回错误提示字符串。"""
        mock_generator.generate.side_effect = Exception("API 超时")
        tool = EmcCompareTool(
            rag_service=MagicMock(),
            generator=mock_generator, config=config,
        )
        result = tool.generate_compare_from_params(
            query="对比",
            params_a={}, params_b={},
            doc_a_name="a", doc_b_name="b",
        )
        assert result.startswith("对比生成失败")


# ============================================================
# try_comparison：集成流程
# ============================================================

class TestTryComparison:
    """测试 try_comparison 集成方法。"""

    def test_filter_expr_skips(self, mock_retriever, mock_generator):
        """filter_expr 已设置时应直接返回 None（跳过对比）。"""
        tool = EmcCompareTool(
            rag_service=mock_retriever,
            generator=mock_generator, config=config,
        )
        result = tool.try_comparison(
            query="对比两颗TVS", top_k=10,
            language="zh", filter_expr="some_filter",
        )
        assert result is None

    def test_no_compare_intent(self, mock_retriever, mock_generator):
        """无对比关键词的 query 应返回 None。"""
        tool = EmcCompareTool(
            rag_service=mock_retriever,
            generator=mock_generator, config=config,
        )
        result = tool.try_comparison(
            query="什么是击穿电压", top_k=10,
            language="zh", filter_expr=None,
        )
        assert result is None

    def test_single_doc_skips(self, mock_retriever, mock_generator):
        """检索结果仅来自单个文档时应返回 None。"""
        # mock_retriever.retrieve 默认返回 1 个 chunk（doc_id="doc_a"）
        tool = EmcCompareTool(
            rag_service=mock_retriever,
            generator=mock_generator, config=config,
        )
        result = tool.try_comparison(
            query="对比这两颗", top_k=10,
            language="zh", filter_expr=None,
        )
        assert result is None

    def test_multi_doc_triggers(self, mock_retriever, mock_generator):
        """检索结果来自多个文档时应触发对比并返回结果。"""
        # 模拟来自两个不同文档的检索结果
        mock_retriever.retrieve.return_value = [
            {"id": "c1", "score": 0.9, "doc_id": "doc_a",
             "file_name": "doc_a.pdf", "page": 1, "modality": "text",
             "content": "Breakdown voltage 27V",
             "searchable_text": "Breakdown voltage 27V", "section": "Elec"},
            {"id": "c2", "score": 0.8, "doc_id": "doc_b",
             "file_name": "doc_b.pdf", "page": 1, "modality": "text",
             "content": "Breakdown voltage 30V",
             "searchable_text": "Breakdown voltage 30V", "section": "Elec"},
        ]
        tool = EmcCompareTool(
            rag_service=mock_retriever,
            generator=mock_generator, config=config,
        )
        # 模拟 _compare_internal 返回简单结果（避免进入复杂的集成路径）
        with patch.object(
            tool, "_compare_internal",
            return_value={"answer": "对比结果", "sources": [], "image_chunks": []},
        ):
            result = tool.try_comparison(
                query="对比这两份文档", top_k=10,
                language="zh", filter_expr=None,
            )
            assert result is not None
            assert result["answer"] == "对比结果"


# ============================================================
# compare：主入口（Mock 线程池拉取）
# ============================================================

class TestCompare:
    """测试 compare 主入口方法。"""

    def test_compare_returns_dict(self, mock_retriever, mock_generator):
        """compare 应返回包含 answer 键的字典。"""
        tool = EmcCompareTool(
            rag_service=mock_retriever,
            generator=mock_generator, config=config,
        )
        # 模拟 _fetch_two_docs_parallel 避免 ThreadPoolExecutor 依赖
        with patch.object(
            tool, "_fetch_two_docs_parallel", return_value=([], []),
        ):
            result = tool.compare(query="对比", doc_a="doc_a", doc_b="doc_b")
            assert "answer" in result
            # 空 hits 时 _run_two_stage_compare 返回无内容结果
            # answer 为 generator.generate() 的 mock 返回值
            assert isinstance(result["answer"], str)
