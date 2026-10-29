#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
元器件替换判断工具 ComponentCheckTool 的 LLM 相关方法单元测试

覆盖以下 LLM 依赖的实例方法（使用 MagicMock 隔离外部调用）：
    - _llm_guided_search
    - _extract_value_with_llm
    - _locate_params_in_chunks
    - compare_with_template
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.component_check import ComponentCheckTool


# ============================================================
# 辅助函数：构造一个 ComponentCheckTool 实例
# ============================================================

def _make_tool(rag_service=None, generator=None, config=None):
    """用默认 MagicMock 快速创建 ComponentCheckTool 实例。"""
    return ComponentCheckTool(
        rag_service=rag_service or MagicMock(),
        generator=generator or MagicMock(),
        config=config,
    )


# ============================================================
# TestLlmGuidedSearch
# ============================================================

class TestLlmGuidedSearch:
    """测试 _llm_guided_search —— 依赖 self._generator.generate()"""

    def test_happy_path(self, mock_generator):
        """
        正常路径：generator 返回 "1,3,5"，只存在 idx 1 和 3
        → 返回 2 个 chunk
        """
        tool = _make_tool(generator=mock_generator)
        mock_generator.generate.return_value = "1,3,5"

        chunk_texts = [
            {"idx": 0, "text": "VBR=27V", "section": "Elec"},
            {"idx": 1, "text": "Package=SOD-323", "section": "Pkg"},
            {"idx": 2, "text": "VBR=30V", "section": "Elec"},
            {"idx": 3, "text": "Temp=-40~125C", "section": "Env"},
            {"idx": 4, "text": "Ir=1uA", "section": "Elec"},
        ]
        result = tool._llm_guided_search(chunk_texts, "VBR", {"unit": "V"}, top_k=3)
        # idx 1,3 存在，5 不存在
        assert len(result) == 2
        assert [c["idx"] for c in result] == [1, 3]

    def test_empty_summaries(self, mock_generator):
        """chunk_texts 为空 → 返回 []"""
        tool = _make_tool(generator=mock_generator)
        result = tool._llm_guided_search([], "VBR", {"unit": "V"}, top_k=3)
        assert result == []

    def test_generator_raises(self, mock_generator):
        """generator.generate() 抛出异常 → 返回 []"""
        tool = _make_tool(generator=mock_generator)
        mock_generator.generate.side_effect = RuntimeError("API error")

        chunk_texts = [
            {"idx": 0, "text": "VBR=27V", "section": "Elec"},
        ]
        result = tool._llm_guided_search(chunk_texts, "VBR", {"unit": "V"}, top_k=3)
        assert result == []

    def test_generator_returns_non_digits(self, mock_generator):
        """generator 返回 "not found"（无数字）→ 返回 []"""
        tool = _make_tool(generator=mock_generator)
        mock_generator.generate.return_value = "not found"

        chunk_texts = [
            {"idx": 0, "text": "VBR=27V", "section": "Elec"},
            {"idx": 1, "text": "Package=SOD-323", "section": "Pkg"},
        ]
        result = tool._llm_guided_search(chunk_texts, "VBR", {"unit": "V"}, top_k=3)
        assert result == []

    def test_top_k_limits(self, mock_generator):
        """generator 返回 "1,2,3,4,5,6" 但 top_k=3 → 只返回 3 个"""
        tool = _make_tool(generator=mock_generator)
        mock_generator.generate.return_value = "1,2,3,4,5,6"

        chunk_texts = [
            {"idx": i, "text": f"chunk {i}", "section": f"Sec{i}"}
            for i in range(10)
        ]
        result = tool._llm_guided_search(chunk_texts, "VBR", {"unit": "V"}, top_k=3)
        assert len(result) == 3


# ============================================================
# TestExtractValueWithLlm
# ============================================================

class TestExtractValueWithLlm:
    """测试 _extract_value_with_llm —— 依赖 self._generator.generate()"""

    def test_success(self, mock_generator):
        """generator 返回 "27V" → 返回 "27V" """
        tool = _make_tool(generator=mock_generator)
        mock_generator.generate.return_value = "27V"

        candidates = [
            {"idx": 0, "text": "VBR=27V", "section": "Elec", "page": "1",
             "content": "Breakdown voltage is 27V"},
        ]
        result = tool._extract_value_with_llm("VBR", {"unit": "V"}, candidates)
        assert result == "27V"

    def test_not_found(self, mock_generator):
        """generator 返回 "NOT_FOUND" → 返回 None"""
        tool = _make_tool(generator=mock_generator)
        mock_generator.generate.return_value = "NOT_FOUND"

        candidates = [{"idx": 0, "text": "nothing here", "section": "", "page": "1",
                       "content": "some random text"}]
        result = tool._extract_value_with_llm("VBR", {"unit": "V"}, candidates)
        assert result is None

    def test_clean_result(self, mock_generator):
        """generator 返回 '"27V"'（带引号）→ 清洗后返回 "27V" """
        tool = _make_tool(generator=mock_generator)
        mock_generator.generate.return_value = '"27V"'

        candidates = [{"idx": 0, "text": "VBR=27V", "section": "Elec", "page": "1",
                       "content": "Breakdown voltage is 27V"}]
        result = tool._extract_value_with_llm("VBR", {"unit": "V"}, candidates)
        assert result == "27V"

    def test_empty_candidates(self, mock_generator):
        """candidates 为空列表 → 不调用 LLM，直接返回 None"""
        tool = _make_tool(generator=mock_generator)
        result = tool._extract_value_with_llm("VBR", {"unit": "V"}, [])
        assert result is None
        # 确保没有调用 generator
        mock_generator.generate.assert_not_called()

    def test_generator_raises(self, mock_generator):
        """generator.generate() 抛出异常 → 返回 None"""
        tool = _make_tool(generator=mock_generator)
        mock_generator.generate.side_effect = RuntimeError("API error")

        candidates = [{"idx": 0, "text": "VBR=27V", "section": "Elec", "page": "1",
                       "content": "Breakdown voltage is 27V"}]
        result = tool._extract_value_with_llm("VBR", {"unit": "V"}, candidates)
        assert result is None


# ============================================================
# TestLocateParamsInChunks
# ============================================================

class TestLocateParamsInChunks:
    """测试 _locate_params_in_chunks（Mock 内部的 keyword_rank + LLM 方法）"""

    # 共享测试 chunk 数据
    CHUNKS = [
        {"idx": 0, "text": "Breakdown voltage 27V", "section": "Electrical",
         "content": "Breakdown voltage is 27V", "page": "1"},
        {"idx": 1, "text": "Package: SOD-323", "section": "Package",
         "content": "Package type is SOD-323", "page": "2"},
        {"idx": 2, "text": "Reverse current 1uA", "section": "Electrical",
         "content": "Reverse current 1uA", "page": "3"},
    ]

    def test_keyword_hit(self):
        """
        关键词匹配成功 + LLM 提取成功
        → values 包含该参数，failed 为空
        """
        tool = _make_tool()

        with (
            patch.object(tool, "_keyword_rank_chunks") as mock_kw,
            patch.object(tool, "_llm_guided_search") as mock_llm_search,
            patch.object(tool, "_extract_value_with_llm") as mock_extract,
        ):
            # keyword_rank 找到候选
            mock_kw.return_value = [self.CHUNKS[0]]
            # LLM 提取返回值
            mock_extract.return_value = "27V"

            params_spec = [("VBR", {"unit": "V", "category": "electrical"})]
            values, failed = tool._locate_params_in_chunks(self.CHUNKS, params_spec)

            assert values == {"VBR": "27V"}
            assert failed == []
            # 验证 keyword_rank 被调用，llm_guided 未被调用
            mock_kw.assert_called_once()
            mock_llm_search.assert_not_called()

    def test_keyword_miss_llm_hit(self):
        """
        关键词匹配未找到 → 降级到 _llm_guided_search 找到
        + LLM 提取成功 → values 包含该参数
        """
        tool = _make_tool()

        with (
            patch.object(tool, "_keyword_rank_chunks") as mock_kw,
            patch.object(tool, "_llm_guided_search") as mock_llm_search,
            patch.object(tool, "_extract_value_with_llm") as mock_extract,
        ):
            # keyword_rank 未找到
            mock_kw.return_value = []
            # llm_guided 找到候选
            mock_llm_search.return_value = [self.CHUNKS[0]]
            # LLM 提取返回值
            mock_extract.return_value = "27V"

            params_spec = [("VBR", {"unit": "V", "category": "electrical"})]
            values, failed = tool._locate_params_in_chunks(self.CHUNKS, params_spec)

            assert values == {"VBR": "27V"}
            assert failed == []
            mock_llm_search.assert_called_once()

    def test_all_fail(self):
        """
        关键词 + LLM 搜索均失败 → 参数被加入 failed 列表
        """
        tool = _make_tool()

        with (
            patch.object(tool, "_keyword_rank_chunks") as mock_kw,
            patch.object(tool, "_llm_guided_search") as mock_llm_search,
            patch.object(tool, "_extract_value_with_llm") as mock_extract,
        ):
            mock_kw.return_value = []
            mock_llm_search.return_value = []

            params_spec = [("VBR", {"unit": "V", "category": "electrical"})]
            values, failed = tool._locate_params_in_chunks(self.CHUNKS, params_spec)

            assert values == {}
            assert failed == ["VBR"]
            # extract 不应该被调用（因为没有候选）
            mock_extract.assert_not_called()

    def test_keyword_hit_but_extract_fails(self):
        """
        关键词匹配找到候选，但 LLM 提取失败
        → 参数被加入 failed 列表
        """
        tool = _make_tool()

        with (
            patch.object(tool, "_keyword_rank_chunks") as mock_kw,
            patch.object(tool, "_llm_guided_search") as mock_llm_search,
            patch.object(tool, "_extract_value_with_llm") as mock_extract,
        ):
            mock_kw.return_value = [self.CHUNKS[0]]
            mock_extract.return_value = None  # 提取失败

            params_spec = [("VBR", {"unit": "V", "category": "electrical"})]
            values, failed = tool._locate_params_in_chunks(self.CHUNKS, params_spec)

            assert values == {}
            assert failed == ["VBR"]

    def test_multiple_params_mixed_results(self):
        """
        多个参数混合场景：
        - param1: keyword hit + extract success
        - param2: keyword miss + llm_guided hit + extract success
        - param3: keyword miss + llm_guided miss → failed
        """
        tool = _make_tool()

        with (
            patch.object(tool, "_keyword_rank_chunks") as mock_kw,
            patch.object(tool, "_llm_guided_search") as mock_llm_search,
            patch.object(tool, "_extract_value_with_llm") as mock_extract,
        ):
            # keyword_rank: 对 param1 返回候选，对 param2/3 返回空
            def kw_side_effect(chunks, keyword, top_k=5):
                if keyword == "VBR":
                    return [self.CHUNKS[0]]
                return []

            mock_kw.side_effect = kw_side_effect

            # llm_guided: 对 param2 返回候选，对 param3 返回空
            def llm_side_effect(chunks, param_name, spec, top_k=3):
                if param_name == "IR":
                    return [self.CHUNKS[2]]
                return []

            mock_llm_search.side_effect = llm_side_effect

            # extract: 对 VBR 返回 "27V"，对 IR 返回 "1uA"
            def extract_side_effect(param_name, spec, candidates):
                mapping = {"VBR": "27V", "IR": "1uA"}
                return mapping.get(param_name)

            mock_extract.side_effect = extract_side_effect

            params_spec = [
                ("VBR", {"unit": "V", "category": "electrical"}),
                ("IR", {"unit": "uA", "category": "electrical"}),
                ("TYPE", {"category": "package"}),
            ]
            values, failed = tool._locate_params_in_chunks(self.CHUNKS, params_spec)

            assert values == {"VBR": "27V", "IR": "1uA"}
            assert failed == ["TYPE"]


# ============================================================
# TestCompareWithTemplate
# ============================================================

class TestCompareWithTemplate:
    """测试 compare_with_template 完整流程（Mock rag_service + 内部方法）"""

    def test_basic(self):
        """
        基本场景：两份文档均有所有参数值且完全一致
        → conclusion 为 "可替换"
        """
        rag_service = MagicMock()
        # mock get_document_chunks 返回简单数据
        rag_service.get_document_chunks.return_value = [
            {"searchable_text": "VBR=27V, type=SOD-323", "content": "", "page": "1",
             "section": "Electrical"},
        ]

        tool = _make_tool(rag_service=rag_service)

        with (
            patch.object(tool, "_locate_params_in_chunks") as mock_locate,
        ):
            # 两份文档都返回相同值
            mock_locate.return_value = ({"VBR": "27V", "type": "SOD-323"}, [])

            template = {
                "electrical": {"VBR": {"unit": "V"}},
                "package": {"type": {"value": "SOD-323"}},
            }
            result = tool.compare_with_template("doc_a", "doc_b", template,
                                                strict_mode=True)

            assert result["conclusion"] == "可替换"
            assert result["failed_params"] == []

    def test_with_failed_params(self):
        """
        参数定位失败场景：
        部分参数在文档中找不到 → failed_params 非空
        """
        rag_service = MagicMock()
        rag_service.get_document_chunks.return_value = [
            {"searchable_text": "some text", "content": "", "page": "1",
             "section": "Other"},
        ]

        tool = _make_tool(rag_service=rag_service)

        with (
            patch.object(tool, "_locate_params_in_chunks") as mock_locate,
        ):
            # doc_a 没有 IR，doc_b 没有 VBR
            def locate_side_effect(chunks, params_spec):
                doc_id = chunks[0].get("searchable_text", "") if chunks else ""
                if "doc_a" in str(id(chunks)):
                    return {"VBR": "27V"}, ["IR"]
                return {"IR": "1uA"}, ["VBR"]

            # 使用更直接的方式：根据 chunks 内容判断
            call_count = [0]

            def locate_side_effect(chunks, params_spec):
                call_count[0] += 1
                if call_count[0] == 1:
                    # doc_a 调用
                    return ({"VBR": "27V"}, ["IR"])
                else:
                    # doc_b 调用
                    return ({"IR": "1uA"}, ["VBR"])

            mock_locate.side_effect = locate_side_effect

            template = {
                "electrical": {"VBR": {"unit": "V"}, "IR": {"unit": "uA"}},
            }
            result = tool.compare_with_template("doc_a", "doc_b", template,
                                                strict_mode=True)

            # 两份文档的 failed 合并去重
            assert set(result["failed_params"]) == {"VBR", "IR"}
            assert result["conclusion"] != ""  # 有结论即可

    def test_empty_template(self):
        """
        空模板 → conclusion 为 "不可替换"
        """
        tool = _make_tool()
        result = tool.compare_with_template("doc_a", "doc_b", {},
                                            strict_mode=True)
        assert result["conclusion"] == "不可替换"
        assert "参数模板为空" in result["conclusion_detail"]
