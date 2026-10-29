# Test values are synthetic fixtures for software behavior.
"""
tools.base 工具函数单元测试

覆盖范围：
    - format_chunks_for_llm：分页排序 / 去重 / 缺字段
    - safe_parse_json：markdown 包裹 / 异常 / 截断
    - extract_doc_ids：命中计数 / 排序 / 空 ID 过滤
    - collect_image_chunks：顶层 / linked_chunks / 去重
    - source_summary：240 字截断 / 缺字段
    - escape_filter_value：双引号转义
"""
from __future__ import annotations

import pytest


# ============================================================
# format_chunks_for_llm
# ============================================================

class TestFormatChunksForLlm:
    """format_chunks_for_llm：将 chunks 格式化为 LLM 可读文本。"""

    def test_empty_list(self):
        """空列表应返回"（文档无内容）"。"""
        from tools.base import format_chunks_for_llm
        assert format_chunks_for_llm([]) == "（文档无内容）"

    def test_single_chunk(self):
        """单个 chunk 应正确格式化。"""
        from tools.base import format_chunks_for_llm
        chunks = [{"content": "测试内容", "page": 3}]
        result = format_chunks_for_llm(chunks)
        assert result == "[p.3] 测试内容"

    def test_multiple_chunks_sorted_by_page(self):
        """多个 chunk 应按 page 升序排列。"""
        from tools.base import format_chunks_for_llm
        chunks = [
            {"content": "第二页", "page": 2},
            {"content": "第一页", "page": 1},
            {"content": "第三页", "page": 3},
        ]
        result = format_chunks_for_llm(chunks)
        lines = result.split("\n\n")
        assert len(lines) == 3
        assert lines[0] == "[p.1] 第一页"
        assert lines[1] == "[p.2] 第二页"
        assert lines[2] == "[p.3] 第三页"

    def test_duplicate_content_dedup(self):
        """重复内容应被去重，只保留第一个出现的。"""
        from tools.base import format_chunks_for_llm
        chunks = [
            {"content": "重复内容", "page": 1},
            {"content": "重复内容", "page": 2},
            {"content": "唯一内容", "page": 3},
        ]
        result = format_chunks_for_llm(chunks)
        # 去重后应只有 2 条
        lines = result.split("\n\n")
        assert len(lines) == 2
        assert "[p.1] 重复内容" in lines
        assert "[p.3] 唯一内容" in lines

    def test_missing_page_field(self):
        """缺失 page 字段时，默认 page 显示为 '?'。"""
        from tools.base import format_chunks_for_llm
        chunks = [{"content": "无页码内容"}]
        result = format_chunks_for_llm(chunks)
        assert result == "[p.?] 无页码内容"

    def test_missing_content_field(self):
        """缺失 content 字段或 content 为空时，应跳过该 chunk。"""
        from tools.base import format_chunks_for_llm
        chunks = [
            {"page": 1},                     # 无 content
            {"content": None, "page": 2},    # content 为 None
            {"content": "", "page": 3},      # content 为空字符串
            {"content": "   ", "page": 4},   # content 为空白
            {"content": "有效内容", "page": 5},
        ]
        result = format_chunks_for_llm(chunks)
        assert result == "[p.5] 有效内容"


# ============================================================
# safe_parse_json
# ============================================================

class TestSafeParseJson:
    """safe_parse_json：安全解析 JSON，处理 LLM 输出的各种格式。"""

    def test_empty_string(self):
        """空字符串应返回空 dict。"""
        from tools.base import safe_parse_json
        assert safe_parse_json("") == {}

    def test_valid_json(self):
        """有效 JSON 应正确解析为 dict。"""
        from tools.base import safe_parse_json
        assert safe_parse_json('{"a": 1, "b": "hello"}') == {"a": 1, "b": "hello"}

    def test_json_with_lang_wrapper(self):
        """```json 代码块包裹的 JSON 应正确提取。"""
        from tools.base import safe_parse_json
        content = '```json\n{"a": 1}\n```'
        assert safe_parse_json(content) == {"a": 1}

    def test_json_with_plain_wrapper(self):
        """```（无语言标记）代码块包裹的 JSON 应正确提取。"""
        from tools.base import safe_parse_json
        content = '```\n{"a": 1}\n```'
        assert safe_parse_json(content) == {"a": 1}

    def test_invalid_json(self):
        """无效 JSON 字符串应返回空 dict。"""
        from tools.base import safe_parse_json
        assert safe_parse_json("not json") == {}

    def test_list_instead_of_dict(self):
        """JSON 数组（list）应返回空 dict。"""
        from tools.base import safe_parse_json
        assert safe_parse_json("[1, 2, 3]") == {}

    def test_truncated_json(self):
        """截断的 JSON 应返回空 dict。"""
        from tools.base import safe_parse_json
        # 缺少闭合括号
        assert safe_parse_json('{"a": 1, "b": 2') == {}
        # 字符串未闭合
        assert safe_parse_json('{"a": "unclosed') == {}

    def test_json_with_extra_whitespace(self):
        """前后有额外空白的 JSON 应正常解析。"""
        from tools.base import safe_parse_json
        assert safe_parse_json('  \n  {"a": 1}  \n  ') == {"a": 1}

    def test_nested_json(self):
        """嵌套 JSON 应正确解析。"""
        from tools.base import safe_parse_json
        content = '{"a": {"b": [1, 2, 3]}, "c": null}'
        assert safe_parse_json(content) == {"a": {"b": [1, 2, 3]}, "c": None}


# ============================================================
# extract_doc_ids
# ============================================================

class TestExtractDocIds:
    """extract_doc_ids：从候选列表中提取 doc_id，按命中数降序。"""

    def test_empty_list(self):
        """空列表应返回空列表。"""
        from tools.base import extract_doc_ids
        assert extract_doc_ids([]) == []

    def test_single_doc_multiple_hits(self):
        """同一文档多次命中应只返回一次该 doc_id。"""
        from tools.base import extract_doc_ids
        candidates = [
            {"doc_id": "doc1", "content": "a"},
            {"doc_id": "doc1", "content": "b"},
            {"doc_id": "doc1", "content": "c"},
        ]
        assert extract_doc_ids(candidates) == ["doc1"]

    def test_multiple_docs_sorted_by_count_desc(self):
        """不同文档按命中 chunk 数降序排列。"""
        from tools.base import extract_doc_ids
        candidates = [
            {"doc_id": "doc_a", "content": "a1"},
            {"doc_id": "doc_b", "content": "b1"},
            {"doc_id": "doc_b", "content": "b2"},
            {"doc_id": "doc_b", "content": "b3"},
            {"doc_id": "doc_c", "content": "c1"},
            {"doc_id": "doc_c", "content": "c2"},
        ]
        # doc_b: 3 次, doc_c: 2 次, doc_a: 1 次
        result = extract_doc_ids(candidates)
        assert result[0] == "doc_b"
        assert result[1] == "doc_c"
        assert result[2] == "doc_a"

    def test_empty_doc_id_filtered(self):
        """doc_id 为空或 None 的条目应被过滤。"""
        from tools.base import extract_doc_ids
        candidates = [
            {"doc_id": "doc1", "content": "a"},
            {"doc_id": "", "content": "b"},
            {"doc_id": None, "content": "c"},
            {},  # 无 doc_id 字段
        ]
        assert extract_doc_ids(candidates) == ["doc1"]

    def test_tie_breaking_same_count(self):
        """相同命中数的文档，顺序任意（不报错即可）。"""
        from tools.base import extract_doc_ids
        candidates = [
            {"doc_id": "doc_x", "content": "x1"},
            {"doc_id": "doc_x", "content": "x2"},
            {"doc_id": "doc_y", "content": "y1"},
            {"doc_id": "doc_y", "content": "y2"},
        ]
        result = extract_doc_ids(candidates)
        # 两个 doc_id 都应出现，且各出现一次
        assert sorted(result) == ["doc_x", "doc_y"]


# ============================================================
# collect_image_chunks
# ============================================================

class TestCollectImageChunks:
    """collect_image_chunks：从 contexts 中收集图片 chunk（去重）。"""

    def test_empty_list(self):
        """空列表应返回空列表。"""
        from tools.base import collect_image_chunks
        assert collect_image_chunks([]) == []

    def test_text_chunks_only(self):
        """仅有文本 chunk（无图片）应返回空列表。"""
        from tools.base import collect_image_chunks
        contexts = [
            {"id": "c1", "modality": "text", "content": "文本内容"},
            {"id": "c2", "modality": "text", "content": "更多文本"},
        ]
        assert collect_image_chunks(contexts) == []

    def test_image_chunks_at_top_level(self):
        """顶层有 image 类型 chunk 应被收集。"""
        from tools.base import collect_image_chunks
        contexts = [
            {
                "id": "img1", "modality": "image",
                "content": "图片描述",
                "page": 1, "file_name": "test.pdf",
                "metadata": {"image_path": "/images/p1.png"},
            },
        ]
        result = collect_image_chunks(contexts)
        assert len(result) == 1
        assert result[0]["id"] == "img1"
        assert result[0]["image_path"] == "/images/p1.png"
        assert result[0]["description"] == "图片描述"

    def test_image_chunks_in_linked_chunks(self):
        """linked_chunks 中的 image 类型 chunk 应被收集。"""
        from tools.base import collect_image_chunks
        contexts = [
            {
                "id": "t1", "modality": "text",
                "content": "引用图片",
                "linked_chunks": [
                    {
                        "id": "img1", "modality": "image",
                        "content": "关联图片",
                        "page": 2, "file_name": "test.pdf",
                        "metadata": {"image_path": "/images/p2.png"},
                    },
                ],
            },
        ]
        result = collect_image_chunks(contexts)
        assert len(result) == 1
        assert result[0]["id"] == "img1"
        assert result[0]["image_path"] == "/images/p2.png"

    def test_duplicate_ids_deduped(self):
        """相同 id 的图片应被去重，优先取顶层。"""
        from tools.base import collect_image_chunks
        contexts = [
            {
                "id": "img1", "modality": "image",
                "content": "顶层图片",
                "page": 1, "file_name": "a.pdf",
                "metadata": {"image_path": "/images/a.png"},
            },
            {
                "id": "t1", "modality": "text",
                "content": "文本",
                "linked_chunks": [
                    {
                        "id": "img1", "modality": "image",
                        "content": "重复图片",
                        "page": 1, "file_name": "a.pdf",
                        "metadata": {"image_path": "/images/a.png"},
                    },
                ],
            },
        ]
        result = collect_image_chunks(contexts)
        assert len(result) == 1  # 去重后只有 1 条

    def test_missing_metadata_field(self):
        """缺失 metadata 字段不应导致异常。"""
        from tools.base import collect_image_chunks
        contexts = [
            {
                "id": "img1", "modality": "image",
                "content": "无 metadata 图片",
                "page": 1, "file_name": "test.pdf",
                # 没有 metadata 字段
            },
        ]
        result = collect_image_chunks(contexts)
        assert len(result) == 1
        assert result[0]["id"] == "img1"
        assert result[0]["image_path"] == ""  # 默认空字符串


# ============================================================
# source_summary
# ============================================================

class TestSourceSummary:
    """source_summary：生成检索结果来源摘要，截断长内容。"""

    def test_short_content_unchanged(self):
        """少于 240 字的内容应保持不变。"""
        from tools.base import source_summary
        hit = {
            "id": "c1",
            "file_name": "test.pdf",
            "page": 3,
            "modality": "text",
            "content": "简短内容",
            "score": 0.85,
        }
        result = source_summary(hit)
        assert result["content"] == "简短内容"
        assert result["id"] == "c1"
        assert result["page"] == 3
        assert result["score"] == 0.85

    def test_content_over_240_chars_truncated(self):
        """超过 240 字的内容应截断并追加 '...'。"""
        from tools.base import source_summary
        long_content = "A" * 300
        hit = {
            "id": "c2",
            "file_name": "doc.pdf",
            "page": 5,
            "modality": "text",
            "content": long_content,
            "score": 0.9,
        }
        result = source_summary(hit)
        assert len(result["content"]) == 243  # 240 字 + "..."
        assert result["content"].endswith("...")

    def test_empty_content(self):
        """空内容应返回空字符串。"""
        from tools.base import source_summary
        hit = {
            "id": "c3",
            "file_name": "test.pdf",
            "page": 1,
            "modality": "text",
            "content": "",
            "score": 0.0,
        }
        result = source_summary(hit)
        assert result["content"] == ""

    def test_missing_content_field(self):
        """缺失 content 字段应返回空字符串，不抛异常。"""
        from tools.base import source_summary
        hit = {
            "id": "c4",
            "file_name": "test.pdf",
            "page": 2,
            "modality": "text",
            "score": 0.5,
            # 没有 content 字段
        }
        result = source_summary(hit)
        assert result["content"] == ""

    def test_content_none(self):
        """content 为 None 应返回空字符串。"""
        from tools.base import source_summary
        hit = {
            "id": "c5",
            "file_name": "test.pdf",
            "page": 1,
            "modality": "text",
            "content": None,
            "score": 0.3,
        }
        result = source_summary(hit)
        assert result["content"] == ""


# ============================================================
# escape_filter_value
# ============================================================

class TestEscapeFilterValue:
    """escape_filter_value：转义 filter 表达式中的双引号。"""

    def test_no_quotes_unchanged(self):
        """无引号的字符串应原样返回。"""
        from tools.base import escape_filter_value
        assert escape_filter_value("hello world") == "hello world"

    def test_with_double_quotes_escaped(self):
        """包含双引号的字符串应将 " 转义为 \\"。"""
        from tools.base import escape_filter_value
        assert escape_filter_value('say "hello"') == 'say \\"hello\\"'

    def test_multiple_quotes(self):
        """多个双引号应全部转义。"""
        from tools.base import escape_filter_value
        assert escape_filter_value('a"b"c"d') == 'a\\"b\\"c\\"d'

    def test_empty_string(self):
        """空字符串应原样返回。"""
        from tools.base import escape_filter_value
        assert escape_filter_value("") == ""

    def test_only_quotes(self):
        """只有引号的字符串应全部转义。"""
        from tools.base import escape_filter_value
        assert escape_filter_value('"""') == '\\"\\"\\"'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
