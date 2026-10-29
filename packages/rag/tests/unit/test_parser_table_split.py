# Test values are synthetic fixtures for software behavior.
"""表格 chunk 切分回归测试：厚表格必须按 token/字符上限切成多块。"""
from __future__ import annotations

from core.parser import DocumentParser
from config import config


def _make_parser(monkeypatch) -> DocumentParser:
    parser = DocumentParser()
    # 避免触发真实 LLM 表格摘要调用
    monkeypatch.setattr(parser, "_annotate_table", lambda text: "")
    return parser


def test_oversized_table_split_into_multiple_chunks(monkeypatch):
    """单表格超大（含超长行）时必须切成多块，且每块不超过字符硬上限。"""
    parser = _make_parser(monkeypatch)
    table_parts = [
        "| " + "A" * 6000,  # 单行超长，需硬切
    ] + [
        "| row %d | " % i + "x" * 700 + " |" for i in range(20)
    ]
    chunks = parser._build_table_child_chunk(
        table_parts=table_parts,
        text_parts=[],
        cur_section="",
        cur_page=1,
        parent_id="parent-1",
        doc_id="doc-1",
        file_name="table_only.docx",
        heading_level=0,
        markdown_table="\n\n".join(table_parts),
    )
    assert len(chunks) > 3
    for chunk in chunks:
        # 首块含 [原始表格] 固定前缀，允许少量固定开销
        assert len(chunk.content) <= config.TABLE_CHUNK_MAX_CHARS + 32
    assert chunks[0].metadata["table_part_total"] == len(chunks)
    assert all(c.metadata["table_part"] == i + 1 for i, c in enumerate(chunks))


def test_small_table_keeps_single_chunk(monkeypatch):
    """小表格保持单块，不额外拆分。"""
    parser = _make_parser(monkeypatch)
    table_parts = ["| a | b |\n|---|---|\n| 1 | 2 |"]
    chunks = parser._build_table_child_chunk(
        table_parts=table_parts,
        text_parts=[],
        cur_section="",
        cur_page=1,
        parent_id="parent-1",
        doc_id="doc-1",
        file_name="small.docx",
        heading_level=0,
        markdown_table=table_parts[0],
    )
    assert len(chunks) == 1
    assert chunks[0].metadata["table_part_total"] == 1
