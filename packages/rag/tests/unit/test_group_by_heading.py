# Test values are synthetic fixtures for software behavior.
"""
切块改造单元测试

覆盖范围：
    - _group_by_heading：跨页同标题合并 / 开关
    - _build_parent_chunk：父子双层 / 表格独立 / token 截断
    - DocumentChunk：chunk_role / is_table_only 字段
    - Retriever._link_related：parent_context 注入
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def parser():
    """构造一个 DocumentParser 实例（mock 掉网络/磁盘依赖）。"""
    from core.parser import DocumentParser, _RawElement
    p = DocumentParser()
    # Mock 网络/磁盘方法
    p._vision_understand = MagicMock(return_value="测试图片描述")
    p._annotate_image = MagicMock(side_effect=lambda desc, **kw: desc)
    p._annotate_table = MagicMock(return_value="ParamA: 1V\nParamB: 2A")
    return p


def _heading(level: int, content: str, page: int = 1):
    from core.parser import _RawElement
    return _RawElement(
        type="heading", level=level, content=content, page=page,
    )


def _text(content: str, page: int = 1):
    from core.parser import _RawElement
    return _RawElement(type="text", level=0, content=content, page=page)


def _table(content: str, page: int = 1):
    from core.parser import _RawElement
    return _RawElement(type="table", level=0, content=content, page=page)


# ============================================================
# _group_by_heading
# ============================================================

class TestGroupByHeading:
    def _patch_config(self, monkeypatch, **kwargs):
        from config import config
        for k, v in kwargs.items():
            monkeypatch.setattr(config, k, v)

    def test_simple_sections(self, parser, monkeypatch):
        """H1 切分为独立 section。"""
        from config import config
        self._patch_config(monkeypatch, ENABLE_HEADING_MERGE=True, CHUNK_HEADING_MIN_LEVEL=2)
        elements = [
            _heading(1, "第一章", page=1),
            _text("内容 A", page=1),
            _heading(1, "第二章", page=2),
            _text("内容 B", page=2),
        ]
        groups = parser._group_by_heading(elements, "doc1", "test.pdf")
        assert len(groups) == 2
        assert groups[0].section == "第一章"
        assert groups[1].section == "第二章"
        assert groups[0].start_page == 1
        assert groups[1].start_page == 2

    def test_h1_h2_split(self, parser, monkeypatch):
        """H1+H2 切分（H1 整体 + H2 子节）。"""
        from config import config
        self._patch_config(monkeypatch, ENABLE_HEADING_MERGE=True, CHUNK_HEADING_MIN_LEVEL=2)
        elements = [
            _heading(1, "第一章 概述", page=1),
            _text("概述正文", page=1),
            _heading(2, "1.1 背景", page=1),
            _text("背景正文", page=1),
            _heading(1, "第二章 详细", page=2),
            _text("详细正文", page=2),
        ]
        groups = parser._group_by_heading(elements, "doc1", "test.pdf")
        # H1 切 2 组，H2 累积到当前 H1
        assert len(groups) == 2
        # 第二组有 H2 "1.1 背景" 在 elements 内
        section_titles = [g.section for g in groups]
        assert "第一章 概述" in section_titles
        assert "第二章 详细" in section_titles

    def test_cross_page_merge(self, parser, monkeypatch):
        """跨页同 H1 标题应合并。"""
        from config import config
        self._patch_config(monkeypatch, ENABLE_HEADING_MERGE=True, CHUNK_HEADING_MIN_LEVEL=2)
        elements = [
            _heading(1, "产品概述", page=1),
            _text("第一页内容", page=1),
            _heading(1, "产品概述", page=2),  # 跨页同名
            _text("第二页内容", page=2),
            _heading(1, "应用电路", page=3),
            _text("第三页内容", page=3),
        ]
        groups = parser._group_by_heading(elements, "doc1", "test.pdf")
        # 同名 "产品概述" 跨页合并 → 2 组而非 3 组
        assert len(groups) == 2
        assert groups[0].section == "产品概述"
        assert groups[0].start_page == 1
        assert groups[0].end_page == 2
        assert groups[1].section == "应用电路"

    def test_disable_merge(self, parser, monkeypatch):
        """关闭合并：每个 heading 独立成 group。"""
        from config import config
        self._patch_config(monkeypatch, ENABLE_HEADING_MERGE=False, CHUNK_HEADING_MIN_LEVEL=2)
        elements = [
            _heading(1, "产品概述", page=1),
            _text("第一页内容", page=1),
            _heading(1, "产品概述", page=2),  # 即使同名也不合并
            _text("第二页内容", page=2),
        ]
        groups = parser._group_by_heading(elements, "doc1", "test.pdf")
        # 关闭合并 → 2 个独立 group
        assert len(groups) == 2

    def test_no_heading(self, parser, monkeypatch):
        """文档无 heading → 1 个虚拟 group。"""
        from config import config
        self._patch_config(monkeypatch, ENABLE_HEADING_MERGE=True, CHUNK_HEADING_MIN_LEVEL=2)
        elements = [
            _text("直接是文本内容", page=1),
            _text("继续文本", page=1),
        ]
        groups = parser._group_by_heading(elements, "doc1", "test.pdf")
        assert len(groups) == 1
        assert groups[0].section == ""


# ============================================================
# _build_parent_chunk
# ============================================================

class TestBuildParentChunk:
    def _patch_config(self, monkeypatch, **kwargs):
        from config import config
        for k, v in kwargs.items():
            monkeypatch.setattr(config, k, v)

    def test_text_only_section(self, parser, monkeypatch):
        """仅文本 section：1 个 parent + 若干 text child。"""
        from config import config
        self._patch_config(
            monkeypatch, CHUNK_HEADING_MIN_LEVEL=2, CHUNK_MAX_TOKENS=100,
            PARENT_MAX_TOKENS=500, ENABLE_TABLE_ISOLATION=True,
        )
        group = parser._SectionGroup(
            section="产品概述", level=1, start_page=1, end_page=1,
            elements=[_heading(1, "产品概述"), _text("内容A"), _text("内容B")],
        )
        parent, children = parser._build_parent_chunk(
            group, "doc1", "test.pdf", image_descriptions={},
        )
        assert parent.chunk_role == "parent"
        assert parent.section == "产品概述"
        assert all(c.chunk_role == "child" for c in children)
        # text child 全部 modality="text"
        text_children = [c for c in children if c.modality == "text"]
        assert len(text_children) >= 1
        # parent 不入向量库（默认）
        assert parent.to_chroma_entity() is None

    def test_table_isolation(self, parser, monkeypatch):
        """表格独立成 child chunk，is_table_only=True。"""
        from config import config
        self._patch_config(
            monkeypatch, CHUNK_HEADING_MIN_LEVEL=2, CHUNK_MAX_TOKENS=100,
            PARENT_MAX_TOKENS=500, ENABLE_TABLE_ISOLATION=True,
        )
        table_md = "| Param | Value |\n|-------|-------|\n| VBR   | 27V   |"
        group = parser._SectionGroup(
            section="电气特性", level=1, start_page=2, end_page=2,
            elements=[
                _heading(1, "电气特性"),
                _text("以下是参数表"),
                _table(table_md),
            ],
        )
        parent, children = parser._build_parent_chunk(
            group, "doc1", "test.pdf", image_descriptions={},
        )
        table_children = [c for c in children if c.is_table_only]
        text_children = [c for c in children if c.modality == "text" and not c.is_table_only]
        assert len(table_children) == 1
        assert table_children[0].modality == "table"
        assert table_children[0].is_table_only is True
        assert table_children[0].parent_chunk_id == parent.chunk_id
        assert len(text_children) >= 1
        # 表格内容应含 [原始表格] 前缀
        assert "[原始表格]" in table_children[0].content

    def test_table_isolation_disabled(self, parser, monkeypatch):
        """关闭表格独立 → 不生成独立 table child。"""
        from config import config
        self._patch_config(
            monkeypatch, CHUNK_HEADING_MIN_LEVEL=2, CHUNK_MAX_TOKENS=100,
            PARENT_MAX_TOKENS=500, ENABLE_TABLE_ISOLATION=False,
        )
        group = parser._SectionGroup(
            section="节A", level=1, start_page=1, end_page=1,
            elements=[_heading(1, "节A"), _text("文本"), _table("|A|1|")],
        )
        parent, children = parser._build_parent_chunk(
            group, "doc1", "test.pdf", image_descriptions={},
        )
        table_children = [c for c in children if c.is_table_only]
        assert len(table_children) == 0

    def test_text_token_splitting(self, parser, monkeypatch):
        """超长文本按 token 切分为多个 child。"""
        from config import config
        self._patch_config(
            monkeypatch, CHUNK_HEADING_MIN_LEVEL=2, CHUNK_MAX_TOKENS=50,
            PARENT_MAX_TOKENS=500, ENABLE_TABLE_ISOLATION=True,
        )
        long_text = "测试文本内容。" * 100  # 100 个 token 左右
        group = parser._SectionGroup(
            section="长文", level=1, start_page=1, end_page=1,
            elements=[_heading(1, "长文"), _text(long_text)],
        )
        parent, children = parser._build_parent_chunk(
            group, "doc1", "test.pdf", image_descriptions={},
        )
        text_children = [c for c in children if c.modality == "text"]
        # 应被切分为多个 child chunk
        assert len(text_children) >= 2
        # 每个 child 的 token 数应不超过 CHUNK_MAX_TOKENS（容差 20%）
        from utils.tokenizer import count_tokens
        for c in text_children:
            assert count_tokens(c.content) <= 60

    def test_parent_token_limit(self, parser, monkeypatch):
        """parent content 超过 PARENT_MAX_TOKENS 时应截断。"""
        from config import config
        self._patch_config(
            monkeypatch, CHUNK_HEADING_MIN_LEVEL=2, CHUNK_MAX_TOKENS=200,
            PARENT_MAX_TOKENS=50, ENABLE_TABLE_ISOLATION=True,
        )
        long_text = "这是很长的章节内容。" * 50
        group = parser._SectionGroup(
            section="巨长章节", level=1, start_page=1, end_page=1,
            elements=[_heading(1, "巨长章节"), _text(long_text)],
        )
        parent, children = parser._build_parent_chunk(
            group, "doc1", "test.pdf", image_descriptions={},
        )
        from utils.tokenizer import count_tokens
        # parent content 应被截断
        assert count_tokens(parent.content) <= 60

    def test_image_child(self, parser, monkeypatch):
        """image 元素生成独立 image child，挂回 parent。"""
        from config import config
        from core.parser import _RawElement
        self._patch_config(
            monkeypatch, CHUNK_HEADING_MIN_LEVEL=2, CHUNK_MAX_TOKENS=100,
            PARENT_MAX_TOKENS=500, ENABLE_TABLE_ISOLATION=True,
        )
        # 3KB 图片（> 2KB 装饰图阈值）
        image_bytes = b"\\x89PNG\\r\\n\\x1a\\n" + b"\\x00" * 3000
        img_el = _RawElement(
            type="image", level=0, content="", page=1,
            image_bytes=image_bytes, image_format="png", image_id="img1",
        )
        group = parser._SectionGroup(
            section="外观图", level=1, start_page=1, end_page=1,
            elements=[_heading(1, "外观图"), img_el],
        )
        parent, children = parser._build_parent_chunk(
            group, "doc1", "test.pdf",
            image_descriptions={"img1": "产品外观示意图"},
        )
        image_children = [c for c in children if c.modality == "image"]
        assert len(image_children) == 1
        assert image_children[0].parent_chunk_id == parent.chunk_id
        # parent.child_images 应含 image child id
        assert image_children[0].chunk_id in parent.child_images

    def test_parent_in_index_disabled_by_default(self, parser, monkeypatch):
        """默认 INCLUDE_PARENT_IN_INDEX=False → parent 不入向量库。"""
        from config import config
        from db.schemas import CHUNK_ROLE_CHILD, CHUNK_ROLE_PARENT
        self._patch_config(
            monkeypatch, CHUNK_HEADING_MIN_LEVEL=2, CHUNK_MAX_TOKENS=100,
            PARENT_MAX_TOKENS=500, ENABLE_TABLE_ISOLATION=True,
            INCLUDE_PARENT_IN_INDEX=False,
        )
        group = parser._SectionGroup(
            section="A", level=1, start_page=1, end_page=1,
            elements=[_heading(1, "A"), _text("内容")],
        )
        parent, _ = parser._build_parent_chunk(
            group, "doc1", "test.pdf", image_descriptions={},
        )
        # parent.to_chroma_entity() 应返回 None（不入库）
        assert parent.to_chroma_entity() is None
        # child chunk 应正常入库
        from db.schemas import DocumentChunk
        child = DocumentChunk(
            id="c1", doc_id="doc1", file_name="test.pdf",
            modality="text", content="x", searchable_text="x",
            chunk_role=CHUNK_ROLE_CHILD,
        )
        assert child.to_chroma_entity() is not None


# ============================================================
# DocumentChunk chunk_role
# ============================================================

class TestDocumentChunkRole:
    def test_default_role_is_child(self):
        from db.schemas import DocumentChunk, CHUNK_ROLE_CHILD
        c = DocumentChunk(
            id="c1", doc_id="d1", file_name="f.pdf",
            modality="text", content="x", searchable_text="x",
        )
        assert c.chunk_role == CHUNK_ROLE_CHILD
        assert c.is_table_only is False
        assert c.parent_chunk_id == ""

    def test_to_chroma_entity_returns_none_for_parent(self):
        from db.schemas import DocumentChunk, CHUNK_ROLE_PARENT
        c = DocumentChunk(
            id="p1", doc_id="d1", file_name="f.pdf",
            modality="mixed", content="x", searchable_text="x",
            chunk_role=CHUNK_ROLE_PARENT,
        )
        assert c.to_chroma_entity() is None

    def test_metadata_includes_role_and_parent(self):
        from db.schemas import DocumentChunk, CHUNK_ROLE_CHILD
        c = DocumentChunk(
            id="c1", doc_id="d1", file_name="f.pdf",
            modality="text", content="x", searchable_text="x",
            chunk_role=CHUNK_ROLE_CHILD, parent_chunk_id="p1",
            is_table_only=True,
        )
        meta = c._build_chroma_metadata()
        assert meta["chunk_role"] == CHUNK_ROLE_CHILD
        assert meta["parent_chunk_id"] == "p1"
        assert meta["is_table_only"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
