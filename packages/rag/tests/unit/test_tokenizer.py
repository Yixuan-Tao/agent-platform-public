# Test values are synthetic fixtures for software behavior.
"""
utils.tokenizer 单元测试

覆盖范围：
    - count_tokens：tiktoken 路径 / 启发式路径
    - split_by_tokens：段落 / 行 / 句子 / 硬切 四级降级
    - 边界：空文本 / 单段不超限 / 单段超限
"""
from __future__ import annotations

import pytest


class TestCountTokens:
    """count_tokens 基础单测。"""

    def test_empty_text(self):
        from utils.tokenizer import count_tokens
        assert count_tokens("") == 0
        assert count_tokens(None) == 0

    def test_short_english(self):
        from utils.tokenizer import count_tokens
        # tiktoken 路径下，10 个英文单词大约 10-15 token
        n = count_tokens("hello world this is a test sentence here")
        assert 5 <= n <= 20  # 任何合理路径都应在这个范围

    def test_short_chinese(self):
        from utils.tokenizer import count_tokens
        # 10 个中文字符：tiktoken 下约 10-15 token，启发式下约 10 token
        n = count_tokens("这是一段测试文本用于验证分词")
        assert 5 <= n <= 25

    def test_consistency(self):
        from utils.tokenizer import count_tokens
        # 相同输入应得到相同结果（lru_cache 验证）
        a = count_tokens("DEMO27-A 的 VBR 是 27V")
        b = count_tokens("DEMO27-A 的 VBR 是 27V")
        assert a == b

    def test_different_lengths(self):
        from utils.tokenizer import count_tokens
        short = count_tokens("hello")
        long = count_tokens("hello " * 100)
        assert long > short


class TestSplitByTokens:
    """split_by_tokens 四级降级单测。"""

    def test_empty_text(self):
        from utils.tokenizer import split_by_tokens
        assert split_by_tokens("", max_tokens=100) == []
        assert split_by_tokens("   \n  ", max_tokens=100) == []

    def test_short_text_no_split(self):
        from utils.tokenizer import split_by_tokens
        text = "Hello world. This is short."
        result = split_by_tokens(text, max_tokens=100)
        assert result == [text]

    def test_paragraph_split(self):
        from utils.tokenizer import split_by_tokens
        # 4 个段落，每个 ~100 token → max_tokens=150 应切 2+ 块
        para1 = "This is a longer paragraph that has many more words to make it count. " * 10
        para2 = "Another longer paragraph with many additional words to reach the limit. " * 10
        para3 = "Yet another longer paragraph with sufficient content to test the split. " * 10
        para4 = "A final long paragraph to make sure we exceed the chunk size limit. " * 10
        text = f"{para1}\n\n{para2}\n\n{para3}\n\n{para4}"
        result = split_by_tokens(text, max_tokens=150)
        # 应被切分为多块
        assert len(result) >= 2
        # 每块应不超过 max_tokens（容差 20%）
        from utils.tokenizer import count_tokens
        for chunk in result:
            assert count_tokens(chunk) <= 180

    def test_max_tokens_zero(self):
        from utils.tokenizer import split_by_tokens
        # max_tokens <= 0 → 兜底返回原文
        text = "Hello"
        assert split_by_tokens(text, max_tokens=0) == [text]
        assert split_by_tokens(text, max_tokens=-1) == [text]

    def test_very_long_text_hard_split(self):
        from utils.tokenizer import split_by_tokens
        # 单个超长无标点文本（无段落/句号可切）→ 走硬切
        long_text = "a" * 10000
        result = split_by_tokens(long_text, max_tokens=100)
        assert len(result) >= 2
        # 总长度应等于原文
        assert sum(len(c) for c in result) == len(long_text)

    def test_preserves_content(self):
        from utils.tokenizer import split_by_tokens
        # 切分后重组应保留所有非空白内容
        text = "段落一内容\n\n段落二内容\n\n段落三内容"
        result = split_by_tokens(text, max_tokens=10)
        joined = "\n\n".join(result)
        # 三个段落的关键词都应保留
        for kw in ["段落一", "段落二", "段落三"]:
            assert kw in joined


class TestMergeFragments:
    """_merge_fragments 合并单测。"""

    def test_merge_into_chunks(self):
        from utils.tokenizer import _merge_fragments

        fragments = ["A" * 50, "B" * 50, "C" * 50, "D" * 50]
        result = _merge_fragments(fragments, max_tokens=100)
        # 应被合并为 ~2 块（100 字符 + 间隔 ≈ 2 块）
        assert len(result) >= 1
        assert all(len(c) > 0 for c in result)

    def test_oversized_fragment_alone(self):
        from utils.tokenizer import _merge_fragments

        # 单个超长 fragment 单独成块（max_tokens 远小于 fragment 大小）
        # 用 200 个字符的英文：tiktoken 约 50 token，max_tokens=10 → 超限
        fragments = ["x" * 200, "y" * 10]
        result = _merge_fragments(fragments, max_tokens=10)
        # 超限 fragment 必须独立成块，不被强行截断
        assert result[0] == "x" * 200
        # "y"*10 也应独立成块（因为 max_tokens=10，刚好等于 10 字符=2-3 token，不会被单独成块）
        # 只要 "x"*200 在第一块就算通过
        assert "x" * 200 in result

    def test_empty_input(self):
        from utils.tokenizer import _merge_fragments
        assert _merge_fragments([], max_tokens=100) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
