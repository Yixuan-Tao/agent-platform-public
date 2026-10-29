# Test values are synthetic fixtures for software behavior.
"""
QueryRewriter 单元测试

覆盖范围：
    - QueryRewriter.rewrite：正常路径 / JSON 解析失败 / 不可重试异常
    - _parse_rewrites 静态方法：去重 / 截断 / 异常输入
    - get_colloquial_pattern：懒加载缓存
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ============================================================
# QueryRewriter._parse_rewrites
# ============================================================

class TestParseRewrites:
    """QueryRewriter._parse_rewrites 解析逻辑单测（不依赖网络）。"""

    def test_normal_json(self):
        from core.query_rewriter import QueryRewriter

        content = json.dumps({
            "rewrites": ["normalized", "variant1", "variant2"],
        })
        result = QueryRewriter._parse_rewrites(content, n_variants=2)
        assert result == ["normalized", "variant1", "variant2"]

    def test_truncate_to_expected_length(self):
        from core.query_rewriter import QueryRewriter

        # 超过 1 + n_variants → 截断
        content = json.dumps({
            "rewrites": ["a", "b", "c", "d", "e"],
        })
        result = QueryRewriter._parse_rewrites(content, n_variants=2)
        assert len(result) == 3

    def test_dedup(self):
        from core.query_rewriter import QueryRewriter

        content = json.dumps({
            "rewrites": ["a", "b", "a", "b", "c"],
        })
        result = QueryRewriter._parse_rewrites(content, n_variants=3)
        assert result == ["a", "b", "c"]

    def test_drop_empty_and_whitespace(self):
        from core.query_rewriter import QueryRewriter

        content = json.dumps({
            "rewrites": ["a", "", "   ", "b"],
        })
        result = QueryRewriter._parse_rewrites(content, n_variants=3)
        assert result == ["a", "b"]

    def test_invalid_json_returns_empty(self):
        from core.query_rewriter import QueryRewriter

        result = QueryRewriter._parse_rewrites("not a json", n_variants=2)
        assert result == []

    def test_empty_content_returns_empty(self):
        from core.query_rewriter import QueryRewriter

        assert QueryRewriter._parse_rewrites("", n_variants=2) == []
        assert QueryRewriter._parse_rewrites(None, n_variants=2) == []

    def test_wrong_shape_returns_empty(self):
        from core.query_rewriter import QueryRewriter

        # rewrites 不是 list
        assert QueryRewriter._parse_rewrites(
            json.dumps({"rewrites": "not a list"}), n_variants=2,
        ) == []
        # 顶层不是 dict
        assert QueryRewriter._parse_rewrites(
            json.dumps(["a", "b"]), n_variants=2,
        ) == []
        # 缺 rewrites 键
        assert QueryRewriter._parse_rewrites(
            json.dumps({"other": []}), n_variants=2,
        ) == []

    def test_non_string_items_filtered(self):
        from core.query_rewriter import QueryRewriter

        content = json.dumps({
            "rewrites": ["a", 123, None, "b", {"x": 1}],
        })
        result = QueryRewriter._parse_rewrites(content, n_variants=4)
        assert result == ["a", "b"]


# ============================================================
# QueryRewriter.rewrite
# ============================================================

class TestRewrite:
    """QueryRewriter.rewrite 集成测试（mock OpenAI 客户端）。"""

    def _make_mock_client(self, content: str):
        """构造一个 mock OpenAI 客户端，返回指定 content。"""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_resp
        return mock_client

    def test_rewrite_success(self):
        from core.query_rewriter import QueryRewriter

        mock_client = self._make_mock_client(
            json.dumps({"rewrites": ["钳位电压 Clamping Voltage VC", "Clamping Voltage"]}),
        )
        rw = QueryRewriter(client=mock_client)
        result = rw.rewrite("钳位电压是多少", language="zh", n_variants=1)
        # 第一个元素是标准化的"钳位电压 Clamping Voltage VC"
        assert result[0] == "钳位电压 Clamping Voltage VC"
        assert len(result) == 2
        # 验证调用参数
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.1
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_rewrite_n_variants_clamped(self):
        """n_variants 超过 2 会被截断到 2。"""
        from core.query_rewriter import QueryRewriter

        mock_client = self._make_mock_client(
            json.dumps({"rewrites": ["a", "b", "c", "d", "e"]}),
        )
        rw = QueryRewriter(client=mock_client)
        result = rw.rewrite("x", n_variants=10)  # 超过上限
        # n_variants 会被截断到 2，所以期望 3 条（1 标准化 + 2 变体）
        assert len(result) == 3

    def test_rewrite_json_parse_failure_fallback(self):
        """LLM 返回非 JSON → 降级为 [原 query]。"""
        from core.query_rewriter import QueryRewriter

        mock_client = self._make_mock_client("not a json")
        rw = QueryRewriter(client=mock_client)
        result = rw.rewrite("原 query 文本", n_variants=2)
        assert result == ["原 query 文本"]

    def test_rewrite_empty_response_fallback(self):
        from core.query_rewriter import QueryRewriter

        mock_client = self._make_mock_client("")
        rw = QueryRewriter(client=mock_client)
        result = rw.rewrite("原 query", n_variants=2)
        assert result == ["原 query"]

    def test_rewrite_non_retryable_exception_fallback(self):
        """非可重试异常 → 降级为 [原 query]，不抛错。"""
        from core.query_rewriter import QueryRewriter

        mock_client = MagicMock()
        # 抛一个非可重试的异常（不包含 429/5xx/connection）
        mock_client.chat.completions.create.side_effect = ValueError("bad input")
        rw = QueryRewriter(client=mock_client)
        result = rw.rewrite("原 query", n_variants=2)
        assert result == ["原 query"]

    def test_rewrite_empty_query(self):
        from core.query_rewriter import QueryRewriter

        mock_client = MagicMock()
        rw = QueryRewriter(client=mock_client)
        assert rw.rewrite("") == []
        assert rw.rewrite("   ") == []

    def test_rewrite_retryable_exception_propagates(self):
        """可重试异常（429）应该抛给装饰器处理。"""
        from core.query_rewriter import QueryRewriter

        mock_client = MagicMock()
        # 模拟 429 异常
        from openai import RateLimitError
        mock_client.chat.completions.create.side_effect = RateLimitError(
            "rate limit", response=MagicMock(status_code=429), body=None,
        )
        rw = QueryRewriter(client=mock_client)
        # 装饰器会重试 3 次都失败后抛出 RateLimitError
        with pytest.raises(Exception):
            rw.rewrite("原 query", n_variants=1)


# ============================================================
# get_colloquial_pattern
# ============================================================

class TestColloquialPattern:
    def test_match_chinese(self):
        from core.query_rewriter import get_colloquial_pattern

        pattern = get_colloquial_pattern()
        assert pattern.search("钳位电压是多少")
        assert pattern.search("为什么不能工作")
        assert pattern.search("这个怎么用")

    def test_match_english(self):
        from core.query_rewriter import get_colloquial_pattern

        pattern = get_colloquial_pattern()
        assert pattern.search("how does it work")
        assert pattern.search("what is the voltage")
        assert pattern.search("why does it fail")

    def test_not_match(self):
        from core.query_rewriter import get_colloquial_pattern

        pattern = get_colloquial_pattern()
        # 不包含任何口语化词的纯技术 query
        assert pattern.search("DEMO27-A 的 VBR 27V 测试条件 IT=1mA") is None
        # "是多少" 命中"多少" → 视为口语化
        assert pattern.search("VBR 是多少") is not None

    def test_lazy_cache(self):
        """连续调用应返回同一个对象。"""
        from core.query_rewriter import get_colloquial_pattern

        p1 = get_colloquial_pattern()
        p2 = get_colloquial_pattern()
        assert p1 is p2


# ============================================================
# QueryPipeline._should_rewrite 触发条件
# ============================================================

class TestShouldRewrite:
    """QueryPipeline._should_rewrite 触发条件单测。

    REQUERY_ENABLED / REQUERY_MIN_QUERY_LENGTH 是 @property，
    每次访问时读 env var，所以 monkeypatch.setenv 即可生效。
    """

    def test_disabled_returns_false(self, monkeypatch):
        from pipeline.query import QueryPipeline

        monkeypatch.setenv("REQUERY_ENABLED", "false")
        pipeline = QueryPipeline()
        assert pipeline._should_rewrite("钳位电压是多少") is False

    def test_short_query_triggers(self, monkeypatch):
        from pipeline.query import QueryPipeline

        monkeypatch.setenv("REQUERY_ENABLED", "true")
        monkeypatch.setenv("REQUERY_MIN_QUERY_LENGTH", "30")
        pipeline = QueryPipeline()
        # < 30 字符
        assert pipeline._should_rewrite("钳位电压") is True
        assert pipeline._should_rewrite("how to use") is True

    def test_long_query_with_colloquial_triggers(self, monkeypatch):
        from pipeline.query import QueryPipeline

        monkeypatch.setenv("REQUERY_ENABLED", "true")
        monkeypatch.setenv("REQUERY_MIN_QUERY_LENGTH", "10")
        pipeline = QueryPipeline()
        # 长度 > 阈值但含口语化词（"为什么"）
        long_with_why = "这个 TVS 型号 DEMO27-A 为什么不能用"
        assert len(long_with_why) > 10
        assert pipeline._should_rewrite(long_with_why) is True

    def test_long_no_colloquial_not_trigger(self, monkeypatch):
        from pipeline.query import QueryPipeline

        monkeypatch.setenv("REQUERY_ENABLED", "true")
        monkeypatch.setenv("REQUERY_MIN_QUERY_LENGTH", "10")
        pipeline = QueryPipeline()
        # 长度 > 阈值且无口语化词
        assert pipeline._should_rewrite(
            "DEMO27-A 的击穿电压 VBR 最小值 27V 测试条件 IT=1mA"
        ) is False

    def test_empty_query_not_trigger(self, monkeypatch):
        from pipeline.query import QueryPipeline

        monkeypatch.setenv("REQUERY_ENABLED", "true")
        pipeline = QueryPipeline()
        assert pipeline._should_rewrite("") is False
        assert pipeline._should_rewrite("   ") is False


# ============================================================
# QueryPipeline._get_rewrites 缓存
# ============================================================

class TestGetRewritesCache:
    """QueryPipeline._get_rewrites 缓存行为单测。"""

    def test_cache_hit_skips_llm(self, monkeypatch):
        from pipeline.query import QueryPipeline

        pipeline = QueryPipeline()
        # 预填缓存
        pipeline._rewrite_cache["dummy_hash"] = (
            ["原 query", "改写 1", "改写 2"],
            9999999999.0,  # 未来时间，永远不过期
        )

        # 强制让 _rewrite_cache_key 返回 dummy_hash
        monkeypatch.setattr(
            QueryPipeline, "_rewrite_cache_key", staticmethod(lambda q, l: "dummy_hash")
        )

        # mock query_rewriter，如果被调用会失败
        pipeline._query_rewriter = MagicMock()
        pipeline._query_rewriter.rewrite.side_effect = AssertionError("LLM 不应被调用")

        result = pipeline._get_rewrites("任意 query", "zh")
        assert result == ["原 query", "改写 1", "改写 2"]
        pipeline._query_rewriter.rewrite.assert_not_called()

    def test_cache_miss_calls_llm(self, monkeypatch):
        from pipeline.query import QueryPipeline

        pipeline = QueryPipeline()
        pipeline._query_rewriter = MagicMock()
        pipeline._query_rewriter.rewrite.return_value = ["原 query", "改写 1"]

        result = pipeline._get_rewrites("原 query", "zh")
        assert result == ["原 query", "改写 1"]
        pipeline._query_rewriter.rewrite.assert_called_once()

    def test_llm_failure_returns_original(self, monkeypatch):
        from pipeline.query import QueryPipeline

        pipeline = QueryPipeline()
        pipeline._query_rewriter = MagicMock()
        pipeline._query_rewriter.rewrite.side_effect = RuntimeError("网络错误")

        result = pipeline._get_rewrites("原 query", "zh")
        assert result == ["原 query"]

    def test_original_query_always_first(self, monkeypatch):
        """LLM 改写后丢失原 query 时，强制把原 query 放回最前。"""
        from pipeline.query import QueryPipeline

        pipeline = QueryPipeline()
        pipeline._query_rewriter = MagicMock()
        # 改写器只返回变体（异常）
        pipeline._query_rewriter.rewrite.return_value = ["改写 1", "改写 2"]

        result = pipeline._get_rewrites("原 query", "zh")
        assert result[0] == "原 query"
        assert "改写 1" in result
        assert "改写 2" in result

    def test_cache_eviction_on_overflow(self, monkeypatch):
        from config import config
        from pipeline.query import QueryPipeline

        monkeypatch.setattr(config, "REQUERY_CACHE_MAX", 4)
        pipeline = QueryPipeline()
        pipeline._query_rewriter = MagicMock()
        pipeline._query_rewriter.rewrite.return_value = ["原", "改"]

        # 填满缓存
        for i in range(4):
            pipeline._get_rewrites(f"query_{i}", "zh")
        assert len(pipeline._rewrite_cache) == 4

        # 再写一个 → 应淘汰最旧的一半
        pipeline._get_rewrites("query_new", "zh")
        assert len(pipeline._rewrite_cache) <= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
