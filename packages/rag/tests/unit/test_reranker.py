# Test values are synthetic fixtures for software behavior.
"""
Reranker 单元测试（BGE-Reranker 版）

覆盖：
    - 基本重排序功能
    - 缓存命中 / 过期
    - 降级链路（rerank → embeddings → fallback）
    - 两级重排策略
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from core.reranker import Reranker


class TestReranker:
    """Reranker 核心功能测试。"""

    @pytest.fixture
    def candidates(self):
        return [
            {"id": "c1", "score": 0.5, "content": "ESD diode 27V breakdown.", "file_name": "a.pdf", "page": 1},
            {"id": "c2", "score": 0.9, "content": "Clamping voltage 43V max.", "file_name": "a.pdf", "page": 2},
            {"id": "c3", "score": 0.3, "content": "Package SOD-323.", "file_name": "a.pdf", "page": 3},
        ]

    def test_rerank_empty_candidates(self):
        """空候选集应返回空列表。"""
        reranker = Reranker()
        result = reranker.rerank("test query", [])
        assert result == []

    def test_rerank_clips_top_k(self, candidates):
        """重排序后应截断到 top_k。"""
        reranker = Reranker()
        # 模拟 rerank 端点返回成功
        with patch.object(reranker, "_try_rerank_endpoint", return_value=[0.8, 0.6, 0.4]):
            result = reranker.rerank("ESD voltage", candidates, top_k=2)
        assert len(result) == 2

    def test_rerank_adds_rerank_score(self, candidates):
        """每个命中项应包含 rerank_score 字段。"""
        reranker = Reranker()
        with patch.object(reranker, "_try_rerank_endpoint", return_value=[0.8, 0.6, 0.4]):
            result = reranker.rerank("ESD voltage", candidates, top_k=3)
        for item in result:
            assert "rerank_score" in item

    def test_rerank_sorted_by_score_desc(self, candidates):
        """重排序结果应按分数降序排列。"""
        reranker = Reranker()
        with patch.object(reranker, "_try_rerank_endpoint", return_value=[0.8, 0.6, 0.4]):
            result = reranker.rerank("ESD voltage", candidates, top_k=3)
        scores = [item["rerank_score"] for item in result]
        assert scores == sorted(scores, reverse=True)

    def test_cache_hit_within_ttl(self, candidates):
        """同一 query + top_k + 候选集在 TTL 内应命中缓存。"""
        reranker = Reranker()
        with patch.object(reranker, "_try_rerank_endpoint", return_value=[0.8, 0.6, 0.4]):
            result1 = reranker.rerank("ESD voltage", candidates, top_k=2)
            result2 = reranker.rerank("ESD voltage", candidates, top_k=2)
        # 缓存命中，结果应一致
        assert [r["id"] for r in result1] == [r["id"] for r in result2]

    def test_different_candidates_no_cache_collision(self, candidates):
        """不同候选集不应命中同一缓存。"""
        reranker = Reranker()
        with patch.object(reranker, "_try_rerank_endpoint", return_value=[0.8, 0.6, 0.4]):
            result1 = reranker.rerank("ESD voltage", candidates, top_k=2)
            different = [candidates[0], candidates[2]]
            result2 = reranker.rerank("ESD voltage", different, top_k=2)
        assert result1 != result2

    def test_cache_expired_after_ttl(self, candidates, monkeypatch):
        """缓存过期后应重新计算。"""
        reranker = Reranker()
        monkeypatch.setattr("core.reranker._CACHE_TTL_SEC", 0)
        with patch.object(reranker, "_try_rerank_endpoint", return_value=[0.8, 0.6, 0.4]):
            result1 = reranker.rerank("ESD voltage", candidates, top_k=2)
            result2 = reranker.rerank("ESD voltage", candidates, top_k=2)
        # TTL=0，每次调用应重新计算（不同对象）
        assert result1 is not result2

    def test_fallback_to_original_score_when_all_fail(self, candidates):
        """所有打分方法失败时，应按原始 score 降序排列。"""
        reranker = Reranker()
        with patch.object(reranker, "_try_rerank_endpoint", return_value=None):
            with patch.object(reranker, "_try_embedding_similarity", return_value=None):
                result = reranker.rerank("ESD voltage", candidates, top_k=3)
        assert len(result) == 3
        # 应按原始 score 降序
        assert result[0]["id"] == "c2"  # score 0.9

    def test_embedding_fallback_on_rerank_failure(self, candidates):
        """rerank 端点失败时，应降级到 embeddings 相似度。"""
        reranker = Reranker()
        with patch.object(reranker, "_try_rerank_endpoint", return_value=None):
            with patch.object(reranker, "_try_embedding_similarity", return_value=[0.9, 0.5, 0.3]):
                result = reranker.rerank("ESD voltage", candidates, top_k=3)
        assert len(result) == 3
        assert result[0]["rerank_score"] == 0.9

    def test_coarse_ranking_with_many_candidates(self):
        """候选超 10 条时，应触发两级重排（粗排 + 精排）。"""
        reranker = Reranker()
        many = [
            {"id": f"c{i}", "score": 1.0 - i * 0.01, "content": f"content {i}"}
            for i in range(20)
        ]
        # 模拟 rerank 返回部分分数（只精排粗排候选项）
        scores = [0.9 - i * 0.05 for i in range(10)]  # coarse_k = max(5*2, 10) = 10
        with patch.object(reranker, "_try_rerank_endpoint", return_value=scores):
            result = reranker.rerank("query", many, top_k=5)
        assert len(result) == 5
        for item in result:
            assert "rerank_score" in item

    def test_rerank_score_index_alignment(self):
        """rerank 返回的分数应按 index 正确对齐。"""
        reranker = Reranker()
        # 直接测试 _apply_scores 的排序逻辑
        candidates = [
            {"id": "c0", "score": 0.5, "content": "a"},
            {"id": "c1", "score": 0.5, "content": "b"},
            {"id": "c2", "score": 0.5, "content": "c"},
        ]
        result = reranker._apply_scores(candidates, [0.3, 0.6, 0.9], 3)
        # 按分数降序：c2(0.9), c1(0.6), c0(0.3)
        assert result[0]["id"] == "c2"
        assert result[1]["id"] == "c1"
        assert result[2]["id"] == "c0"

    def test_cosine_similarity_identical_vectors(self):
        """相同向量余弦相似度应为 1.0。"""
        v = [0.1, 0.2, 0.3]
        assert Reranker._cosine_similarity(v, v) == pytest.approx(1.0, rel=0.001)

    def test_cosine_similarity_orthogonal_vectors(self):
        """正交向量余弦相似度应为 0.0。"""
        assert Reranker._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, rel=0.001)

    def test_cosine_similarity_empty_vectors(self):
        """空向量应返回 0.0。"""
        assert Reranker._cosine_similarity([], []) == 0.0
        assert Reranker._cosine_similarity([1.0], []) == 0.0
