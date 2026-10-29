# Test values are synthetic fixtures for software behavior.
"""
Retriever.multi_query_retrieve 单元测试

覆盖范围：
    - 空查询 / 全部空字符串
    - 单 query 退化为原 retrieve
    - 3 路并行检索 + RRF 融合正确性
    - 重复 chunk 去重（按 chunk_id）
    - 任一 query 失败时降级（不影响其他路）
    - top_k 截断
    - _link_related 调用
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_hit(chunk_id: str, score: float = 0.5) -> dict:
    """构造一个标准化的 hit dict（与 Retriever._normalize_hit 输出一致）。"""
    return {
        "id": chunk_id,
        "score": score,
        "content": f"content for {chunk_id}",
        "searchable_text": f"text {chunk_id}",
        "file_name": "test.pdf",
        "page": 1,
        "modality": "text",
        "metadata": {},
        "doc_id": "doc1",
        "section": "",
        "child_images": [],
        "parent_chunk_id": "",
        "linked_chunks": [],
    }


class TestMultiQueryRetrieve:
    """Retriever.multi_query_retrieve 行为单测。"""

    def _make_retriever(self):
        """构造一个 Retriever，chroma/embedder 用 mock。"""
        from core.retriever import Retriever

        retriever = Retriever(MagicMock(), MagicMock())
        return retriever

    # ---- 防御性边界 ----

    def test_empty_queries_returns_empty(self):
        retriever = self._make_retriever()
        assert retriever.multi_query_retrieve([]) == []
        assert retriever.multi_query_retrieve(["", "  "]) == []

    def test_single_query_falls_back_to_retrieve(self):
        """单 query 时退化为 self.retrieve（避免无意义并发）。"""
        retriever = self._make_retriever()
        retriever.retrieve = MagicMock(return_value=[_make_hit("c1")])
        retriever._link_related = MagicMock(side_effect=lambda x: x)

        result = retriever.multi_query_retrieve(["only one query"], top_k=5)

        retriever.retrieve.assert_called_once_with(
            "only one query", top_k=5, filter_expr=None,
        )
        assert len(result) == 1
        assert result[0]["id"] == "c1"

    # ---- RRF 融合正确性 ----

    def test_rrf_fusion_three_queries(self):
        """3 路检索，每路 2 个 chunk，验证 RRF 融合排序。"""
        retriever = self._make_retriever()
        retriever._link_related = MagicMock(side_effect=lambda x: x)

        # query1: [A, B]   → A rank 0, B rank 1
        # query2: [B, C]   → B rank 0, C rank 1
        # query3: [A, C]   → A rank 0, C rank 1
        # RRF 分数（k=60, rank 从 0 开始，contrib = 1/(k+rank+1)）：
        #   A: 1/61 + 1/61 = 0.03279  (q1 rank0 + q3 rank0, 两次 rank 0 → 最高)
        #   B: 1/62 + 1/61 = 0.03252  (q1 rank1 + q2 rank0)
        #   C: 1/62 + 1/62 = 0.03226  (q2 rank1 + q3 rank1)
        # 期望排序：A > B > C
        retriever.retrieve = MagicMock(side_effect=[
            [_make_hit("A"), _make_hit("B")],   # query1
            [_make_hit("B"), _make_hit("C")],   # query2
            [_make_hit("A"), _make_hit("C")],   # query3
        ])

        result = retriever.multi_query_retrieve(
            ["q1", "q2", "q3"], top_k=5, fusion_k=60,
        )

        ids = [h["id"] for h in result]
        assert ids == ["A", "B", "C"]

    def test_rrf_single_appearance(self):
        """只在 1 路出现的 chunk 也能被召回。"""
        retriever = self._make_retriever()
        retriever._link_related = MagicMock(side_effect=lambda x: x)
        retriever.retrieve = MagicMock(side_effect=[
            [_make_hit("X")],   # query1 独有
            [_make_hit("Y")],   # query2 独有
            [_make_hit("X"), _make_hit("Y")],  # query3
        ])

        result = retriever.multi_query_retrieve(
            ["q1", "q2", "q3"], top_k=10, fusion_k=60,
        )
        # X 和 Y 都应被召回
        ids = set(h["id"] for h in result)
        assert ids == {"X", "Y"}

    # ---- 失败降级 ----

    def test_one_query_failure_does_not_affect_others(self):
        """中间一路失败时，其他路的结果仍能被融合。"""
        retriever = self._make_retriever()
        retriever._link_related = MagicMock(side_effect=lambda x: x)

        # query2 失败
        def fake_retrieve(query, top_k=None, filter_expr=None):
            if query == "q2":
                raise RuntimeError("embedding 服务不可用")
            return [_make_hit(f"hit_{query}")]

        retriever.retrieve = MagicMock(side_effect=fake_retrieve)

        result = retriever.multi_query_retrieve(
            ["q1", "q2", "q3"], top_k=10, fusion_k=60,
        )
        # q1 和 q3 的结果应被融合（q2 降级为空）
        ids = [h["id"] for h in result]
        assert "hit_q1" in ids
        assert "hit_q3" in ids
        # q2 的结果不应出现（因为 q2 整路失败）
        assert "hit_q2" not in ids

    def test_all_queries_fail_returns_empty(self):
        retriever = self._make_retriever()
        retriever._link_related = MagicMock(side_effect=lambda x: x)
        retriever.retrieve = MagicMock(side_effect=RuntimeError("all fail"))

        result = retriever.multi_query_retrieve(["q1", "q2"], top_k=5)
        assert result == []

    def test_some_queries_return_empty(self):
        """部分 query 召回为空时，不影响其他路。"""
        retriever = self._make_retriever()
        retriever._link_related = MagicMock(side_effect=lambda x: x)
        retriever.retrieve = MagicMock(side_effect=[
            [],   # query1 无命中
            [_make_hit("B")],   # query2
            [_make_hit("A"), _make_hit("B")],   # query3
        ])

        result = retriever.multi_query_retrieve(
            ["q1", "q2", "q3"], top_k=10, fusion_k=60,
        )
        ids = [h["id"] for h in result]
        # A 在 q3 排第 1 → 1/61
        # B 在 q2 排第 1 + q3 排第 2 → 1/61 + 1/62 (B 应高于 A)
        assert ids[0] == "B"
        assert ids[1] == "A"

    # ---- top_k 截断 ----

    def test_top_k_truncation(self):
        """融合后只返回 top_k 个 chunk。"""
        retriever = self._make_retriever()
        retriever._link_related = MagicMock(side_effect=lambda x: x)

        # 单 query 返回 5 个 chunk
        retriever.retrieve = MagicMock(side_effect=[
            [_make_hit(f"c{i}") for i in range(5)],
            [_make_hit(f"c{i}") for i in range(5)],  # 重复 chunk 验证去重
        ])

        result = retriever.multi_query_retrieve(["q1", "q2"], top_k=3, fusion_k=60)
        assert len(result) == 3

    # ---- 重复 chunk 去重 ----

    def test_duplicate_chunks_deduped_by_id(self):
        """同一 chunk_id 在多路中出现时，RRF 分数累加。"""
        retriever = self._make_retriever()
        retriever._link_related = MagicMock(side_effect=lambda x: x)

        # 同一 chunk "shared" 在两路都出现
        retriever.retrieve = MagicMock(side_effect=[
            [_make_hit("shared"), _make_hit("q1_only")],
            [_make_hit("q2_only"), _make_hit("shared")],
        ])

        result = retriever.multi_query_retrieve(["q1", "q2"], top_k=10, fusion_k=60)
        ids = [h["id"] for h in result]
        # shared 在两路都出现（rank 0 + rank 1），RRF 最高
        assert ids[0] == "shared"
        # 重复 chunk 已被去重
        assert ids.count("shared") == 1
        # 独占 chunk 也在结果中
        assert set(ids) == {"shared", "q1_only", "q2_only"}

    # ---- 异常 chunk 处理 ----

    def test_chunks_without_id_ignored(self):
        """没有 id 的 chunk 不会进入融合。"""
        retriever = self._make_retriever()
        retriever._link_related = MagicMock(side_effect=lambda x: x)

        bad_hit = _make_hit("good")
        bad_hit.pop("id", None)  # 移除 id

        retriever.retrieve = MagicMock(side_effect=[
            [bad_hit, _make_hit("c1")],
            [_make_hit("c2")],
        ])

        result = retriever.multi_query_retrieve(["q1", "q2"], top_k=10, fusion_k=60)
        # bad_hit 被忽略，c1 和 c2 保留
        ids = [h["id"] for h in result]
        assert "good" not in ids
        assert "c1" in ids
        assert "c2" in ids

    # ---- _link_related 调用 ----

    def test_link_related_called_on_merged_result(self):
        """融合后应再做一次 _link_related 图文关联增强。"""
        retriever = self._make_retriever()
        retriever.retrieve = MagicMock(side_effect=[
            [_make_hit("a"), _make_hit("b")],
            [_make_hit("b"), _make_hit("c")],
        ])
        retriever._link_related = MagicMock(side_effect=lambda hits: [
            {**h, "linked_chunks": ["linked"]} for h in hits
        ])

        result = retriever.multi_query_retrieve(["q1", "q2"], top_k=10, fusion_k=60)
        # _link_related 应被调用一次（在融合之后）
        assert retriever._link_related.call_count == 1
        # 所有 chunk 应有 linked_chunks
        assert all(h.get("linked_chunks") == ["linked"] for h in result)

    def test_link_related_failure_returns_merged_without_raise(self):
        """_link_related 失败时降级为融合结果，不抛错。"""
        retriever = self._make_retriever()
        retriever.retrieve = MagicMock(side_effect=[
            [_make_hit("a")],
            [_make_hit("b")],
        ])
        retriever._link_related = MagicMock(side_effect=RuntimeError("link failed"))

        result = retriever.multi_query_retrieve(["q1", "q2"], top_k=10, fusion_k=60)
        # 降级为融合后的 hits（无 linked_chunks）
        ids = [h["id"] for h in result]
        assert set(ids) == {"a", "b"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
