"""
多模态 RAG 系统 — 重排序层 (Task 5)

职责：基于 BGE-Reranker 交叉编码器对候选文档打分重排。
  - 优先调用 OneAPI /v1/rerank 端点
  - 不支持时降级为 /v1/embeddings + 余弦相似度
  - 全部失败时 fallback 到原 score 排序
  - 同 query+top_k 在 300s 内复用缓存结果
  - 首次调用并行探测 rerank 和 embedding 端点，后续直接用可用端点
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx

from config import config
from utils.logger import log
from utils.retry import httpx_retry_post
from utils.urls import build_embeddings_url, build_rerank_url
from utils.runtime_budget import submit_context, cache_namespace, current_budget

_CACHE_TTL_SEC: int = 300
# 内容截断长度（避免 payload 过大）
_CONTENT_PREVIEW: int = 800

# 带 429 重试的 httpx post 封装
_rerank_post = httpx_retry_post(max_retries=3, base_delay=1.0, max_delay=30.0, timeout=30.0)
_embed_post = httpx_retry_post(max_retries=3, base_delay=1.0, max_delay=30.0, timeout=15.0)

# 模态微调权重：参数表/文本应优先于图片描述（避免图片片段抢占参数答案）
_MODALITY_BOOST: dict[str, float] = {"table": 0.08, "text": 0.0, "image": -0.05}


class Reranker:
    """基于 BGE-Reranker 的专用重排序器。

    优先使用 /v1/rerank 端点（Jina/Cohere 兼容协议），
    不支持时降级为 /v1/embeddings + 余弦相似度打分。
    """

    def __init__(self) -> None:
        self.model = config.RERANK_MODEL
        # 缓存: { (query, k): (result, timestamp) }
        self._cache: dict[tuple[str, int], tuple[list[dict], float]] = {}
        self._cache_lock = threading.Lock()
        # 记录最近一次重排方式，用于评估报告
        self.last_rerank_method: str = "none"
        # 端点探测：None=未探测，True=rerank可用，False=仅embedding可用
        self._rerank_available: Optional[bool] = None
        self._probe_lock = threading.Lock()

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """对候选重排并取 top_k；返回的 hit 含 rerank_score 字段。

        两级策略：
          - 候选 ≤ 10：直接调用 rerank 接口
          - 候选 > 10：先按原始 score 粗排取 2x top_k，仅对粗排结果精排
        """
        if not candidates:
            return []
        k = top_k if top_k is not None else config.RERANK_TOP_K
        if k <= 0:
            k = config.RERANK_TOP_K

        # 缓存键：(query, k, candidate_ids) 确保不同候选集不共享缓存
        candidate_ids = tuple(sorted(c.get("id", "") for c in candidates))
        cache_key = (cache_namespace() + (query or ""), k, candidate_ids)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # 两级重排
        COARSE_THRESHOLD = 10

        if len(candidates) <= COARSE_THRESHOLD:
            scores = self._score_with_reranker(query, candidates)
            ranked = self._apply_scores(candidates, scores, k)
        else:
            coarse = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
            coarse_k = min(len(coarse), max(k * 2, COARSE_THRESHOLD))
            coarse_candidates = coarse[:coarse_k]
            tail_candidates = coarse[coarse_k:]

            scores = self._score_with_reranker(query, coarse_candidates)
            ranked = self._apply_scores(coarse_candidates, scores, k)
            # 精排后不足 k 条时，用粗排尾部补充
            if len(ranked) < k:
                for c in tail_candidates:
                    item = dict(c)
                    item.setdefault("rerank_score", c.get("score", 0.0))
                    ranked.append(item)
                    if len(ranked) >= k:
                        break

        self._cache_set(cache_key, ranked)
        return ranked

    # ------------------------ 打分 ------------------------

    def _score_with_reranker(self, query: str, candidates: list[dict]) -> Optional[list[float]]:
        """调用 BGE-Reranker 打分，首次并行探测端点，后续用已知可用端点。

        内容选择优先级：searchable_text > content。
        searchable_text 包含文件名、章节、上下文，比 content（仅 VL 描述）更适合 reranker 打分。
        """
        documents = []
        for c in candidates:
            text = (c.get("searchable_text") or c.get("content") or "").strip()
            if config.RERANK_CONTEXT_PREFIX_ENABLED:
                # 语境前缀：让交叉编码器看到 chunk 属于哪份文档/章节
                # （表格块 searchable_text 是裸表，缺语境导致跨族排序失效）
                fname = c.get("file_name") or ""
                section = (
                    c.get("section_path")
                    or c.get("section")
                    or c.get("section_id")
                    or ""
                )
                prefix = ""
                if fname:
                    prefix += f"《{fname}》"
                if section:
                    prefix += f" 章节:{str(section).strip()[:120]}"
                if prefix:
                    text = f"{prefix}\n{text}"
            documents.append(text[:_CONTENT_PREVIEW])
        if not documents:
            return [0.0] * len(candidates)

        if current_budget.get() is not None:
            # Fixed ordered discovery avoids process-global probe warmup affecting comparisons.
            scores = self._try_rerank_endpoint(query, documents)
            if scores is not None:
                self.last_rerank_method = "rerank"
                return scores
            scores = self._try_embedding_similarity(query, documents)
            self.last_rerank_method = "embeddings" if scores is not None else "raw_score"
            return scores

        # 已探测过：直接用已知可用端点
        if self._rerank_available is True:
            scores = self._try_rerank_endpoint(query, documents)
            if scores is not None:
                self.last_rerank_method = "rerank"
                return scores
            log.warning("Rerank 端点突然不可用，降级为 embeddings")
            self._rerank_available = False

        if self._rerank_available is False:
            scores = self._try_embedding_similarity(query, documents)
            if scores is not None:
                self.last_rerank_method = "embeddings"
                return scores
            # embeddings 也失败（如瞬时 400/限流）→ 重置探测状态，允许下次重探
            # rerank 端点，避免"一次瞬时失败后整进程粘性降级为原始序"。
            self._rerank_available = None
            self.last_rerank_method = "raw_score"
            return None

        # 首次调用：并行探测两个端点，用先成功的
        with self._probe_lock:
            if self._rerank_available is not None:
                # 在等锁期间已被其他线程探测完，用其结果
                return self._score_with_reranker(query, candidates)

            scores_box: list[Optional[list[float]]] = [None, None]
            def _try_rerank():
                scores_box[0] = self._try_rerank_endpoint(query, documents)
            def _try_embed():
                scores_box[1] = self._try_embedding_similarity(query, documents)

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="r-probe") as pool:
                f1 = submit_context(pool, _try_rerank)
                f2 = submit_context(pool, _try_embed)
                # 任一端点先成功即返回，不等较慢的那个
                from concurrent.futures import FIRST_COMPLETED, wait
                done, _ = wait([f1, f2], return_when=FIRST_COMPLETED)
                # 取消未完成的 future
                for f in [f1, f2]:
                    if f not in done:
                        f.cancel()

            if scores_box[0] is not None:
                self._rerank_available = True
                self.last_rerank_method = "rerank"
                return scores_box[0]
            if scores_box[1] is not None:
                self._rerank_available = False
                self.last_rerank_method = "embeddings"
                return scores_box[1]

            # 都失败
            self.last_rerank_method = "raw_score"
            return None

    def _try_rerank_endpoint(self, query: str, documents: list[str]) -> Optional[list[float]]:
        """调用 /v1/rerank 端点（Jina/Cohere 兼容协议）。"""
        url = build_rerank_url()
        headers = {
            "Authorization": f"Bearer {config.ONEAPI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        try:
            resp = _rerank_post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                log.warning(
                    "Rerank 端点返回 %d: %s",
                    resp.status_code, resp.text[:200],
                )
                return None
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None
            # 按 index 对齐
            scores = [0.0] * len(documents)
            for item in results:
                idx = int(item.get("index", 0))
                score = float(item.get("relevance_score", 0.0))
                if 0 <= idx < len(scores):
                    scores[idx] = score
            return scores
        except Exception as e:
            log.warning("Rerank 端点调用失败: %s", e)
            return None

    def _try_embedding_similarity(
        self, query: str, documents: list[str],
    ) -> Optional[list[float]]:
        """降级方案：用 /v1/embeddings 分别编码 query 和 documents，计算余弦相似度。"""
        url = build_embeddings_url()
        headers = {
            "Authorization": f"Bearer {config.ONEAPI_API_KEY}",
            "Content-Type": "application/json",
        }
        # 编码 query + documents 合并为一次请求
        all_texts = [query] + documents
        try:
            resp = _embed_post(
                url,
                headers=headers,
                json={"model": config.TEXT_EMBED_MODEL, "input": all_texts},
            )
            if resp.status_code != 200:
                log.warning("Embeddings 降级返回 %d", resp.status_code)
                return None
            data = resp.json()
            items = data.get("data", [])
            if len(items) < 2:
                return None
            # 提取向量
            query_vec = items[0].get("embedding", [])
            if not query_vec:
                return None
            scores = []
            for i in range(1, len(items)):
                doc_vec = items[i].get("embedding", [])
                if not doc_vec:
                    scores.append(0.0)
                else:
                    scores.append(self._cosine_similarity(query_vec, doc_vec))
            return scores
        except Exception as e:
            log.warning("Embeddings 降级调用失败: %s", e)
            return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算余弦相似度。"""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------ 排序 + 缓存 ------------------------

    @staticmethod
    def _apply_scores(
        candidates: list[dict],
        scores: Optional[list[float]],
        k: int,
    ) -> list[dict]:
        """scores=None 时按原 score 排序；否则按 rerank 分数排序。"""
        if scores is None:
            ordered = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
            for c in ordered:
                c.setdefault("rerank_score", c.get("score", 0.0))
            return ordered[:k]
        n = len(candidates)
        aligned = list(scores) + [0.0] * max(0, n - len(scores))
        scored: list[dict] = []
        for i, c in enumerate(candidates):
            item = dict(c)
            item["rerank_score"] = float(aligned[i]) + _MODALITY_BOOST.get(
                c.get("modality", "text"), 0.0
            )
            scored.append(item)
        scored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return scored[:k]

    def _cache_get(self, key: tuple[str, int]) -> Optional[list[dict]]:
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            result, ts = entry
            if time.time() - ts > _CACHE_TTL_SEC:
                self._cache.pop(key, None)
                return None
            return [dict(item) for item in result]

    def _cache_set(self, key: tuple[str, int], result: list[dict]) -> None:
        with self._cache_lock:
            if len(self._cache) >= 256:
                sorted_keys = sorted(self._cache, key=lambda k: self._cache[k][1])
                for old in sorted_keys[:128]:
                    self._cache.pop(old, None)
            self._cache[key] = ([dict(item) for item in result], time.time())


__all__ = ["Reranker"]
