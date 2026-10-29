"""
多模态 RAG 系统 — 检索层 (Task 5)

职责：混合检索 + 图文关联增强。
  - text/table 通过 metadata.child_images 关联 image
  - image 通过 metadata.parent_chunk_id 关联 text/table
  - 关联 chunk 拿不到时，linked_chunks 留空，不抛错
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from utils.runtime_budget import submit_context
from typing import Optional

from config import config
from db.schemas import COLLECTION_NAME
from utils.logger import log


class Retriever:
    """混合检索 + 图文关联增强。"""

    def __init__(self, chroma_client: "ChromaClientSingleton", embedder: "Embedder") -> None:
        self.chroma = chroma_client
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """执行一次混合检索并完成图文关联增强。"""
        if not query or not query.strip():
            return []
        k = top_k if top_k is not None else config.RETRIEVAL_TOP_K
        if k <= 0:
            k = config.RETRIEVAL_TOP_K

        try:
            dense_vec, sparse_vec = self.embedder.embed_query(query)
        except Exception as exc:
            log.warning("embed_query 失败: %s", exc)
            return []
        # 按需 load（首次检索前等待索引构建完成）
        try:
            self.chroma.ensure_loaded()
        except Exception as exc:
            log.warning("ensure_loaded 失败，继续走 fallback: %s", exc)
        try:
            raw_hits = self.chroma.hybrid_search(
                query_dense=dense_vec, query_sparse=sparse_vec,
                top_k=k, filter_expr=filter_expr, query_text=query,
            )
        except Exception as exc:
            log.warning("hybrid_search 失败: %s", exc)
            return []

        normalized = [self._normalize_hit(h) for h in (raw_hits or [])]
        # 可选：并入图像向量检索结果（文搜图），默认关闭（ENABLE_VISUAL_SEARCH=true 开启）
        if config.ENABLE_VISUAL_SEARCH:
            try:
                visual_hits = self.visual_search(query, top_k=k, filter_expr=filter_expr)
                seen_ids = {h.get("id") for h in normalized if h.get("id")}
                for vh in visual_hits:
                    if vh.get("id") and vh["id"] not in seen_ids:
                        normalized.append(vh)
                        seen_ids.add(vh["id"])
                normalized = normalized[:k]
            except Exception as exc:
                log.warning("visual_search 并入失败: %s", exc)
        try:
            return self._link_related(normalized)
        except Exception as exc:
            log.warning("_link_related 失败: %s", exc)
            return normalized

    def visual_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """图像向量检索（文搜图/图搜图）：文本查询 → 多模态向量 → image_chunks 集合。

        需要百炼原生多模态协议（密钥/WorkspaceId 已配置）；不可用时返回空列表，
        不影响主链路。
        """
        if not query or not query.strip():
            return []
        k = top_k if top_k is not None else config.RETRIEVAL_TOP_K
        if k <= 0:
            k = config.RETRIEVAL_TOP_K
        try:
            image_vec = self.embedder.embed_image_query(query)
        except Exception as exc:
            log.warning("embed_image_query 失败: %s", exc)
            return []
        if not image_vec:
            return []
        try:
            self.chroma.ensure_loaded()
        except Exception as exc:
            log.warning("ensure_loaded 失败，继续走 fallback: %s", exc)
        try:
            raw_hits = self.chroma.image_search(
                query_image_vector=image_vec,
                top_k=k,
                filter_expr=filter_expr,
            )
        except Exception as exc:
            log.warning("image_search 失败: %s", exc)
            return []
        normalized = [self._normalize_hit(h) for h in (raw_hits or [])]
        try:
            return self._link_related(normalized)
        except Exception as exc:
            log.warning("_link_related 失败: %s", exc)
            return normalized

    def get_by_id(self, chunk_id: str) -> Optional[dict]:
        """按主键回查单条 chunk。"""
        if not chunk_id:
            return None
        try:
            entity = self.chroma.get_by_id(chunk_id)
            if entity is None:
                return None
            return self._normalize_hit({"id": chunk_id, "score": 0.0, "entity": entity})
        except Exception as exc:
            log.warning("get_by_id 失败: %s", exc)
            return None

    def retrieve_all_by_doc(
        self,
        doc_id: str,
        modalities: Optional[tuple[str, ...]] = None,
        section_id_prefix: Optional[str] = None,
    ) -> list[dict]:
        """对比专用：按 doc_id 拉取 chunk，可选按 section_id 前缀过滤。

        与 ``retrieve()`` 的区别：
            - 不需要 query embedding，跳过语义检索
            - 不依赖 Reranker
            - 用于跨文档对比场景，保证参数提取的完整覆盖

        Args:
            doc_id: 目标文档 ID。
            modalities: 模态过滤（"text"/"table"/"image"），None=全量。
            section_id_prefix: section_id 前缀过滤（None=全量），如 ``"5.3"`` 匹配所有
                ``"5.3.x"`` 子章节。空字符串视为全量（与 None 等价）。ChromaDB where
                不直接支持 regex，故改在 Python 层用 ``startswith()`` 过滤。

        Returns:
            标准化 chunk 列表，按 page 升序排序。
        """
        if not doc_id:
            return []
        # 按需 load（首次检索前等待索引构建完成）
        try:
            self.chroma.ensure_loaded()
        except Exception as exc:
            log.warning("ensure_loaded 失败，继续走 fallback: %s", exc)
        try:
            raw_hits = self.chroma.get_all_by_doc_id(doc_id, modalities=modalities)
        except Exception as exc:
            log.warning("get_all_by_doc_id 失败: %s", exc)
            return []
        if not raw_hits:
            return []

        # 2) Python 层按 section_id_prefix 过滤（ChromaDB where 不支持 regex）
        if section_id_prefix:
            prefix = str(section_id_prefix)
            filtered = []
            for h in raw_hits:
                # 兼容两种数据结构：
                # - 真实 _build_entity 输出：section_id 在 entity 内
                # - mock / 其它直传 dict：section_id 在 hit 顶层
                entity = h.get("entity", {}) or {}
                sid = (
                    h.get("section_id")
                    or entity.get("section_id", "")
                    or ""
                )
                sid = str(sid).strip()
                # 兜底：旧数据 section_id 为空时全部保留，避免空库
                if not sid or sid.startswith(prefix):
                    filtered.append(h)
            raw_hits = filtered
            log.info(
                "section 过滤: prefix=%s → %d chunks", prefix, len(raw_hits)
            )

        normalized = [self._normalize_hit(h) for h in raw_hits]
        try:
            return self._link_related(normalized)
        except Exception as exc:
            log.warning("_link_related 失败: %s", exc)
            return normalized

    def retrieve_by_section(
        self,
        doc_id: str,
        section_id: str,
        section_topic: Optional[str] = None,
    ) -> list[dict]:
        """按 section_id（+可选 topic）拉取文档指定章节的 chunk。

        Args:
            doc_id: 文档 ID。
            section_id: 章节编号前缀（支持子章节匹配，如 ``"5.3"`` 匹配
                ``"5.3.1"`` / ``"5.3.2"``）。**空字符串表示按 topic 全量检索**（仅
                按 section_topic 过滤，不做 section_id 前缀匹配），用于章节路由对比。
            section_topic: 可选 topic 过滤（如 ``"immunity.bci"``），与
                section_id 联合过滤。section_id 为空时仍生效。

        Returns:
            标准化 chunk 列表（与 ``retrieve()`` 一致结构）。无匹配返回空 list，不抛异常。
        """
        if not doc_id:
            return []
        # section_id 为空时：走全量（仅按 topic 过滤），用于章节路由
        section_id_prefix = section_id if section_id else None
        try:
            chunks = self.retrieve_all_by_doc(doc_id, section_id_prefix=section_id_prefix)
        except Exception as exc:
            log.warning("retrieve_by_section 失败: %s", exc)
            return []
        if section_topic:
            topic = str(section_topic)
            chunks = [c for c in chunks if (c.get("section_topic") or "") == topic]
        return chunks

    def multi_query_retrieve(
        self,
        queries: list[str],
        top_k: Optional[int] = None,
        filter_expr: Optional[str] = None,
        fusion_k: int = 60,
    ) -> list[dict]:
        """多 query 并行检索 + RRF (Reciprocal Rank Fusion) 融合。

        流程：
            1) ThreadPoolExecutor 并行调用 self.retrieve(q) for q in queries
            2) 任一 query 失败时该路降级为空（不影响其他路）
            3) 按 RRF 公式对所有命中 chunk 计算融合分数：
               score(chunk) = sum(1 / (fusion_k + rank_in_query_i))
            4) 按 RRF 分数降序排序，去重后取 top_k
            5) 对最终结果再做一次 _link_related 图文关联增强

        Args:
            queries: 检索 query 列表（通常含原 query + 改写变体）。
            top_k: 最终返回的 chunk 数。
            filter_expr: ChromaDB 过滤表达式（对所有 query 一致）。
            fusion_k: RRF 常数 k（参考 Cormack et al. 2009，默认 60）。

        Returns:
            按 RRF 分数排序的标准化 chunk 列表（已带 linked_chunks）。
        """
        # 防御：空查询或全部空字符串 → 返回空
        queries = [q for q in (queries or []) if q and q.strip()]
        if not queries:
            return []
        k = top_k if top_k is not None else config.RETRIEVAL_TOP_K
        if k <= 0:
            k = config.RETRIEVAL_TOP_K

        # 单 query 退化为原 retrieve，避免无意义并发
        if len(queries) == 1:
            return self.retrieve(queries[0], top_k=k, filter_expr=filter_expr)

        # ---- 阶段1: 并行多路检索（任一失败不影响其他） ----
        per_query_hits: list[list[dict]] = [[] for _ in queries]
        with ThreadPoolExecutor(
            max_workers=min(len(queries), 3),
            thread_name_prefix="multi-q",
        ) as pool:
            futures = {
                submit_context(pool, self.retrieve, q, top_k=k, filter_expr=filter_expr): idx
                for idx, q in enumerate(queries)
            }
            for fut in futures:
                idx = futures[fut]
                try:
                    per_query_hits[idx] = fut.result() or []
                except Exception as exc:  # noqa: BLE001
                    log.warning("multi_query 第 %d 路检索失败: %s", idx, exc)
                    per_query_hits[idx] = []

        # ---- 阶段2: RRF 融合 + 去重 ----
        # rrf_scores: chunk_id -> (rrf_score, chunk_dict)
        rrf_scores: dict[str, tuple[float, dict]] = {}
        for hits in per_query_hits:
            for rank, hit in enumerate(hits):
                cid = str(hit.get("id", ""))
                if not cid:
                    continue
                contrib = 1.0 / (fusion_k + rank + 1)  # rank 从 0 开始
                if cid in rrf_scores:
                    prev_score, prev_hit = rrf_scores[cid]
                    rrf_scores[cid] = (prev_score + contrib, prev_hit)
                else:
                    rrf_scores[cid] = (contrib, hit)

        if not rrf_scores:
            return []

        # 按 RRF 分数降序
        merged = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        # 取 top_k（去重已通过 chunk_id 保证）
        top_hits = [hit for _, hit in merged[:k]]

        # ---- 阶段3: 图文关联增强（与单 query 一致） ----
        try:
            return self._link_related(top_hits)
        except Exception as exc:  # noqa: BLE001
            log.warning("multi_query _link_related 失败: %s", exc)
            return top_hits

    def _normalize_hit(self, hit: dict) -> dict:
        """规整 ChromaDB hit 为标准 dict；从 metadata 拆出 child_images / parent_chunk_id。"""
        entity = hit.get("entity", {}) or {}
        meta = entity.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        child_images = meta.get("child_images") or []
        parent_chunk_id = meta.get("parent_chunk_id") or ""
        return {
            "id": str(hit.get("id", "")),
            "score": float(hit.get("score", 0.0)),
            "content": entity.get("content", "") or "",
            "searchable_text": entity.get("searchable_text", "") or "",
            "file_name": entity.get("file_name", "") or "",
            "page": entity.get("page", 0) or 0,
            "modality": entity.get("modality", "text") or "text",
            "metadata": meta,
            "doc_id": entity.get("doc_id", "") or "",
            "section": entity.get("section", "") or "",
            "child_images": list(child_images) if isinstance(child_images, list) else [],
            "parent_chunk_id": str(parent_chunk_id),
            "linked_chunks": [],
            # 章节结构字段（按章节检索用，Task 2 新增）
            "section_id": entity.get("section_id", "") or "",
            "section_path": entity.get("section_path", "") or "",
            "section_topic": entity.get("section_topic", "") or "",
        }

    def _link_related(self, hits: list[dict]) -> list[dict]:
        """图文关联增强：把关联 chunk 信息补充到 linked_chunks；附带 parent context 摘要。"""
        if not hits:
            return hits
        # 收集需要回查的 chunk_id
        needed: set[str] = set()
        for h in hits:
            for cid in h.get("child_images") or []:
                if cid:
                    needed.add(str(cid))
            pid = h.get("parent_chunk_id") or ""
            if pid:
                needed.add(str(pid))
        # 并行回查所有关联 chunk
        id_to_hit: dict[str, dict] = {}
        if needed:
            with ThreadPoolExecutor(max_workers=min(8, len(needed)), thread_name_prefix="link") as pool:
                futures = {pool.submit(self.get_by_id, cid): cid for cid in needed}
                for fut in futures:
                    cid = futures[fut]
                    try:
                        related = fut.result()
                        if related:
                            id_to_hit[cid] = related
                    except Exception as e:
                        log.warning("_link_related 回查失败 (cid=%s): %s", cid, e)
        # 填充 linked_chunks
        for h in hits:
            linked: list[dict] = []
            pid = h.get("parent_chunk_id") or ""
            if pid and pid in id_to_hit:
                rel = id_to_hit[pid]
                linked.append(self._linked_entry(rel, fallback_modality="text"))
                # 父子双层结构：附加 parent context 摘要到 hit 上
                parent_ctx = rel.get("content", "") or ""
                snippet_chars = config.PARENT_CONTEXT_SNIPPET_CHARS
                if parent_ctx and len(parent_ctx) > snippet_chars:
                    parent_ctx = parent_ctx[:snippet_chars] + "..."
                h["parent_context"] = parent_ctx
                h["parent_section"] = rel.get("section", "")
            for cid in h.get("child_images") or []:
                cid = str(cid)
                if cid in id_to_hit:
                    rel = id_to_hit[cid]
                    linked.append(self._linked_entry(rel, fallback_modality="image"))
            h["linked_chunks"] = linked
        return hits

    @staticmethod
    def _linked_entry(rel: dict, fallback_modality: str) -> dict:
        """构造 linked_chunks 中的单条关联摘要。"""
        return {
            "id": rel["id"],
            "modality": rel.get("modality", fallback_modality),
            "file_name": rel.get("file_name", ""),
            "page": rel.get("page", 0),
            "content": rel.get("content", ""),
        }


__all__ = ["Retriever"]
