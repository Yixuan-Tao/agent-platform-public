"""
ChromaDB 客户端封装（单例 + 双 collection 混合检索 + RRF 融合）。

设计要点：
- 进程内单例，避免重复连接 ChromaDB 文件
- 使用 chromadb.PersistentClient 做本地持久化存储
- 双 collection 策略：
    - ``document_chunks``：text/table 模态（dense_vector 1024 维）
    - ``image_chunks``：image 模态（image_vector 1024 维）
- 混合检索：跨 collection top_k 后 RRF 融合
- 所有方法异常都经过 ``utils.logger.log`` 记录，并尽量抛出以便上层感知
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import config
from db.schemas import (
    COLLECTION_NAME,
    IMAGE_COLLECTION_NAME,
    MODALITY_IMAGE,
    DocumentChunk,
)
from utils.logger import log


# RRF 融合常数（经典取 60）
_RRF_K: int = 60


class ChromaClientSingleton:
    """进程内全局唯一的 ChromaDB 客户端封装。

    用法：
        client = ChromaClientSingleton.get_instance()
        client.insert(chunks)                  # 自动按 modality 路由
        hits = client.hybrid_search(dense_vec)  # 跨 collection 检索
    """

    _instance: Optional["ChromaClientSingleton"] = None
    _instance_lock = threading.Lock()

    # ------------------------ 单例相关 ------------------------

    @classmethod
    def get_instance(cls) -> "ChromaClientSingleton":
        """获取/创建单例（线程安全）。默认使用 ``config.CHROMA_DB_PATH``。"""
        if cls._instance is None:
            with cls._instance_lock:
                # 双重检查：锁内再次确认，避免竞态条件
                if cls._instance is None:
                    cls._instance = cls(db_path=config.CHROMA_DB_PATH)
                    cls._instance.connect()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（线程安全，测试场景使用）。"""
        with cls._instance_lock:
            if cls._instance is not None:
                try:
                    cls._instance.close()
                except Exception:  # noqa: BLE001 - 重置场景忽略错误
                    pass
            cls._instance = None

    # ------------------------ 初始化 ------------------------

    def __init__(self, db_path: str) -> None:
        self._db_path: str = db_path
        self._client: Optional[chromadb.PersistentClient] = None
        self._doc_collection: Optional[Any] = None
        self._image_collection: Optional[Any] = None
        # list_documents 缓存（避免每次 O(N) 全表扫描）
        self._list_cache: Optional[list[dict[str, Any]]] = None
        self._list_cache_ts: float = 0.0
        self._list_cache_lock = threading.Lock()
        self._LIST_CACHE_TTL: float = 30.0

    # ------------------------ 连接管理 ------------------------

    @property
    def is_connected(self) -> bool:
        """检查 ChromaDB 客户端是否已连接且可用。"""
        return self._client is not None

    def connect(self) -> None:
        """建立 ChromaDB 连接并自动获取/创建 collection。"""
        if self._client is not None:
            log.debug("ChromaDB 客户端已连接，跳过重复 connect")
            return

        try:
            log.info("连接 ChromaDB：path=%s", self._db_path)
            self._client = chromadb.PersistentClient(
                path=self._db_path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            # 获取或创建 collection
            self._doc_collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "多模态 RAG 文档切片 - 文本/表格"},
            )
            self._image_collection = self._client.get_or_create_collection(
                name=IMAGE_COLLECTION_NAME,
                metadata={"description": "多模态 RAG 文档切片 - 图片"},
            )
            log.info(
                "ChromaDB 就绪：collections=[%s, %s]",
                COLLECTION_NAME, IMAGE_COLLECTION_NAME,
            )
        except Exception as exc:
            log.exception("连接 ChromaDB 失败：%s", exc)
            raise

    def ensure_loaded(self, timeout: float = 30.0) -> None:
        """ChromaDB PersistentClient 无需显式加载，保留接口兼容。"""
        if self._client is None:
            raise RuntimeError("ChromaDB 客户端未连接，请先调用 connect()")

    def close(self) -> None:
        """关闭底层连接（ChromaDB PersistentClient 无 close 方法，仅置空引用）。"""
        if self._client is not None:
            try:
                log.info("ChromaDB 客户端已关闭")
            except Exception as exc:  # noqa: BLE001
                log.warning("关闭 ChromaDB 客户端时出错：%s", exc)
            finally:
                self._client = None
                self._doc_collection = None
                self._image_collection = None

    # ------------------------ 写入 ------------------------

    def insert(self, chunks: list[DocumentChunk]) -> dict[str, Any]:
        """按 modality 自动路由到对应 collection，返回汇总统计。

        image 模态 chunk 同时写入两个 collection：
          - ``image_chunks``：以图搜图（image_vector）
          - ``document_chunks``：文本检索命中（dense_vector，来自 VL 描述）
        """
        if self._client is None:
            log.warning("ChromaDB 未连接，insert 跳过")
            return {"insert_count": 0, "ids": []}
        if not chunks:
            return {"insert_count": 0, "ids": []}

        text_chunks = [c for c in chunks if c.modality != MODALITY_IMAGE]
        image_chunks = [c for c in chunks if c.modality == MODALITY_IMAGE]

        total = {"insert_count": 0, "ids": []}
        if text_chunks:
            r = self._insert_into(self._doc_collection, text_chunks)
            total["insert_count"] += r.get("insert_count", 0)
            total["ids"] += r.get("ids", [])
        if image_chunks:
            # 1) 写入 image_chunks（以图搜图，使用 image_vector）
            r = self._insert_into(self._image_collection, image_chunks)
            total["insert_count"] += r.get("insert_count", 0)
            total["ids"] += r.get("ids", [])
            # 2) 同时写入 document_chunks（文本检索可命中，使用 dense_vector）
            r2 = self._insert_into(self._doc_collection, image_chunks, use_text_entity=True)
            total["insert_count"] += r2.get("insert_count", 0)
        # 写入会改变 list_documents 结果，失效缓存
        if total["insert_count"] > 0:
            self.invalidate_list_cache()
        log.info("插入 %d 条 chunk 到 collections", total["insert_count"])
        return total

    def _insert_into(
        self,
        collection: Any,
        chunks: list[DocumentChunk],
        use_text_entity: bool = False,
    ) -> dict[str, Any]:
        """向指定 collection 插入 chunks。

        Args:
            collection: ChromaDB collection 对象
            chunks: 待插入的 chunk 列表
            use_text_entity: 若为 True，对 image 模态使用 to_text_entity()
                             输出 dense_vector 而非 image_vector
        """
        if collection is None:
            return {"insert_count": 0, "ids": []}
        try:
            ids: list[str] = []
            embeddings: list[list[float]] = []
            metadatas: list[dict[str, Any]] = []
            documents: list[str] = []

            for chunk in chunks:
                if use_text_entity:
                    cid, emb, meta, doc = chunk.to_text_entity()
                else:
                    cid, emb, meta, doc = chunk.to_chroma_entity()
                # 跳过空向量
                if not emb or all(v == 0.0 for v in emb):
                    continue
                ids.append(cid)
                embeddings.append(emb)
                metadatas.append(meta)
                documents.append(doc)

            if not ids:
                return {"insert_count": 0, "ids": []}

            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            log.info("插入 %d 条 chunk 到 %s", len(ids), collection.name)
            return {"insert_count": len(ids), "ids": ids}
        except Exception as exc:
            log.exception("插入失败 %s: %s", collection.name if collection else "?", exc)
            raise

    # ------------------------ 检索（跨 collection 混合 + RRF） ------------------------

    def hybrid_search(
        self,
        query_dense: list[float],
        query_sparse: Optional[dict[int, float]] = None,
        top_k: int = 50,
        filter_expr: Optional[str] = None,
        query_text: str = "",
    ) -> list[dict[str, Any]]:
        """跨 collection 检索（RRF 融合）。

        策略：
            - 对 ``document_chunks`` 用 ``dense_vector`` 检索 top_k
            - 提取 query 关键词做全文检索（where_document $contains）
            - 两路 RRF 融合后取 top_k

        Args:
            query_dense: 文本稠密向量。
            query_sparse: BM25 稀疏向量（保留签名，当前用于触发关键词检索）。
            top_k: 最终返回的 top_k。
            filter_expr: 过滤表达式（ChromaDB where 格式）。
            query_text: 原始查询文本，用于提取关键词做全文检索。
        """
        if self._client is None:
            raise RuntimeError("ChromaDB 客户端未连接，请先调用 connect()")
        if not query_dense:
            raise ValueError("query_dense 不能为空")

        rrf_scores: dict[str, float] = {}
        hit_entities: dict[str, dict[str, Any]] = {}

        where = self._build_where(filter_expr)

        try:
            # 路1: 稠密向量检索
            results = self._doc_collection.query(
                query_embeddings=[query_dense],
                n_results=top_k,
                where=where,
                include=["metadatas", "documents", "distances"],
            )
            self._accumulate_rrf(results, rrf_scores, hit_entities)
        except Exception as exc:
            log.exception("文本 collection 检索失败：%s", exc)
            raise

        # 路2: 关键词全文检索（基于 query_text 提取关键词，并行执行）
        if query_text and query_text.strip():
            keywords = self._extract_keywords(query_text)
            if keywords:
                # 并行调 ChromaDB query，避免 5 次串行 RTT
                kw_top_k = min(top_k, 10)

                def _search_one_kw(kw: str) -> Optional[dict[str, Any]]:
                    try:
                        return self._doc_collection.query(
                            query_embeddings=[query_dense],
                            n_results=kw_top_k,
                            where=where,
                            where_document={"$contains": kw},
                            include=["metadatas", "documents", "distances"],
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.warning("关键词检索失败 (kw=%s): %s", kw, exc)
                        return None

                with ThreadPoolExecutor(
                    max_workers=min(5, len(keywords)),
                    thread_name_prefix="kw-search",
                ) as pool:
                    results_list = list(pool.map(_search_one_kw, keywords[:5]))
                for kw_results in results_list:
                    if kw_results is None:
                        continue
                    self._accumulate_rrf(kw_results, rrf_scores, hit_entities)

        # ---- 融合排序 ----
        merged: list[dict[str, Any]] = [
            {"id": hid, "score": rrf_scores[hid], "entity": hit_entities[hid]}
            for hid in rrf_scores
        ]
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    @staticmethod
    def _extract_keywords(query_text: str) -> list[str]:
        """从查询文本中提取关键词用于全文检索。

        中英文混合策略：
            - 中文使用 jieba 分词（精确模式），提取名词/动词等实义词
            - 英文/数字按正则切分
            - 过滤停用词和短词（< 2 字符）
        """
        import re
        # 尝试 jieba 分词（中文场景），失败则降级为字符正则
        try:
            import jieba
            chinese_tokens = list(jieba.cut(query_text))
        except Exception:
            chinese_tokens = re.findall(r'[\u4e00-\u9fff]+', query_text)
        # 英文/数字/型号
        eng_tokens = re.findall(r'[a-zA-Z0-9-]+', query_text.lower())
        # 合并：中文 jieba 分词 + 英文正则
        all_tokens = [t.strip() for t in chinese_tokens if t.strip()] + eng_tokens
        # 过滤停用词和短词（< 2 字符）
        stopwords = {'的', '是', '在', '和', '了', '有', '为', '与', '或', '不',
                     'the', 'is', 'a', 'an', 'of', 'in', 'to', 'for', 'and', 'or',
                     '什么', '怎么', '如何', '哪些', '哪个', '多少', '为什么',
                     'what', 'how', 'which', 'why', 'when', 'where', '请', '描述', '一下'}
        keywords = []
        seen = set()
        for t in all_tokens:
            t = t.lower().strip()
            if len(t) >= 2 and t not in stopwords and t not in seen:
                seen.add(t)
                keywords.append(t)
        return keywords

    def image_search(
        self,
        query_image_vector: list[float],
        top_k: int = 20,
        filter_expr: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """图片向量检索：从 ``image_chunks`` collection 中找出 top_k 相似图片。"""
        if self._client is None:
            raise RuntimeError("ChromaDB 客户端未连接，请先调用 connect()")
        if not query_image_vector:
            raise ValueError("query_image_vector 不能为空")
        try:
            where = self._build_where(filter_expr)
            results = self._image_collection.query(
                query_embeddings=[query_image_vector],
                n_results=top_k,
                where=where,
                include=["metadatas", "documents", "distances"],
            )
            merged: list[dict[str, Any]] = []
            ids_list = results.get("ids", [[]])[0] or []
            distances_list = results.get("distances", [[]])[0] or []
            metadatas_list = results.get("metadatas", [[]])[0] or []
            documents_list = results.get("documents", [[]])[0] or []

            for i, hid in enumerate(ids_list):
                if not hid:
                    continue
                entity = self._build_entity(
                    hid, metadatas_list[i] if i < len(metadatas_list) else {},
                    documents_list[i] if i < len(documents_list) else "",
                )
                merged.append({
                    "id": str(hid),
                    "score": float(distances_list[i]) if i < len(distances_list) else 0.0,
                    "entity": entity,
                })
            return merged
        except Exception as exc:
            log.exception("图片检索失败：%s", exc)
            raise

    # ------------------------ 对比专用：全量拉取 ------------------------

    def get_all_by_doc_id(
        self,
        doc_id: str,
        modalities: Optional[tuple[str, ...]] = None,
        page_size: int = 100,
        max_chunks: int = 500,
    ) -> list[dict[str, Any]]:
        """按 doc_id 全量拉取所有 chunk（不走 query 语义匹配）。

        对比场景专用：保证完整覆盖文档全部内容，避免 query-based 检索的漏召。

        Args:
            doc_id: 目标文档 ID。
            modalities: 模态过滤，None=全量；如 ("text","table") 仅返回非图片 chunk。
            page_size: 单次分页大小（ChromaDB 内部 limit）。
            max_chunks: 安全上限，防止单文档超长时返回过大。

        Returns:
            标准化 dict 列表（与 ``hybrid_search`` 输出一致），按 page 升序排序。
        """
        if self._client is None:
            log.warning("ChromaDB 未连接，get_all_by_doc_id 返回空列表")
            return []
        if not doc_id:
            return []

        # 只查 document_chunks（text/table 模态 + image 模态的 dense_vector 副本）
        # image_chunks collection 的 image_vector 在文本对比时用不到，跳过
        targets = [self._doc_collection] if self._doc_collection else []

        all_hits: list[dict[str, Any]] = []
        for coll in targets:
            if coll is None:
                continue
            offset = 0
            while len(all_hits) < max_chunks:
                try:
                    batch = coll.get(
                        where={"doc_id": doc_id},
                        limit=page_size,
                        offset=offset,
                        include=["metadatas", "documents"],
                    )
                except Exception as exc:
                    log.warning("get_all_by_doc_id 拉取失败: %s", exc)
                    break

                ids = batch.get("ids", []) or []
                if not ids:
                    break

                metadatas = batch.get("metadatas", []) or []
                documents = batch.get("documents", []) or []

                for i, hid in enumerate(ids):
                    if len(all_hits) >= max_chunks:
                        break
                    if not hid:
                        continue
                    meta = metadatas[i] if i < len(metadatas) else {}
                    doc = documents[i] if i < len(documents) else ""
                    # 模态过滤
                    mod = (meta or {}).get("modality", "text")
                    if modalities and mod not in modalities:
                        continue
                    entity = self._build_entity(hid, meta, doc)
                    all_hits.append({
                        "id": str(hid),
                        # 全量拉取无 score 概念，给一个常量便于排序兼容
                        "score": 1.0,
                        "entity": entity,
                    })

                if len(ids) < page_size:
                    break  # 已到末尾
                offset += page_size

        # 按 page 升序排序（page 缺失时排到末尾），保证 LLM 看到文档顺序
        def _sort_key(h: dict) -> tuple[int, int, str]:
            try:
                pg = int(h.get("entity", {}).get("page", 0) or 0)
            except (TypeError, ValueError):
                pg = 0
            mod = h.get("entity", {}).get("modality", "text")
            # text/table 优先（便于 LLM 优先看正文），再 page 升序
            mod_rank = 0 if mod != "image" else 1
            return (mod_rank, pg, h.get("id", ""))

        all_hits.sort(key=_sort_key)
        return all_hits

    def cross_modal_search(
        self,
        query_dense: list[float],
        query_image_vector: Optional[list[float]] = None,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """跨模态检索：文本 query + 可选 image 向量。"""
        rrf_scores: dict[str, float] = {}
        hit_entities: dict[str, dict[str, Any]] = {}

        # 1) 文本
        try:
            results = self._doc_collection.query(
                query_embeddings=[query_dense],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
            )
            self._accumulate_rrf(results, rrf_scores, hit_entities)
        except Exception as exc:
            log.warning("文本检索失败: %s", exc)

        # 2) 图片（可选）
        if query_image_vector:
            try:
                img_results = self._image_collection.query(
                    query_embeddings=[query_image_vector],
                    n_results=top_k,
                    include=["metadatas", "documents", "distances"],
                )
                self._accumulate_rrf(img_results, rrf_scores, hit_entities)
            except Exception as exc:
                log.warning("图片检索失败: %s", exc)

        merged: list[dict[str, Any]] = [
            {"id": hid, "score": rrf_scores[hid], "entity": hit_entities[hid]}
            for hid in rrf_scores
        ]
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    @staticmethod
    def _accumulate_rrf(
        results: dict[str, Any],
        rrf_scores: dict[str, float],
        hit_entities: dict[str, dict[str, Any]],
    ) -> None:
        """将 ChromaDB query 返回结果累加进 RRF 表。

        ChromaDB query 返回格式：
            {"ids": [[id1, id2, ...]], "distances": [[d1, d2, ...]],
             "metadatas": [[{...}, {...}]], "documents": [["doc1", "doc2"]]}
        """
        ids_list = results.get("ids", [[]])
        distances_list = results.get("distances", [[]])
        metadatas_list = results.get("metadatas", [[]])
        documents_list = results.get("documents", [[]])

        if not ids_list or not ids_list[0]:
            return

        for rank, hid in enumerate(ids_list[0], start=1):
            if not hid:
                continue
            hid = str(hid)
            rrf_scores[hid] = rrf_scores.get(hid, 0.0) + 1.0 / (_RRF_K + rank)
            if hid not in hit_entities:
                meta = metadatas_list[0][rank - 1] if rank - 1 < len(metadatas_list[0]) else {}
                doc = documents_list[0][rank - 1] if rank - 1 < len(documents_list[0]) else ""
                hit_entities[hid] = ChromaClientSingleton._build_entity(hid, meta, doc)

    @staticmethod
    def _build_entity(hit_id: str, meta: dict, doc: str) -> dict[str, Any]:
        """从 ChromaDB metadata + document 重建 entity 字典（兼容原有 retriever 接口）。"""
        # 解析 metadata_json 还原原始 metadata
        metadata_json = meta.get("metadata_json", "{}")
        try:
            parsed_meta = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        except (json.JSONDecodeError, TypeError):
            parsed_meta = {}

        return {
            "id": str(hit_id),
            "doc_id": meta.get("doc_id", ""),
            "file_name": meta.get("file_name", ""),
            "modality": meta.get("modality", "text"),
            "content": doc or "",
            "searchable_text": meta.get("searchable_text", ""),
            "page": meta.get("page", 0),
            "section": meta.get("section", ""),
            "metadata": parsed_meta if isinstance(parsed_meta, dict) else {},
            # ChromaDB 顶层 metadata 字段（不在 metadata_json 内，需显式回传）
            "parent_chunk_id": meta.get("parent_chunk_id", ""),
            "image_path": meta.get("image_path", ""),
            "chunk_role": meta.get("chunk_role", "child"),
            "is_table_only": bool(meta.get("is_table_only", False)),
            # 章节结构字段（章节路由 / 按章节检索用，Task 2 新增）
            "section_id": meta.get("section_id", "") or "",
            "section_path": meta.get("section_path", "") or "",
            "section_topic": meta.get("section_topic", "") or "",
        }

    @staticmethod
    def _build_where(filter_expr: Optional[str]) -> Optional[dict[str, Any]]:
        """将 filter_expr 转为 ChromaDB where 格式。

        支持的格式：
            - ``id != ''`` → None（匹配所有）
            - ``doc_id == "abc"`` → {"doc_id": "abc"}（字符串字段：doc_id / file_name /
              modality / section_id / section_topic / chunk_role 等）
            - ``file_name == "test.pdf"`` → {"file_name": "test.pdf"}
            - ``file_name $contains "keyword"`` → {"file_name": {"$contains": "keyword"}}
            - ``is_table_only == true`` → {"is_table_only": True}（bool 字段，值必须为
              Python 的 True/False 而非 1/0 或 "true"/"false"）

        注意：ChromaDB where 中的 bool 字段精确匹配必须传 Python bool 值，字符串无法匹配。
        """
        if not filter_expr or filter_expr == "id != ''":
            return None
        # $contains 语法
        match = re.match(r'(\w+)\s*\$contains\s*"([^"]*)"', filter_expr)
        if match:
            return {match.group(1): {"$contains": match.group(2)}}
        # == 语法（字符串值，带双引号）
        match = re.match(r'(\w+)\s*==\s*"([^"]*)"', filter_expr)
        if match:
            return {match.group(1): match.group(2)}
        # == 语法（bool 值，不带引号）：is_table_only == true / false
        match = re.match(r'(\w+)\s*==\s*(true|false)\b', filter_expr)
        if match:
            return {match.group(1): match.group(2) == "true"}
        return None

    # ------------------------ 查询 ------------------------

    def get_by_id(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """按 chunk_id 从 document_chunks collection 回查单条记录。

        Returns:
            ``{"id": str, "content": str, "metadata": dict, ...}`` 或 None。
        """
        if self._doc_collection is None:
            return None
        try:
            results = self._doc_collection.get(
                ids=[chunk_id],
                include=["metadatas", "documents"],
            )
            if not results or not results.get("ids"):
                return None
            meta = results["metadatas"][0] if results.get("metadatas") else {}
            doc = results["documents"][0] if results.get("documents") else ""
            return self._build_entity(chunk_id, meta, doc)
        except Exception as exc:
            log.warning("get_by_id 失败: %s", exc)
            return None

    # ------------------------ 删除 ------------------------

    def _delete_by_filter(self, collection: Any, where: dict[str, Any]) -> int:
        """按条件删除 collection 中的记录。"""
        if collection is None:
            return 0
        try:
            # 先查询要删除的 id
            existing = collection.get(where=where)
            if not existing or not existing.get("ids"):
                return 0
            count = len(existing["ids"])
            collection.delete(where=where)
            log.info("按条件删除：%s, 命中 %d 条", collection.name, count)
            return count
        except Exception as exc:
            log.exception("按条件删除失败 %s: %s", collection.name, exc)
            raise

    def delete_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除所有 collection 中的相关记录。"""
        if self._client is None:
            return 0
        total = 0
        where = {"doc_id": doc_id}
        for coll in (self._doc_collection, self._image_collection):
            try:
                total += self._delete_by_filter(coll, where)
            except Exception:
                pass
        if total > 0:
            self.invalidate_list_cache()
        log.info("按 doc_id 删除：%s, 命中 %d 条", doc_id, total)
        return total

    def delete_by_file_name(self, file_name: str) -> int:
        """跨 collection 删除同名文件的所有 chunk。"""
        if self._client is None:
            return 0
        total = 0
        where = {"file_name": file_name}
        for coll in (self._doc_collection, self._image_collection):
            try:
                total += self._delete_by_filter(coll, where)
            except Exception:
                pass
        if total > 0:
            self.invalidate_list_cache()
        return total

    # ------------------------ 统计 / 列表 ------------------------

    def list_documents(self) -> list[dict[str, Any]]:
        """列出所有唯一 file_name 及其 chunk 数量（跨 collection 合并）。

        同时带回每个 file_name 对应的 ``doc_id``：``/compare/*`` 等接口按 doc_id
        定位文档，调用方（如平台 EMC 工具）只有文件名，需要映射才能用。

        带 30 秒 TTL 缓存：/documents 端点可能高频调用，避免每次 O(N) 全表扫描。
        """
        if self._client is None:
            log.warning("ChromaDB 未连接，list_documents 返回空列表")
            return []

        # 缓存命中检查（避免重复 O(N) 扫描）
        now = time.time()
        with self._list_cache_lock:
            if (
                self._list_cache is not None
                and now - self._list_cache_ts < self._LIST_CACHE_TTL
            ):
                return list(self._list_cache)

        counter: dict[str, int] = {}
        doc_ids: dict[str, str] = {}
        for coll in (self._doc_collection, self._image_collection):
            if coll is None:
                continue
            try:
                results = coll.get(include=["metadatas"])
                if not results or not results.get("metadatas"):
                    continue
                for meta in results["metadatas"]:
                    fn = meta.get("file_name", "") if meta else ""
                    if fn:
                        counter[fn] = counter.get(fn, 0) + 1
                        # 同一文件名可能因重复上存在多个 doc_id，取首个（与评测脚本一致）
                        if not doc_ids.get(fn) and meta.get("doc_id"):
                            doc_ids[fn] = str(meta["doc_id"])
            except Exception as exc:
                log.warning("list_documents %s 失败: %s", coll.name, exc)
        result = [
            {"file_name": fn, "chunk_count": cnt, "doc_id": doc_ids.get(fn, "")}
            for fn, cnt in sorted(counter.items())
        ]
        with self._list_cache_lock:
            self._list_cache = result
            self._list_cache_ts = now
        return result

    def invalidate_list_cache(self) -> None:
        """主动失效 list_documents 缓存（摄入/删除文档后调用）。"""
        with self._list_cache_lock:
            self._list_cache = None
            self._list_cache_ts = 0.0

    def clear_all(self) -> int:
        """清空本机两个 collection，仅供显式确认的离线索引重建使用。"""
        total = 0
        for collection in (self._doc_collection, self._image_collection):
            if collection is None:
                continue
            results = collection.get()
            ids = results.get("ids", []) if results else []
            if ids:
                collection.delete(ids=ids)
                total += len(ids)
        self.invalidate_list_cache()
        return total

    def get_table_chunks_by_docs(
        self, file_names: list[str], limit: int = 3,
    ) -> list[dict[str, Any]]:
        """按文档集合拉取 table 模态片段（供检索补召回，避免参数表被图片挤掉）。"""
        if self._doc_collection is None or not file_names:
            return []
        try:
            where = {
                "$and": [
                    {"file_name": {"$in": list(file_names)}},
                    {"modality": "table"},
                ]
            }
            data = self._doc_collection.get(
                where=where, limit=limit, include=["metadatas", "documents"],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("get_table_chunks_by_docs 失败: %s", exc)
            return []
        out: list[dict[str, Any]] = []
        for i, cid in enumerate(data.get("ids", [])):
            md = (data.get("metadatas") or [])[i] or {}
            doc = (data.get("documents") or [])[i] or ""
            out.append({
                "id": cid,
                "doc_id": md.get("doc_id", ""),
                "file_name": md.get("file_name", ""),
                "modality": "table",
                "content": doc,
                "searchable_text": doc,
                "page": md.get("page", 0),
                "score": 1.0,
                "metadata": dict(md),
            })
        return out
        with self._list_cache_lock:
            self._list_cache = None
            self._list_cache_ts = 0.0

    def get_collection_stats(self) -> dict[str, Any]:
        """返回所有 collection 的统计（text/table + image）。"""
        total = 0
        names = []
        for coll in (self._doc_collection, self._image_collection):
            if coll is None:
                continue
            try:
                total += coll.count()
                names.append(coll.name)
            except Exception as exc:
                log.warning("get_collection_stats %s 失败: %s", coll.name, exc)
        return {"names": names, "row_count": total}


# 导出别名
ChromaClient = ChromaClientSingleton
