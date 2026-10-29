"""
多模态 RAG 系统 — 查询 Pipeline (Task 7)

职责：
    1. 编排 retriever → (reranker) → generator 的完整 RAG 流程
    2. 同时提供同步 ``query`` 与流式 ``query_stream`` 两种入口
    3. 暴露给 API 层的标准化返回结构（answer / sources / language / used_rerank）
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from utils.runtime_budget import submit_context, cache_namespace
from typing import Any, Iterator, Optional

from config import config
from core.embedder import Embedder
from core.generator import Generator
from core.query_rewriter import QueryRewriter, get_colloquial_pattern
from core.reranker import Reranker
from core.retriever import Retriever
from core.standard_registry import (
    STANDARD_FAMILIES,
    alias_file_fragments,
    detect_families,
    family_member_files,
    norm as norm_text,
)
from db.chroma_client import ChromaClientSingleton
from rag.service import RagService
from tools.base import escape_filter_value
from utils.language_detector import detect_language
from utils.logger import log

# 产品型号提取正则（匹配类似 DEMO27-A、DEMO24-B 等型号）
_PRODUCT_MODEL_RE = re.compile(
    r"\b([A-Z]{2,}\d[\w-]*\d|\d{3}\s*\d{3}\s*\d{3})\b", re.IGNORECASE
)

# ???????????/??????????????
_COMPARE_HINTS = (
    "\u5bf9\u6bd4", "\u6bd4\u8f83", "\u5f02\u540c", "\u533a\u522b",
    "\u5dee\u5f02", "\u6709\u4ec0\u4e48\u4e0d\u540c", "\u4e0d\u540c\u70b9",
    "versus", " vs ",
)
_COMPARE_ALIASES = {}


_COMPARE_SYSTEM_PROMPT = (
    "\u4f60\u662f\u786c\u4ef6\u884c\u4e1a\u7684 EMC \u6807\u51c6\u5bf9\u6bd4\u4e13\u5bb6\u3002"
    "\u5f53\u7528\u6237\u8981\u6c42\u5bf9\u6bd4\u4e24\u4efd\u6807\u51c6/\u7248\u672c\u65f6\uff0c\u8bf7\u4e25\u683c\u6309\u4ee5\u4e0b\u7ed3\u6784\u8f93\u51fa\uff1a\n"
    "\u4e00\u3001\u6807\u51c6 A \u8bd5\u9a8c\u9879\u76ee\u6e05\u5355\uff1a\u9010\u9879\u5217\u51fa\uff08\u542b\u4e3b\u8981\u53c2\u6570/\u9650\u503c/\u5224\u5b9a\u8981\u6c42\uff09\u3002\n"
    "\u4e8c\u3001\u6807\u51c6 B \u8bd5\u9a8c\u9879\u76ee\u6e05\u5355\uff1a\u9010\u9879\u5217\u51fa\u3002\n"
    "\u4e09\u3001\u9010\u9879\u5bf9\u6bd4\uff1a\u5bf9\u6bcf\u4e00\u7c7b\u8bd5\u9a8c\uff08\u8f90\u5c04\u53d1\u5c04/\u4f20\u5bfc\u53d1\u5c04/BCI/ESD/"
    "\u77ac\u6001/\u6d6a\u6d8c/\u8c10\u6ce2/\u78c1\u573a\u7b49\uff09\u5bf9\u6bd4\u4e24\u8fb9\u7684\u9891\u6bb5\u3001\u7b49\u7ea7\u3001\u9650\u503c\u3001\u5224\u5b9a\uff0c"
    "\u5e76\u6807\u6ce8 [n] \u6765\u6e90\u3002\n"
    "\u56db\u3001\u5dee\u5f02\u6c47\u603b\uff1a\u5217\u51fa\u4e3b\u8981\u5dee\u5f02\u4e0e\u7ed3\u8bba\u3002\n"
    "\u89c4\u5219\uff1a\u4ec5\u57fa\u4e8e\u68c0\u7d22\u4e0a\u4e0b\u6587\uff0c\u7981\u6b62\u7f16\u9020\uff1b\u67d0\u4efd\u6807\u51c6\u8d44\u6599\u7f3a\u5931\u67d0\u53c2\u6570\u65f6"
    "\u5982\u5b9e\u8bf4\u660e\u201c\u8be5\u7248\u672c\u8d44\u6599\u672a\u63d0\u4f9b\u201d\u5e76\u6807\u6ce8\u53ef\u67e5\u4f4d\u7f6e\uff1b"
    "\u6bcf\u4e2a\u6570\u503c\u4e0e\u7ed3\u8bba\u540e\u5fc5\u987b\u6807\u6ce8 [n]\uff1b\u5f15\u7528\u4e0d\u8db3\u4f1a\u663e\u8457\u6263\u5206\uff1b"
    "\u82e5\u6807\u51c6\u88ab\u62c6\u5206\u4e3a\u591a\u4e2a\u6587\u6863\uff0c\u8bf7\u8de8\u5168\u90e8\u76f8\u5173\u6587\u6863\u6c47\u603b\u3002"
)

# 单试验项对比触发词（BCI/ESD/RE/浪涌等 item 类问题走定向检索）
_ITEM_KEYWORDS = (
    "BCI", "大电流注入", "ESD", "静电放电", "辐射发射", "传导发射",
    "浪涌", "谐波", "瞬态", "磁场", "RE", "CE",
)

_EXTRACT_PROFILE_PROMPT = (
    "你是 EMC 标准分析专家。以下内容摘自《{label}》（该标准可能被拆分为多个文件保存，"
    "每段参考资料都带有 [n] 编号和文件/章节信息）。\n"
    "请提取该标准的【试验项目完整清单】：\n"
    "1. 逐项列出试验项目名称；\n"
    "2. 每项给出主要参数/限值/判定要求（频率范围、等级、限值、功能等级、试验方法要点等，"
    "能提取多少写多少，数值必须与参考资料一致）；\n"
    "3. 每项注明来源文件名。\n"
    "规则：只基于给定内容，禁止编造；内容未覆盖的参数不要填写，可写“未提供”；"
    "如果该标准被拆成多个文件，必须跨全部相关文件汇总，不能只列一部分。\n"
    "输出格式（Markdown 文本）：\n"
    "### 试验项目清单\n"
    "1. 项目名称 — 参数：…；判定：…（来源：文件名）\n"
    "2. …"
)

_COMPARE_STAGE_B_PROMPT = (
    "你是 EMC 跨标准对比专家。用户要求对比两份标准：{label_a} 与 {label_b}。\n\n"
    "【标准A 试验项目清单（阶段A 提取）】\n{profile_a}\n\n"
    "【标准B 试验项目清单（阶段A 提取）】\n{profile_b}\n\n"
    "【参考资料】下方 [1]..[N] 为两份标准的相关原文片段（含文件名/页码/章节），"
    "用于核验数值并标注引用。\n\n"
    "请输出：\n"
    "一、标准A 试验项目清单（简列，每项带 [n] 引用）；\n"
    "二、标准B 试验项目清单（简列，每项带 [n] 引用）；\n"
    "三、逐项对比：对每个试验项目对比两边的频率范围、限值、等级、判定要求，"
    "每个关键数值后标注 [n]；\n"
    "四、差异汇总。\n"
    "规则：\n"
    "- 仅基于清单与参考资料，禁止编造；某参数两份资料都未提供时，"
    "明确写“该版本资料未提供”，不要放弃该项对比；\n"
    "- 引用编号 [n] 必须指向【参考资料】编号，引用不足会显著扣分；\n"
    "- 若标准被拆分为多个文件，须跨全部相关文件汇总后再对比。"
)

_ENUM_FORMAT_PROMPT = (
    "你是 EMC 标准分析专家。以下是从《{label}》（拆分为多个文件保存）中提取的试验项目清单。\n"
    "请整理成完整、连贯的答案：\n"
    "1. 先说明该标准共分为多少个部分；\n"
    "2. 完整列出全部部分名称（一项不漏）；\n"
    "3. 每项标注 [n] 引用（指向下方参考资料编号）。\n"
    "规则：只使用给定清单与参考资料内容，不得编造。\n\n"
    "【试验项目清单】\n{profile}"
)


class QueryPipeline:
    """查询编排：retrieve → (rerank) → generate。"""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        reranker: Optional[Reranker] = None,
        generator: Optional[Generator] = None,
        embedder: Optional[Embedder] = None,
        query_rewriter: Optional[QueryRewriter] = None,
    ) -> None:
        # 默认从 Embedder + ChromaDB 单例组合一个 retriever
        self._embedder = embedder
        self._chroma: Optional[ChromaClientSingleton] = None
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._query_rewriter = query_rewriter

        # RagService 实例（供工具层消费，延迟构造）
        self._rag_service: Optional[RagService] = None

        # 查询结果缓存：避免相同 query 重复走检索→重排→生成全链路
        self._query_cache: dict[str, tuple[dict, float]] = {}
        self._query_cache_lock = threading.Lock()
        self._query_cache_ttl: float = 300.0  # 5 分钟
        self._query_cache_max: int = 128

        # 改写结果缓存：原 query → 改写后的 query 列表（含原 query 在最前）
        # 单独缓存改写结果，避免每次 query 都触发 LLM 调用
        self._rewrite_cache: dict[str, tuple[list[str], float]] = {}
        self._rewrite_cache_lock = threading.Lock()

        # 两阶段对比：阶段A profile 缓存（族/版本组 + query → 提取结果）
        self._profile_cache: dict[str, tuple[str, float]] = {}
        self._profile_cache_lock = threading.Lock()

    # ---------- 懒加载属性 ----------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def chroma(self) -> Optional[ChromaClientSingleton]:
        if self._chroma is None:
            try:
                self._chroma = ChromaClientSingleton.get_instance()
            except Exception as e:  # noqa: BLE001
                log.warning("ChromaDB 不可用: %s", e)
                self._chroma = None
        return self._chroma

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            # 默认构造：基于懒加载的 chroma + embedder
            # chroma 可能为 None（连接失败），Retriever 内部已做容错
            self._retriever = Retriever(self.chroma, self.embedder)
        return self._retriever

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator()
        return self._generator

    @property
    def query_rewriter(self) -> QueryRewriter:
        if self._query_rewriter is None:
            self._query_rewriter = QueryRewriter()
        return self._query_rewriter

    @property
    def rag_service(self) -> RagService:
        """统一的 RagService 实例（供工具层消费）。

        与 API 层共享同一个底层存储/检索组件实例，避免重复创建。
        """
        if self._rag_service is None:
            self._rag_service = RagService(
                chroma=self.chroma,
                embedder=self.embedder,
                retriever=self.retriever,
            )
        return self._rag_service

    # ---------- 公开方法 ----------

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: bool = True,
        language: Optional[str] = None,
        filter_expr: Optional[str] = None,
        generate: bool = True,
    ) -> dict[str, Any]:
        """同步 RAG 流程。

        流程：
            1. 语种检测
            2. 检索（含自动型号过滤）
            3. 重排序（可选）
            4. 生成答案

        Args:
            filter_expr: ChromaDB 过滤表达式，如 ``doc_id == "abc"``。
            generate: ``False`` 时只做检索+重排，不调用生成模型（供工具层复用，
                避免"检索一次答案、Agent 再生成一次"的重复开销）。
        """
        try:
            lang = "zh"

            # ---- 查询级缓存：相同参数命中则直接返回，跳过全链路 ----
            cache_key = self._cache_key(
                query, top_k or config.RETRIEVAL_TOP_K, use_rerank, lang, filter_expr or "",
            ) + ("|answer" if generate else "|retrieve")
            with self._query_cache_lock:
                entry = self._query_cache.get(cache_key)
                if entry is not None:
                    result, ts = entry
                    if time.time() - ts < self._query_cache_ttl:
                        log.info("查询缓存命中: %.40s...", query)
                        return dict(result)

            # ---- 普通 RAG 模式 ----
            # candidates 在以下两种路径中填充：
            #   a) requery 触发 → multi_query_retrieve（多路召回 + RRF 融合）
            #   b) requery 未触发 → _retrieve_with_doc_filter（单 query + 型号过滤）
            candidates: list[dict] = []
            doc_filter: Optional[str] = None
            # 型号过滤（requery 与单 query 两条路径共用）
            compare_files = self._detect_compare_docs(query)
            family_keys = self._detect_compare_families(query)
            two_stage_result: Optional[dict] = None
            if family_keys and config.TWO_STAGE_COMPARE_ENABLED and generate:
                try:
                    two_stage_result = self._two_stage_compare(
                        query, family_keys, top_k, lang,
                    )
                    log.info(
                        "two-stage compare 触发: families=%s groups=%d contexts=%d",
                        ",".join(family_keys),
                        len(two_stage_result.get("groups", [])),
                        len(two_stage_result.get("contexts", [])),
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning("two-stage compare 失败，回退单次对比: %s", e)
                    two_stage_result = None
            if two_stage_result is None and family_keys:
                candidates = self._retrieve_compare_families(family_keys, query, top_k)
            elif two_stage_result is None and compare_files:
                candidates = self._retrieve_compare(compare_files, query, top_k)
            matched_files = self._match_doc_files(query)
            if (
                two_stage_result is None
                and not family_keys
                and not compare_files
                and filter_expr is None
                and self._should_rewrite(query)
            ):
                # ---- 阶段1.5: requery（仅普通模式 + 无 filter 时触发） ----
                try:
                    rewrites = self._get_rewrites(query, lang)
                    if len(rewrites) > 1:
                        candidates = self.retriever.multi_query_retrieve(
                            queries=rewrites,
                            top_k=top_k,
                            filter_expr=None,
                            fusion_k=config.REQUERY_FUSION_K,
                        )
                        log.info(
                            "requery 触发: %d 路融合 → %d chunk (query=%.40s...)",
                            len(rewrites), len(candidates), query,
                        )
                        if matched_files and candidates:
                            before = len(candidates)
                            candidates = [
                                c for c in candidates
                                if (c.get("file_name") or "") in matched_files
                            ]
                            candidates = self._supplement_tables(candidates, matched_files)
                            log.info(
                                "requery 型号过滤+表格补召回: %d → %d (docs=%d)",
                                before, len(candidates), len(matched_files),
                            )
                except Exception as e:  # noqa: BLE001
                    # 任何失败（LLM 改写失败 / 多路检索失败）→ 降级为单 query 检索
                    log.warning("requery 失败，降级为单 query 检索: %s", e)
                    candidates = []

            # 降级路径：requery 未触发或失败
            if two_stage_result is None and not candidates:
                candidates, doc_filter = self._retrieve_with_doc_filter(
                    query, top_k, filter_expr,
                )
            if two_stage_result is None and matched_files and not family_keys:
                candidates = self._supplement_matched_files(
                    candidates, matched_files, query, top_k,
                )

            if two_stage_result is not None:
                contexts = two_stage_result["contexts"]
                answer = two_stage_result["answer"]
                used_rerank = bool(two_stage_result.get("used_rerank", False))
            else:
                final_top_k = top_k if top_k else config.RERANK_TOP_K
                compare_intent = bool(family_keys or compare_files) or any(
                    h in query for h in _COMPARE_HINTS
                )
                if compare_intent:
                    final_top_k = max(final_top_k, 16)
                used_rerank = False
                if use_rerank and candidates and len(candidates) >= min(final_top_k, 2):
                    candidates = self.reranker.rerank(
                        query, candidates, top_k=max(final_top_k, config.RERANK_TOP_K),
                    )
                    used_rerank = True

                contexts = candidates[:final_top_k]
                if not generate:
                    answer = ""
                elif compare_intent:
                    answer = self.generator.generate(
                        query, contexts, language=lang, system_prompt=_COMPARE_SYSTEM_PROMPT,
                    )
                else:
                    answer = self.generator.generate(query, contexts, language=lang)

            result = {
                "answer": answer,
                "sources": [self._source_summary(c) for c in contexts],
                "image_chunks": self._collect_image_chunks(contexts),
                "language": lang,
                "used_rerank": used_rerank,
            }
            # 写入查询缓存（FIFO 淘汰：超过 max 时清理最旧的一半）
            with self._query_cache_lock:
                if len(self._query_cache) >= self._query_cache_max:
                    evict_count = self._query_cache_max // 2
                    keys = list(self._query_cache.keys())[:evict_count]
                    for k in keys:
                        del self._query_cache[k]
                self._query_cache[cache_key] = (dict(result), time.time())
            return result
        except Exception as e:  # noqa: BLE001
            log.exception("查询失败: %s", e)
            return {
                "answer": f"查询失败: {e}",
                "error": str(e),
                "sources": [],
                "image_chunks": [],
                "language": language or "en",
                "used_rerank": False,
            }

    def query_stream(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: bool = True,
        language: Optional[str] = None,
        filter_expr: Optional[str] = None,
    ) -> Iterator[str]:
        """流式 RAG：仅 yield 答案文本片段。

        检索 + 重排序与同步版一致；生成阶段使用 ``generate_stream``。
        """
        lang = "zh"

        # ---- 查询缓存（与 query() 共享，统一使用 _cache_key 生成键） ----
        cache_key = self._cache_key(
            query or "", top_k or config.RETRIEVAL_TOP_K, use_rerank, lang, filter_expr or "",
        )
        with self._query_cache_lock:
            entry = self._query_cache.get(cache_key)
            if entry is not None:
                result, ts = entry
                if time.time() - ts < self._query_cache_ttl:
                    log.info("stream 缓存命中: %.40s...", query)
                    yield result.get("answer", "")
                    return

        # ---- 普通流式 ----
        # candidates 在以下两种路径中填充：
        #   a) requery 触发 → multi_query_retrieve（多路召回 + RRF 融合）
        #   b) requery 未触发或失败 → _retrieve_with_doc_filter（单 query）
        candidates: list[dict] = []
        compare_files = self._detect_compare_docs(query)
        family_keys = self._detect_compare_families(query)
        if family_keys:
            candidates = self._retrieve_compare_families(family_keys, query, top_k)
        elif compare_files:
            candidates = self._retrieve_compare(compare_files, query, top_k)
        matched_files = self._match_doc_files(query)
        if not family_keys and not compare_files and filter_expr is None and self._should_rewrite(query):
            try:
                rewrites = self._get_rewrites(query, lang)
                if len(rewrites) > 1:
                    candidates = self.retriever.multi_query_retrieve(
                        queries=rewrites,
                        top_k=top_k,
                        filter_expr=None,
                        fusion_k=config.REQUERY_FUSION_K,
                    )
                    log.info(
                        "stream requery 触发: %d 路融合 → %d chunk (query=%.40s...)",
                        len(rewrites), len(candidates), query,
                    )
                    if matched_files and candidates:
                        before = len(candidates)
                        candidates = [
                            c for c in candidates
                            if (c.get("file_name") or "") in matched_files
                        ]
                        candidates = self._supplement_tables(candidates, matched_files)
                        log.info(
                            "stream requery 型号过滤+表格补召回: %d → %d (docs=%d)",
                            before, len(candidates), len(matched_files),
                        )
            except Exception as e:  # noqa: BLE001
                log.warning("stream requery 失败，降级为单 query 检索: %s", e)
                candidates = []

        if not candidates:
            try:
                candidates, _ = self._retrieve_with_doc_filter(query, top_k, filter_expr)
            except Exception as e:  # noqa: BLE001
                log.warning("stream 检索失败: %s", e)
                candidates = []
        if matched_files and not family_keys:
            candidates = self._supplement_matched_files(
                candidates, matched_files, query, top_k,
            )

        final_top_k = top_k if top_k else config.RERANK_TOP_K
        compare_intent = bool(family_keys or compare_files) or any(
            h in query for h in _COMPARE_HINTS
        )
        if compare_intent:
            final_top_k = max(final_top_k, 16)
        if use_rerank and candidates and len(candidates) >= min(final_top_k, 2):
            try:
                candidates = self.reranker.rerank(
                    query, candidates, top_k=max(final_top_k, config.RERANK_TOP_K),
                )
            except Exception as e:  # noqa: BLE001
                log.warning("stream 重排失败: %s", e)

        contexts = candidates[:final_top_k]
        gen_kwargs = (
            {"system_prompt": _COMPARE_SYSTEM_PROMPT} if compare_intent else {}
        )
        for piece in self.generator.generate_stream(
            query, contexts, language=lang, **gen_kwargs,
        ):
            yield piece

        # 产出检索到的图片 chunk 引用（三视图/封装图等），供前端渲染
        # 图片 chunk 可能出现在两个位置：
        #   1. contexts 顶层（直接检索到的 image 类型 chunk）
        #   2. linked_chunks 中（通过 _link_related 图文关联回查到的图片 chunk）
        image_chunks = self._collect_image_chunks(contexts)
        if image_chunks:
            yield f"__IMAGE_CHUNKS__:{json.dumps(image_chunks, ensure_ascii=False)}"

    def image_search(
        self,
        image_bytes: bytes,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """以图搜图：把图片编码后到 image_chunks collection 检索。

        失败说明：若 OneAPI 的 ``IMAGE_EMBED_MODEL`` 不支持 OpenAI 兼容模式，
        编码会返回 None，此时本方法返回空列表。

        Args:
            image_bytes: 原始图片 bytes。
            top_k: 返回前 k 个相似图片。

        Returns:
            ``[{"id": str, "file_name": str, "page": int, "score": float, "content": str}, ...]``
        """
        if not image_bytes or self.chroma is None:
            return []
        try:
            vec = self.embedder.image.embed(image_bytes)
        except Exception as e:  # noqa: BLE001
            log.warning("image embed 失败: %s", e)
            return []
        if not vec:
            return []
        try:
            hits = self.chroma.image_search(vec, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            log.warning("image_search 失败: %s", e)
            return []
        return [
            {
                "id": h["id"],
                "file_name": (h.get("entity") or {}).get("file_name", ""),
                "page": (h.get("entity") or {}).get("page", 0),
                "modality": (h.get("entity") or {}).get("modality", "image"),
                "content": ((h.get("entity") or {}).get("content") or "")[:240],
                "score": h.get("score", 0.0),
            }
            for h in hits
        ]

    # ---------- 内部方法 ----------

    def _retrieve_with_doc_filter(
        self, query: str, top_k: Optional[int], filter_expr: Optional[str],
    ) -> tuple[list[dict], Optional[str]]:
        """统一的"检索 + 型号过滤"封装，供 query() / query_stream() 复用。

        行为：
            - 提取 query 中的产品型号（仅 1 个时启用）
            - 有 doc_filter 时把 retriever top_k 扩大到 3x，避免被 ChromaDB 截断后再过滤
            - 在 Python 层按 file_name 做子串匹配（ChromaDB $contains 对中文/特殊字符不可靠）

        Returns:
            (candidates, doc_filter) 元组；doc_filter 为 None 表示未启用型号过滤。
        """
        # 自动检测 query 中的产品型号（匹配已入库文档）
        doc_filter: Optional[str] = None
        matched_files: Optional[list[str]] = None
        if not filter_expr:
            matched_files = self._match_doc_files(query)
            if matched_files:
                doc_filter = ",".join(matched_files)
                log.info("自动文档过滤(模型): %s", doc_filter)

        # 有 doc_filter 时扩大 top_k，避免被截断后再过滤导致召回不足
        fetch_k = top_k
        if doc_filter:
            fetch_k = (top_k or config.RETRIEVAL_TOP_K) * 3
        candidates = self.retriever.retrieve(
            query, top_k=fetch_k, filter_expr=filter_expr,
        )

        # Python 层按文档集合过滤 file_name
        if matched_files and candidates:
            before = len(candidates)
            candidates = [
                c for c in candidates
                if (c.get("file_name", "") or "") in matched_files
            ]
            candidates = self._supplement_tables(candidates, matched_files)
            log.info("Python 层过滤: %d → %d (docs=%d)", before, len(candidates), len(matched_files))
        return candidates, doc_filter

    def _supplement_tables(
        self, candidates: list[dict], matched_files: list[str],
    ) -> list[dict]:
        """表格补召回：参数类查询的候选里表格片段不足时，从命中文档补拉 table 片段。"""
        if not matched_files or not self.chroma:
            return candidates
        have_tables = sum(1 for c in candidates if c.get("modality") == "table")
        if have_tables >= 3:
            return candidates
        try:
            extras = self.chroma.get_table_chunks_by_docs(
                matched_files, limit=max(3 - have_tables, 1),
            )
        except Exception:  # noqa: BLE001
            return candidates
        if not extras:
            return candidates
        seen = {c.get("id") for c in candidates}
        return candidates + [e for e in extras if e.get("id") not in seen]

    def _match_doc_files(self, query: str) -> Optional[list[str]]:
        """从 query 提取型号 token，匹配已入库文档文件名，返回命中文档列表。

        确定性规则（替代纯 LLM 判断）：只有 query 中的大写字母+数字 token 恰好出现在
        某份已入库文档名中才启用过滤；未匹配或匹配过多时返回 None（不过滤）。
        """
        if not query or not self.chroma:
            return None
        try:
            files = [d.get("file_name", "") for d in self.chroma.list_documents()]
        except Exception:  # noqa: BLE001
            return None
        files = [f for f in files if f]
        if not files:
            return None

        tokens = set(re.findall(r"[A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*", query.upper()))
        matched: list[str] = []
        for tok in tokens:
            for fn in files:
                if tok in fn.upper() and fn not in matched:
                    matched.append(fn)
        # Only explicitly configured aliases may expand file matches.
        if not matched:
            for frag in alias_file_fragments(query):
                for fn in files:
                    if frag in norm_text(fn) and fn not in matched:
                        matched.append(fn)
        return matched or None

    @staticmethod
    def _is_enumeration_query(query: str) -> bool:
        """判断是否为"完整枚举/汇总"类问题（触发标准族分解检索）。"""
        if not query:
            return False
        hints = (
            "多少个部分", "全部试验项目", "完整列出", "所有部分", "分别有哪些",
            "分别提取", "哪些试验项目", "逐一提取", "逐一列出", "分别是什么", "全部部分",
        )
        return any(h in query for h in hints)

    def _detect_compare_families(self, query: str) -> list[str]:
        """检测 query 引用的标准族（用于全标准对比/枚举的查询分解）。"""
        if not query or not self.chroma:
            return []
        keys = detect_families(query)
        if not keys:
            return []
        # 仅"全标准枚举/汇总"类问题触发族分解；单试验项对比（BCI/ESD/RE 参数）
        # 走原语义路径 + 别名补充，避免 31 个分拆文件的候选稀释掉目标试验项。
        full_intent = self._is_enumeration_query(query) or any(
            t in query for t in ("汇总", "逐项对比", "逐一对比", "逐一提取", "全部", "完整")
        )
        item_intent = self._has_item_keyword(query)
        if self._is_enumeration_query(query):
            return keys
        if len(keys) >= 2 and (full_intent or item_intent):
            return keys
        # A configured alias can also identify a single document family.
        return keys if alias_file_fragments(query) and len(keys) == 1 else []

    @staticmethod
    def _has_item_keyword(query: str) -> bool:
        """判断 query 是否命中单试验项关键词（BCI/ESD/RE/浪涌/谐波/瞬态…）。"""
        if not query:
            return False
        q = query.lower()
        for kw in _ITEM_KEYWORDS:
            if re.fullmatch(r"[A-Za-z]+", kw):
                if re.search(rf"\b{kw}\b", query, re.IGNORECASE):
                    return True
            elif kw.lower() in q:
                return True
        return False

    @staticmethod
    def _item_aliases(query: str) -> set[str]:
        """返回 query 命中试验项的中英文别名集合（用于文件名过滤）。"""
        aliases: set[str] = set()
        q = query.lower()
        table = {
            "BCI": ("BCI", "大电流注入"),
            "大电流注入": ("BCI", "大电流注入"),
            "ESD": ("ESD", "静电放电"),
            "静电放电": ("ESD", "静电放电"),
            "RE": ("RE", "辐射发射", "辐射骚扰"),
            "CE": ("CE", "传导发射", "传导骚扰"),
            "浪涌": ("浪涌",),
            "谐波": ("谐波",),
            "瞬态": ("瞬态",),
            "磁场": ("磁场",),
        }
        for kw, al in table.items():
            if kw.lower() in q or (
                re.fullmatch(r"[a-z]+", kw) and re.search(rf"\b{kw}\b", query, re.IGNORECASE)
            ):
                aliases.update(al)
        return aliases

    def _list_doc_files(self) -> list[str]:
        try:
            files = [d.get("file_name", "") for d in self.chroma.list_documents()]
        except Exception:  # noqa: BLE001
            files = []
        return [f for f in files if f]

    def _resolve_compare_groups(
        self, family_keys: list[str], query: str, files: list[str],
    ) -> list[dict]:
        """把标准族解析为对比组；族内多版本时按 query 引用的版本拆分。"""
        from core.standard_registry import STANDARD_FAMILIES as _FAMILIES

        item_aliases = self._item_aliases(query)
        versions = set()
        for m in re.finditer(r"\b(20\d{2})\b|\b(\d\.\d)\b", query):
            versions.add(m.group(0))
        groups: list[dict] = []
        for key in family_keys:
            members = family_member_files(key, files)
            if not members:
                continue
            # 单项题：按试验项别名过滤族内文件（如 BCI 只留第 7 部分）
            if item_aliases and len(members) > 1:
                filtered = [
                    fn for fn in members
                    if any(a.lower() in fn.lower() for a in item_aliases)
                ]
                if filtered:
                    members = filtered
            label = (_FAMILIES.get(key) or {}).get("label", key)
            if len(members) >= 2 and versions:
                sub = self._split_by_version(members, versions)
                if len(sub) >= 2:
                    for v, fns in sub.items():
                        groups.append({"label": f"{label}（{v}）", "files": fns})
                    continue
            groups.append({"label": label, "files": members})
        return groups

    @staticmethod
    def _split_by_version(members: list[str], versions: set[str]) -> dict[str, list[str]]:
        """按 query 引用的版本把文件分桶；无版本归属的文件丢弃。"""
        groups = {v: [] for v in versions}
        for fn in members:
            for v in versions:
                if re.search(re.escape(v), fn):
                    groups[v].append(fn)
                    break
        return {k: v for k, v in groups.items() if v}

    def _retrieve_compare_families(
        self, family_keys: list[str], query: str, top_k: Optional[int],
    ) -> list[dict]:
        """按标准族分解检索：枚举族内全部文档（如 JAC 2026 的 21 个分拆部分），
        逐文件检索后合并，保证全标准对比/枚举的上下文覆盖。"""
        k = top_k or config.RETRIEVAL_TOP_K
        try:
            files = [d.get("file_name", "") for d in self.chroma.list_documents()]
        except Exception:  # noqa: BLE001
            files = []
        files = [f for f in files if f]
        members: dict[str, list[str]] = {}
        for key in family_keys:
            members[key] = family_member_files(key, files)
        total_files = sum(len(v) for v in members.values()) or 1
        per = max(3, min(6, (k * 3) // total_files + 1))
        per_family: dict[str, list[dict]] = {}
        seen: set[str] = set()
        for key in family_keys:
            bucket: list[dict] = []
            for fn in members.get(key, []):
                expr = f'file_name == "{escape_filter_value(fn)}"'
                try:
                    hits = self.retriever.retrieve(query, top_k=per, filter_expr=expr)
                except Exception as e:  # noqa: BLE001
                    log.warning("family retrieve %s: %s", fn, e)
                    hits = []
                for h in hits or []:
                    cid = h.get("id")
                    if cid and cid not in seen:
                        seen.add(cid)
                        bucket.append(h)
            per_family[key] = bucket
        # 按族平衡：每族按检索分取 top，再轮询合并，保证双方标准都能进入最终重排
        # （避免 93 条候选经"按原始分粗排取 2x"后单一标准被整体挤出）
        per_family_keep = max(6, (k * 2) // max(len(family_keys), 1))
        for key in family_keys:
            bucket = per_family.get(key, [])
            bucket.sort(key=lambda c: float(c.get("score") or 0.0), reverse=True)
            per_family[key] = bucket[:per_family_keep]
        merged: list[dict] = []
        max_len = max((len(v) for v in per_family.values()), default=0)
        for i in range(max_len):
            for key in family_keys:
                bucket = per_family.get(key, [])
                if i < len(bucket):
                    merged.append(bucket[i])
        log.info(
            "family compare: %s -> %d chunk (per=%d, files=%d)",
            ",".join(family_keys), len(merged), per, total_files,
        )
        return merged

    def _two_stage_compare(
        self, query: str, family_keys: list[str], top_k: Optional[int], language: str,
    ) -> dict:
        """两阶段对比：阶段A 逐组提取试验项 profile，阶段B 对比生成。

        Returns:
            {"answer", "contexts", "groups", "used_rerank"}
        """
        files = self._list_doc_files()
        groups = self._resolve_compare_groups(family_keys, query, files)
        if not groups:
            raise ValueError("无可用对比组")
        item_intent = self._has_item_keyword(query)
        enumeration = self._is_enumeration_query(query)

        # 阶段A：各组检索 → 组内重排 → 并行提取 profile
        group_chunks: dict[str, list[dict]] = {}
        for g in groups:
            group_chunks[g["label"]] = self._retrieve_group_chunks(
                g, query, item_intent,
            )
        # 全局编号：阶段A 提取与阶段B 对比共用同一编号体系（保证 [n] 可核验）
        final_contexts: list[dict] = []
        seen: set[str] = set()
        next_idx = 1
        for g in groups:
            for c in group_chunks.get(g["label"], []):
                cid = c.get("id")
                if cid and cid not in seen:
                    seen.add(cid)
                    c["index"] = next_idx
                    next_idx += 1
                    final_contexts.append(c)
        final_contexts = final_contexts[: config.COMPARE_FINAL_TOP_K]
        profiles: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=min(2, len(groups))) as pool:
            futures = {
                submit_context(pool,
                    self._extract_profile,
                    g["label"], group_chunks[g["label"]], query, g.get("files", []),
                ): g["label"]
                for g in groups
            }
            for fut in futures:
                label = futures[fut]
                try:
                    profiles[label] = fut.result()
                except Exception as e:  # noqa: BLE001
                    log.warning("profile 提取失败 %s: %s", label, e)
                    profiles[label] = ""

        # 阶段B：上下文并集（与阶段A 一致的 chunk，保证 [n] 可核验）
        answer = self._generate_compare(
            query, profiles, final_contexts, language, enumeration,
        )
        return {
            "answer": answer,
            "contexts": final_contexts,
            "groups": groups,
            "used_rerank": True,
        }

    def _retrieve_group_chunks(
        self, group: dict, query: str, item_intent: bool,
    ) -> list[dict]:
        """单组检索：逐文件检索 + 组内重排，返回 PROFILE_CHUNKS_PER_GROUP 条。"""
        per = (
            config.ITEM_TARGETED_TOP_K
            if item_intent else config.PROFILE_PER_FILE_TOP_K
        )
        # 单文件标准（如吉利/沃尔沃/小米单文档）：一次拉足，避免 4-6 块覆盖不全
        if len(group.get("files", [])) == 1:
            per = max(per, config.PROFILE_CHUNKS_PER_GROUP + 4)
        retrieve_query = query
        if item_intent:
            aliases = sorted(self._item_aliases(query))
            if aliases:
                retrieve_query = (
                    f"{query} {' '.join(aliases)} 频率范围 限值 等级 判定"
                )
        chunks: list[dict] = []
        seen: set[str] = set()
        for fn in group.get("files", [])[:30]:
            expr = f'file_name == "{escape_filter_value(fn)}"'
            try:
                hits = self.retriever.retrieve(retrieve_query, top_k=per, filter_expr=expr)
            except Exception as e:  # noqa: BLE001
                log.warning("group retrieve %s: %s", fn, e)
                hits = []
            for h in hits or []:
                cid = h.get("id")
                if cid and cid not in seen:
                    seen.add(cid)
                    chunks.append(h)
        # 提取场景优先文本/表格块：图片块（VL 描述）信息密度低，候选足够时剔除
        non_image = [c for c in chunks if c.get("modality") != "image"]
        if (
            len(non_image) >= max(3, config.PROFILE_CHUNKS_PER_GROUP // 2)
            and len(non_image) < len(chunks)
        ):
            chunks = non_image
        # 原始候选上限（防超大组把重排池撑爆）
        if len(chunks) > config.PROFILE_MAX_RAW_CHUNKS_PER_GROUP:
            chunks = sorted(
                chunks, key=lambda c: float(c.get("score") or 0.0), reverse=True,
            )[: config.PROFILE_MAX_RAW_CHUNKS_PER_GROUP]
        if len(chunks) > config.PROFILE_CHUNKS_PER_GROUP:
            try:
                chunks = self.reranker.rerank(
                    retrieve_query, chunks, top_k=config.PROFILE_CHUNKS_PER_GROUP,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("group rerank 失败 %s: %s", group.get("label"), e)
                chunks = chunks[: config.PROFILE_CHUNKS_PER_GROUP]
        return chunks

    def _extract_profile(
        self, label: str, chunks: list[dict], query: str, group_files: Optional[list[str]] = None,
    ) -> str:
        """阶段A：基于组内 chunk 提取试验项目清单（带缓存）。"""
        cache_key = f"{cache_namespace()}{label}|{query}"
        now = time.time()
        with self._profile_cache_lock:
            entry = self._profile_cache.get(cache_key)
            if entry is not None and now - entry[1] < config.PROFILE_CACHE_TTL:
                return entry[0]
        if not chunks:
            return ""
        # 文件名元数据注入：分拆标准（如 JAC 2015/2026）的文件名即各部分名称，
        # 直接提供给提取器可保证"全部部分"不被漏列（metadata-aware extraction）
        files_seen: list[str] = []
        for fn in group_files or []:
            if fn and fn not in files_seen:
                files_seen.append(fn)
        for c in chunks:
            fn = c.get("file_name") or ""
            if fn and fn not in files_seen:
                files_seen.append(fn)
        file_hint = ""
        if files_seen:
            file_hint = (
                "\n提示：该标准由以下文件/文档构成（对分拆保存的标准，文件名即各部分名称，"
                "清单必须覆盖全部文件对应的部分）：\n"
                + "\n".join(f"- {fn}" for fn in files_seen)
                + "\n"
            )
        system_prompt = _EXTRACT_PROFILE_PROMPT.format(label=label) + file_hint
        profile = self.generator.generate(
            query=f"请提取《{label}》的试验项目完整清单（含参数/限值/判定，注明来源文件名）",
            contexts=chunks,
            language="zh",
            system_prompt=system_prompt,
        )
        with self._profile_cache_lock:
            if len(self._profile_cache) >= config.PROFILE_CACHE_MAX:
                stale = sorted(self._profile_cache, key=lambda k: self._profile_cache[k][1])
                for old in stale[: config.PROFILE_CACHE_MAX // 2]:
                    self._profile_cache.pop(old, None)
            self._profile_cache[cache_key] = (profile, now)
        return profile

    def _generate_compare(
        self, query: str, profiles: dict[str, str],
        final_contexts: list[dict], language: str, enumeration: bool,
    ) -> str:
        """阶段B：对比/枚举生成。"""
        labels = list(profiles.keys())
        if enumeration and len(profiles) == 1:
            prompt = _ENUM_FORMAT_PROMPT.format(
                label=labels[0], profile=profiles[labels[0]],
            )
        else:
            label_a = labels[0] if labels else "标准A"
            label_b = labels[1] if len(labels) > 1 else "标准B"
            prompt = _COMPARE_STAGE_B_PROMPT.format(
                label_a=label_a,
                profile_a=profiles.get(label_a, "") or "（该标准资料提取为空）",
                label_b=label_b,
                profile_b=profiles.get(label_b, "") or "（该标准资料提取为空）",
            )
        return self.generator.generate(
            query, contexts=final_contexts, language=language, system_prompt=prompt,
        )

    def _supplement_matched_files(
        self, candidates: list[dict], matched_files: list[str],
        query: str, top_k: Optional[int],
    ) -> list[dict]:
        """别名补充召回：候选里缺失命中文档时，按文件名过滤检索补齐其 chunk。"""
        if not matched_files or not candidates:
            return candidates
        present = {c.get("file_name") for c in candidates}
        missing = [fn for fn in matched_files if fn not in present]
        if not missing:
            return candidates
        k = top_k or config.RETRIEVAL_TOP_K
        per = max(2, k // max(len(missing), 1))
        seen = {c.get("id") for c in candidates if c.get("id")}
        added: list[dict] = []
        for fn in missing[:8]:
            expr = f'file_name == "{escape_filter_value(fn)}"'
            try:
                hits = self.retriever.retrieve(query, top_k=per, filter_expr=expr)
            except Exception as e:  # noqa: BLE001
                log.warning("supplement %s: %s", fn, e)
                hits = []
            for h in hits or []:
                cid = h.get("id")
                if cid and cid not in seen:
                    seen.add(cid)
                    added.append(h)
        if added:
            log.info(
                "alias supplement: +%d chunk (missing=%d)",
                len(added), len(missing),
            )
            return candidates + added
        return candidates

    def _detect_compare_docs(self, query: str) -> Optional[list[str]]:
        """???????? query ??? >=2 ?????????????????

        ?????query ?????????????? 2 ?????? token
        ???/?????????????????????? 2 ??? None?
        """
        if not query or not self.chroma:
            return None
        if not any(h in query for h in _COMPARE_HINTS):
            return None
        try:
            files = [d.get("file_name", "") for d in self.chroma.list_documents()]
        except Exception:  # noqa: BLE001
            return None
        files = [f for f in files if f]
        if not files:
            return None
        q_up = query.upper()
        tokens: set[str] = set()
        for m in re.finditer(r"[A-Z0-9]{4,}", q_up):
            tok = m.group(0)
            # ???"??+??"?????/?? token??? "TL"/"800"/"ESD" ??????
            if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
                tokens.add(tok)
        matched: list[str] = []
        for fn in files:
            fu = fn.upper().replace(" ", "").replace("-", "")
            hit = any(tok and tok in fu for tok in tokens)
            if not hit:
                hit = any(
                    alias_key in query and alias_val.upper().replace(" ", "") in fu
                    for alias_key, alias_val in _COMPARE_ALIASES.items()
                )
            if hit and fn not in matched:
                matched.append(fn)
        return matched if len(matched) >= 2 else None

    def _retrieve_compare(
        self, files: list[str], query: str, top_k: Optional[int],
    ) -> list[dict]:
        """?????????????????????????"""
        k = top_k or config.RETRIEVAL_TOP_K
        per = max(2, k // len(files) + 1)
        merged: list[dict] = []
        seen: set[str] = set()
        for fn in files:
            expr = f'file_name == "{escape_filter_value(fn)}"'
            try:
                hits = self.retriever.retrieve(query, top_k=per * 2, filter_expr=expr)
            except Exception as e:  # noqa: BLE001
                log.warning("???????? %s: %s", fn, e)
                hits = []
            for h in hits or []:
                cid = h.get("id")
                if cid and cid not in seen:
                    seen.add(cid)
                    merged.append(h)
        log.info("??????: %d ??? -> %d chunk", len(files), len(merged))
        return merged[: k * 2]

    @staticmethod
    def _cache_key(query: str, top_k: int, use_rerank: bool, language: str, filter_expr: str) -> str:
        """生成查询级别的缓存键（基于完整参数哈希）。"""
        raw = f"{cache_namespace()}{query}|{top_k}|{use_rerank}|{language}|{filter_expr}"
        return hashlib.md5(raw.encode()).hexdigest()

    # ---------- Requery 相关方法 ----------

    def _should_rewrite(self, query: str) -> bool:
        """判断是否触发 requery（query 改写 + 多路召回融合）。

        触发条件（全部满足）：
            1. REQUERY_ENABLED 开关开启
            2. query 非空
            3. 满足以下任一：
               a) query 长度 < REQUERY_MIN_QUERY_LENGTH（短查询）
               b) query 命中口语化词正则（"怎么/如何/为什么/why/how/which/what/多少" 等）
        """
        if not config.REQUERY_ENABLED:
            return False
        if not query or not query.strip():
            return False
        # 短查询 → 优先触发
        if len(query.strip()) < config.REQUERY_MIN_QUERY_LENGTH:
            return True
        # 口语化词 → 绕过长度阈值
        try:
            if get_colloquial_pattern().search(query):
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def _get_rewrites(self, query: str, language: str) -> list[str]:
        """获取改写后的 query 列表（带缓存）。

        行为：
            - 缓存命中且未过期 → 直接返回
            - 缓存未命中 → 调用 QueryRewriter.rewrite
            - 任何异常 → 降级为 [原 query]（不影响后续流程）
            - 超过缓存上限 → 淘汰最旧的一半（FIFO）

        Returns:
            改写 query 列表，**始终**包含原 query 作为第一个元素。
        """
        cache_key = self._rewrite_cache_key(query, language)
        now = time.time()

        # 读缓存
        with self._rewrite_cache_lock:
            entry = self._rewrite_cache.get(cache_key)
            if entry is not None:
                rewrites, ts = entry
                if now - ts < config.REQUERY_CACHE_TTL:
                    return list(rewrites)

        # 缓存未命中 → 调 LLM 改写
        try:
            rewrites = self.query_rewriter.rewrite(
                query, language=language, n_variants=config.REQUERY_N_VARIANTS,
            )
        except Exception as e:  # noqa: BLE001
            # 装饰器耗尽重试后仍失败 → 降级为 [原 query]
            log.warning("query 改写最终失败，降级为原 query: %s", e)
            rewrites = [query]

        # 防御：保证原 query 在最前
        if not rewrites or rewrites[0] != query:
            rewrites = [query] + [r for r in rewrites if r != query]

        # 写缓存（FIFO 淘汰）
        with self._rewrite_cache_lock:
            if len(self._rewrite_cache) >= config.REQUERY_CACHE_MAX:
                evict_count = config.REQUERY_CACHE_MAX // 2
                old_keys = list(self._rewrite_cache.keys())[:evict_count]
                for k in old_keys:
                    del self._rewrite_cache[k]
            self._rewrite_cache[cache_key] = (list(rewrites), now)
        return rewrites

    @staticmethod
    def _rewrite_cache_key(query: str, language: str) -> str:
        """生成改写缓存键（基于 query + language 哈希）。"""
        raw = f"{cache_namespace()}{query}|{language}"
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def _extract_product_models(query: str) -> list[str]:
        """从 query 中提取产品型号（如 DEMO27-A、DEMO24-B）。"""
        if not query:
            return []
        return [m.group(1) for m in _PRODUCT_MODEL_RE.finditer(query)]

    @staticmethod
    def _build_doc_filter(query: str) -> Optional[str]:
        """从 query 中提取产品型号，构建 ChromaDB filter_expr。

        仅当 query 中恰好出现 1 个产品型号时才构建 filter，
        避免误过滤多文档对比场景。
        """
        models = QueryPipeline._extract_product_models(query)
        if len(models) == 1:
            return f'file_name $contains "{escape_filter_value(models[0])}"'
        return None

    @staticmethod
    def _collect_image_chunks(contexts: list[dict]) -> list[dict]:
        """从 contexts 中收集所有图片 chunk 信息（去重）。

        图片 chunk 可能出现在两个位置：
          1. contexts 顶层 — 直接检索到的 image 类型 chunk
          2. linked_chunks — _link_related 图文关联回查到的图片 chunk

        返回的每个条目包含 image_path（磁盘路径），供前端/报告读取图片文件。
        """
        seen: set[str] = set()
        result: list[dict] = []

        def _append(chunk: dict) -> None:
            cid = str(chunk.get("id", ""))
            if not cid or cid in seen:
                return
            seen.add(cid)
            # 从 ChromaDB metadata 中获取磁盘图片路径
            meta = chunk.get("metadata", {}) or {}
            image_path = meta.get("image_path", "")
            result.append({
                "id": cid,
                "page": chunk.get("page", 0),
                "file_name": chunk.get("file_name", ""),
                "description": (chunk.get("content") or "")[:200],
                "image_path": image_path,
            })

        for ctx in contexts:
            if ctx.get("modality") == "image":
                _append(ctx)
            for linked in ctx.get("linked_chunks") or []:
                if linked.get("modality") == "image":
                    _append(linked)

        return result

    # ---------- 公开工具 ----------

    @staticmethod
    def _source_summary(hit: dict) -> dict[str, Any]:
        """把 hit 字典压成 API 友好的摘要（避免回传大字段）。"""
        content = hit.get("content", "") or ""
        # 摘要：截前 240 字，保留可读性
        snippet = content[:240] + ("..." if len(content) > 240 else "")
        return {
            "id": hit.get("id", ""),
            "file_name": hit.get("file_name", ""),
            "page": hit.get("page", 0),
            "modality": hit.get("modality", "text"),
            "content": snippet,
            "score": hit.get("score", 0.0),
        }


__all__ = ["QueryPipeline"]
