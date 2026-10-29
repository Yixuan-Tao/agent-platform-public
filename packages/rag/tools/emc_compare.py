"""
EMC 文档对比工具 — 从 QueryPipeline 剥离的独立对比模块。

职责：
    1. 对比意图检测（关键词正则匹配）
    2. 全量文档拉取 + 两阶段对比（参数提取 → 对比生成）
    3. 章节路由对比（按 query topic 精准定位 section 做对比）

使用方式：
    tool = EmcCompareTool(rag_service=..., generator=..., config=...)
    result = tool.compare(query, doc_a="id1", doc_b="id2")
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from config import config
from core.generator import Generator
from rag.service import RagService
from tools.base import (
    collect_image_chunks,
    escape_filter_value,
    extract_doc_ids,
    format_chunks_for_llm,
    safe_parse_json,
    source_summary,
)
from utils.logger import log
from utils.runtime_budget import submit_context

# ============================================================
# 对比意图检测常量
# ============================================================

_COMPARISON_PATTERNS = [
    # 中文
    r"对比", r"比较", r"区别", r"差异", r"不同", r"哪个更", r"优缺点",
    r"有什么不同", r"有什么不一样", r"区别在哪", r"差别",
    # 英文
    r"\bcompare\b", r"\bdifference\b", r"\bvs\b", r"\bversus\b",
    r"\bwhich one\b", r"\bpros and cons\b", r"\bdistinguish\b",
    # 德文
    r"\bvergleichen\b", r"\bvergleich\b", r"\bunterschied\b",
    r"\bwelche\b.*\bbesser\b", r"\bvor und nachteile\b",
]
_COMPARISON_RE = re.compile("|".join(_COMPARISON_PATTERNS), re.IGNORECASE)

# 产品型号提取正则（匹配类似 DEMO27-A、DEMO24-B 等型号）
_PRODUCT_MODEL_RE = re.compile(
    r"\b([A-Z]{2,}\d[\w-]*\d|\d{3}\s*\d{3}\s*\d{3})\b", re.IGNORECASE
)

# ============================================================
# 参数提取 & 对比生成 Prompt（原位于 core/generator.py，随 Task 4 删除后迁移至此）
# ============================================================

_EXTRACT_PARAMETERS_PROMPT = """你是专业的电子元器件文档分析专家。请从以下技术文档内容中提取该元器件的关键参数。

要求：
1. 以 JSON 格式输出，key 为参数名称（使用中文），value 为参数值（保持原文单位）
2. 只提取与元器件电气特性、物理特性、环境特性相关的参数
3. 如果你不确定某个参数的值，不要编造，不要包含在输出中
4. 输出必须是一个合法的 JSON 对象，不要包含 markdown 包裹或额外说明
5. 参数名称应简洁规范，例如：反向击穿电压、钳位电压、峰值脉冲功率、封装类型、工作温度范围等

【文档内容】
{context}

【注意事项】
- 忽略与文档无关的通用介绍性文字
- 优先提取表格中的结构化数据
- 相同的参数取最完整或最新的值"""

_COMPARE_FROM_PARAMS_PROMPT = """你是专业的硬件对比分析师。请基于两份 EMC 元器件的参数表，进行系统性的对比分析。

要求：
1. 先概述两份文档对应的元器件型号和基本定位
2. 按参数类别（如电气参数、物理参数、环境参数）逐一对比
3. 对差异较大的参数重点说明
4. 最后给出总结性的对比结论和建议
5. 使用清晰的结构（## 标题）组织内容
6. 在正文中标注参数来源，例如 (文档A) / (文档B)

【文档A：{doc_a}】
参数表：
{params_a}

【文档B：{doc_b}】
参数表：
{params_b}

【用户问题】
{query}

请输出结构化的对比分析报告。"""


def _extract_product_models(query: str) -> list[str]:
    """从 query 中提取产品型号（如 DEMO27-A、DEMO24-B）。"""
    if not query:
        return []
    return [m.group(1) for m in _PRODUCT_MODEL_RE.finditer(query)]


def _match_models_to_docs(
    models: list[str], candidates: list[dict], doc_ids: list[str],
) -> list[str]:
    """检查产品型号是否出现在检索结果的文档名中，返回匹配的 doc_id 列表。"""
    doc_names: dict[str, str] = {}
    for h in candidates:
        did = h.get("doc_id", "")
        if did and did not in doc_names:
            doc_names[did] = (h.get("file_name") or "").lower()
    matched: list[str] = []
    for model in models:
        model_lower = model.lower()
        for did in doc_ids:
            if did in matched:
                continue
            fname = doc_names.get(did, "")
            if model_lower in fname or model_lower in did.lower():
                matched.append(did)
                break
    return matched


def get_section_topic_for_query(query: str) -> list[str]:
    """从 query 文本中匹配章节主题，返回命中的 topic 列表。

    规则：
        - 遍历 ``config.SECTION_TOPIC_KEYWORDS`` 的每个 (topic, keywords)
        - 用 case-insensitive 包含匹配检查 query 是否包含任一 keyword
        - 多个 keyword 命中同一 topic 只算一次
        - 多 topic 命中按定义的顺序返回

    Args:
        query: 用户查询文本。

    Returns:
        命中的 topic 列表（如 ``["immunity.bci", "test_method"]``）。
        无命中或 query 为空时返回空列表。
    """
    if not query or not query.strip():
        return []
    topics_map = getattr(config, "SECTION_TOPIC_KEYWORDS", None) or {}
    if not topics_map:
        return []
    q_lower = query.lower()
    hits: list[str] = []
    for topic, keywords in topics_map.items():
        for kw in keywords:
            if not kw:
                continue
            if kw.lower() in q_lower:
                if topic not in hits:
                    hits.append(topic)
                break
    return hits


class EmcCompareTool:
    """EMC 文档对比工具。

    封装了对比意图检测、两阶段对比（参数提取 + 对比生成）、章节路由对比等能力。
    依赖通过构造函数注入，不硬编码全局单例。
    """

    def __init__(
        self,
        rag_service: RagService,
        generator: Generator,
        config: Any,
    ) -> None:
        self._rag = rag_service
        self.generator = generator
        self.config = config

    # ============================================================
    # 公开方法
    # ============================================================

    def compare(
        self,
        query: str,
        doc_a: str,
        doc_b: str,
        top_k: Optional[int] = None,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        """从两份文档中检索并生成对比答案。

        两阶段流程：
            1) 分别从 doc_a / doc_b 全量拉取所有 chunk
            2) 并行调用 LLM 对每份文档做参数提取（结构化 JSON 参数表）
            3) 基于两份参数表做对比生成

        当 ``ENABLE_COMPARE_SECTION_ROUTING`` 为 True 时，先尝试按 query 命中的
        section topic 路由对比；未命中或路由失败时降级到全量。

        Args:
            query: 对比问题，如 "对比两颗 TVS 的钳位电压"
            doc_a: 文档A 的 doc_id
            doc_b: 文档B 的 doc_id
            top_k: 兼容旧接口保留
            language: 回答语种

        Returns:
            ``{"answer": str, "sources": [...], "params_a": dict, "params_b": dict,
               "doc_a": str, "doc_b": str, "language": str,
               "compare_route": "section" | "full" | "fallback"}``
        """
        lang = language or "zh"

        # ---- 章节路由优先 ----
        if getattr(self.config, "ENABLE_COMPARE_SECTION_ROUTING", True):
            try:
                section_result = self._try_section_routing(query, doc_a, doc_b, lang)
                if section_result is not None:
                    return section_result
            except Exception as e:  # noqa: BLE001
                log.warning("section routing 异常，降级到全量对比: %s", e)

        # ---- 阶段1: 并行全量拉取两份文档 ----
        hits_a, hits_b = self._fetch_two_docs_parallel(doc_a, doc_b)
        log.info("compare() 全量拉取完成: a=%d chunks, b=%d chunks", len(hits_a), len(hits_b))

        if not hits_a and not hits_b:
            return {
                "answer": "未在指定文档中找到内容。",
                "sources": [], "image_chunks": [],
                "params_a": {}, "params_b": {},
                "doc_a": doc_a, "doc_b": doc_b,
                "language": lang, "used_rerank": False,
                "compare_route": "fallback",
            }

        result = self._run_two_stage_compare(query, hits_a, hits_b, doc_a, doc_b, lang)
        result["compare_route"] = "full"
        return result

    def try_comparison(
        self,
        query: str,
        top_k: Optional[int],
        language: str,
        filter_expr: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """检测对比意图并执行对比流程（供 QueryPipeline.query() 外部调用）。

        当 query 含对比关键词 且 检索结果跨多个文档时，自动触发对比模式。
        返回对比结果 dict 或 None（未触发对比）。
        """
        if filter_expr or not self._is_comparison_query(query):
            return None

        probe_k = max(top_k or getattr(self.config, "COMPARE_RETRIEVAL_TOP_K", 20), 20)
        try:
            candidates = self._rag.retrieve(query, top_k=probe_k)
        except Exception as e:  # noqa: BLE001
            log.warning("对比检测检索失败: %s", e)
            return None

        doc_ids = extract_doc_ids(candidates)
        if len(doc_ids) < 2:
            return None

        # 型号匹配：优先使用 query 中提取到的型号定位文档
        product_models = _extract_product_models(query)
        if product_models:
            target_docs = _match_models_to_docs(product_models, candidates, doc_ids)
            if len(target_docs) >= 2:
                log.info(
                    "检测到对比意图，型号匹配成功 (query=%.40s..., models=%s, docs=%s)",
                    query, product_models, target_docs,
                )
                return self._compare_internal(query, candidates, target_docs, top_k, language)

        log.info("检测到对比意图，触发多文档对比 (query=%.40s..., docs=%s)", query, doc_ids)
        return self._compare_internal(query, candidates, doc_ids, top_k, language)

    # ============================================================
    # 参数提取 & 对比生成（原 Generator 方法，随 Task 4 删除后迁移至此）
    # ============================================================

    def extract_parameters(
        self,
        hits: list[dict],
        doc_name: str,
        language: str = "zh",
    ) -> dict[str, Any]:
        """从文档 chunks 中提取结构化参数表（JSON），内部调用 self.generator.generate()。

        Args:
            hits: 单份文档的所有 chunk 列表。
            doc_name: 文档显示名称（用于日志）。
            language: 语种（暂未使用，保留签名兼容）。

        Returns:
            解析后的参数 dict，失败时返回空 dict。
        """
        if not hits:
            return {}
        context_text = format_chunks_for_llm(hits)
        # 截断上下文避免超长
        max_chars = getattr(self.config, "EXTRACT_MAX_CONTEXT_TOKENS", 12000) * 4
        if len(context_text) > max_chars:
            context_text = context_text[:max_chars] + "\n\n...（截断）"
        system_prompt = _EXTRACT_PARAMETERS_PROMPT.format(context=context_text)
        try:
            answer = self.generator.generate(
                query=f"请提取文档 {doc_name} 的所有关键参数",
                contexts=[],
                language=language,
                system_prompt=system_prompt,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("参数提取失败: doc=%s err=%s", doc_name, e)
            return {}
        result = safe_parse_json(answer)
        if not result:
            log.warning("参数提取 JSON 解析为空: doc=%s", doc_name)
        return result

    def generate_compare_from_params(
        self,
        query: str,
        params_a: dict,
        params_b: dict,
        doc_a_name: str,
        doc_b_name: str,
        language: str = "zh",
    ) -> str:
        """基于两份参数表生成对比答案，内部调用 self.generator.generate()。

        Args:
            query: 用户对比问题。
            params_a: 文档A 的参数表 dict。
            params_b: 文档B 的参数表 dict。
            doc_a_name: 文档A 的显示名称。
            doc_b_name: 文档B 的显示名称。
            language: 回答语种。

        Returns:
            对比分析报告文本。
        """
        system_prompt = _COMPARE_FROM_PARAMS_PROMPT.format(
            doc_a=doc_a_name or "文档A",
            doc_b=doc_b_name or "文档B",
            params_a=json.dumps(params_a, ensure_ascii=False, indent=2),
            params_b=json.dumps(params_b, ensure_ascii=False, indent=2),
            query=query,
        )
        try:
            return self.generator.generate(
                query=query,
                contexts=[],
                language=language,
                system_prompt=system_prompt,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("对比生成失败: %s", e)
            return f"对比生成失败: {e}"

    # ============================================================
    # 内部方法
    # ============================================================

    def _is_comparison_query(self, query: str) -> bool:
        """检查 query 文本是否包含对比关键词。"""
        return bool(query and _COMPARISON_RE.search(query))

    def _fetch_two_docs_parallel(
        self, doc_a: str, doc_b: str,
    ) -> tuple[list[dict], list[dict]]:
        """并行全量拉取两份文档的所有 chunk。"""
        safe_a = escape_filter_value(doc_a)
        safe_b = escape_filter_value(doc_b)
        results: dict[str, list[dict]] = {"a": [], "b": []}
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cmp-fetch") as pool:
            fut_a = submit_context(pool, self._rag.retrieve_all_by_doc, safe_a)
            fut_b = submit_context(pool, self._rag.retrieve_all_by_doc, safe_b)
            try:
                results["a"] = fut_a.result() or []
            except Exception as e:
                log.warning("对比全量拉取 A 失败: %s", e)
            try:
                results["b"] = fut_b.result() or []
            except Exception as e:
                log.warning("对比全量拉取 B 失败: %s", e)
        return results["a"], results["b"]

    def _extract_params_parallel(
        self,
        hits_a: list[dict], hits_b: list[dict],
        doc_a_name: str, doc_b_name: str,
        language: str,
    ) -> tuple[dict, dict]:
        """并行调用 LLM 提取两份文档的参数表。"""
        params_a: dict = {}
        params_b: dict = {}
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cmp-ext") as pool:
            fut_a = submit_context(pool,
                self.extract_parameters, hits_a, doc_a_name, language,
            )
            fut_b = submit_context(pool,
                self.extract_parameters, hits_b, doc_b_name, language,
            )
            try:
                params_a = fut_a.result() or {}
            except Exception as e:  # noqa: BLE001
                log.warning("文档A参数提取失败: %s", e)
            try:
                params_b = fut_b.result() or {}
            except Exception as e:  # noqa: BLE001
                log.warning("文档B参数提取失败: %s", e)
        return params_a, params_b

    def _run_two_stage_compare(
        self,
        query: str,
        hits_a: list[dict],
        hits_b: list[dict],
        doc_a: str,
        doc_b: str,
        language: str,
    ) -> dict[str, Any]:
        """两阶段对比流程（参数提取 + 对比生成）。"""
        doc_a_name = self._resolve_doc_name(hits_a, doc_a)
        doc_b_name = self._resolve_doc_name(hits_b, doc_b)

        params_a, params_b = self._extract_params_parallel(
            hits_a, hits_b, doc_a_name, doc_b_name, language,
        )

        try:
            answer = self.generate_compare_from_params(
                query, params_a, params_b, doc_a_name, doc_b_name, language=language,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("对比生成失败: %s", e)
            answer = f"对比生成失败: {e}"

        all_hits = hits_a + hits_b
        all_hits.sort(key=lambda c: (c.get("file_name", ""), c.get("page", 0)))
        sources = [source_summary(c) for c in all_hits[:20]]

        return {
            "answer": answer,
            "sources": sources,
            "image_chunks": collect_image_chunks(hits_a + hits_b),
            "params_a": params_a,
            "params_b": params_b,
            "doc_a": doc_a_name,
            "doc_b": doc_b_name,
            "language": language,
            "used_rerank": False,
        }

    def _compare_internal(
        self,
        query: str,
        candidates: list[dict],
        doc_ids: list[str],
        top_k: Optional[int],
        language: str,
    ) -> dict[str, Any]:
        """从 top 2 文档全量拉取，做两阶段对比生成。"""
        doc_a, doc_b = doc_ids[0], doc_ids[1]

        # 章节路由优先
        if getattr(self.config, "ENABLE_COMPARE_SECTION_ROUTING", True):
            try:
                section_result = self._try_section_routing(query, doc_a, doc_b, language)
                if section_result is not None:
                    return section_result
            except Exception as e:  # noqa: BLE001
                log.warning("_compare_internal section routing 异常，降级到全量对比: %s", e)

        hits_a, hits_b = self._fetch_two_docs_parallel(doc_a, doc_b)
        log.info(
            "_compare_internal 全量拉取完成: a=%d chunks, b=%d chunks (doc_a=%s, doc_b=%s)",
            len(hits_a), len(hits_b), doc_a, doc_b,
        )

        if not hits_a and not hits_b:
            return {
                "answer": "对比检索失败：两份文档均无结果。",
                "sources": [], "image_chunks": [],
                "language": language, "used_rerank": False,
                "compare_route": "fallback",
            }

        result = self._run_two_stage_compare(query, hits_a, hits_b, doc_a, doc_b, language)
        result["compare_route"] = "full"
        return result

    # ============================================================
    # 章节路由对比
    # ============================================================

    def _try_section_routing(
        self,
        query: str,
        doc_a: str,
        doc_b: str,
        language: str,
    ) -> Optional[dict[str, Any]]:
        """章节路由入口：按 query 命中的 topic 路由对比，未命中返回 None。"""
        if not getattr(self.config, "ENABLE_COMPARE_SECTION_ROUTING", True):
            return None
        topics = get_section_topic_for_query(query)
        if not topics:
            log.info("section routing miss (no topic hit), fallback to full-doc compare")
            return None
        topics = topics[:3]

        # 并行拉取两份文档的 topic chunks
        results: dict[str, dict[str, list[dict]]] = {
            t: {"a": [], "b": []} for t in topics
        }
        max_workers = min(6, len(topics) * 2)
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="cmp-sec-fetch",
        ) as pool:
            futures: dict = {}
            for topic in topics:
                futures[submit_context(pool,
                    self._rag.retrieve_by_section, doc_a, "", topic,
                )] = ("a", topic)
                futures[submit_context(pool,
                    self._rag.retrieve_by_section, doc_b, "", topic,
                )] = ("b", topic)
            for fut, (which, topic) in futures.items():
                try:
                    hits = fut.result() or []
                except Exception as e:  # noqa: BLE001
                    log.warning("section retrieve 失败: topic=%s which=%s err=%s", topic, which, e)
                    hits = []
                results[topic][which] = hits

        # 过滤：任一文档为空则跳过该 topic
        topic_results: list[tuple[str, list[dict], list[dict]]] = []
        max_k = getattr(self.config, "MAX_CHUNKS_PER_SECTION_FOR_EXTRACT", 80)
        for topic in topics:
            a_hits = results[topic]["a"]
            b_hits = results[topic]["b"]
            if a_hits and b_hits:
                if len(a_hits) > max_k:
                    a_hits = a_hits[:max_k]
                if len(b_hits) > max_k:
                    b_hits = b_hits[:max_k]
                topic_results.append((topic, a_hits, b_hits))
        if not topic_results:
            log.info("section routing miss (no chunks), fallback to full-doc compare")
            return None
        log.info("section routing hit, topics=%s", [t for t, _, _ in topic_results])
        return self._compare_multi_section(
            query, topic_results, doc_a, doc_b, language,
        )

    def _run_section_compare(
        self,
        query: str,
        topic: str,
        hits_a: list[dict],
        hits_b: list[dict],
        doc_a_name: str,
        doc_b_name: str,
        language: str,
    ) -> dict[str, Any]:
        """对单一 section 跑"参数提取 + 对比"流程。"""
        topic_label = getattr(self.config, "SECTION_TOPIC_LABELS", {}).get(topic, topic)
        params_a, params_b = self._extract_params_parallel(
            hits_a, hits_b, doc_a_name, doc_b_name, language,
        )
        routed_query = f"【对比维度：{topic_label}】{query}"
        try:
            answer = self.generate_compare_from_params(
                routed_query, params_a, params_b, doc_a_name, doc_b_name, language=language,
            )
        except Exception as e:  # noqa: BLE001
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg or "限流" in err_msg:
                log.warning("section 对比触发限流: topic=%s", topic)
            else:
                log.exception("section 对比失败: topic=%s", topic)
            answer = f"({topic_label} 对比生成失败: {e})"
        all_hits = hits_a + hits_b
        all_hits.sort(key=lambda c: (c.get("file_name", ""), c.get("page", 0)))
        sources = [source_summary(c) for c in all_hits[:20]]
        image_chunks = collect_image_chunks(hits_a + hits_b)
        return {
            "topic": topic,
            "topic_label": topic_label,
            "answer": answer,
            "params_a": params_a,
            "params_b": params_b,
            "sources": sources,
            "image_chunks": image_chunks,
        }

    def _compare_multi_section(
        self,
        query: str,
        topic_results: list[tuple[str, list[dict], list[dict]]],
        doc_a: str,
        doc_b: str,
        language: str,
    ) -> dict[str, Any]:
        """对多个 topic 并行执行 _run_section_compare，合并结果。"""
        first_a, first_b = topic_results[0][1], topic_results[0][2]
        doc_a_name = self._resolve_doc_name(first_a, doc_a)
        doc_b_name = self._resolve_doc_name(first_b, doc_b)
        section_results: list[dict[str, Any]] = []
        max_workers = min(3, len(topic_results))
        log.info(
            "section 并行对比: topics=%s, max_workers=%d",
            [t for t, _, _ in topic_results], max_workers,
        )
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="cmp-sec-ext",
        ) as pool:
            futs = []
            for topic, hits_a, hits_b in topic_results:
                fut = submit_context(pool,
                    self._run_section_compare,
                    query, topic, hits_a, hits_b, doc_a_name, doc_b_name, language,
                )
                futs.append(fut)
            for f in futs:
                try:
                    section_results.append(f.result() or {})
                except Exception as e:  # noqa: BLE001
                    log.warning("section 并行对比失败: %s", e)
                    section_results.append({})
        return self._merge_section_results(
            section_results, doc_a_name, doc_b_name, doc_a, doc_b, language,
        )

    def _merge_section_results(
        self,
        section_results: list[dict[str, Any]],
        doc_a_name: str,
        doc_b_name: str,
        doc_a: str,
        doc_b: str,
        language: str,
    ) -> dict[str, Any]:
        """合并多 section 对比结果。

        合并策略：
            - answer 按 topic 顺序用 "## {label} 对比" 拼接，section 之间用 "---" 分隔
            - sources 按 chunk_id 去重，最多 30 个
            - image_chunks 按 id 去重，最多 20 个
            - params_a / params_b 嵌套合并：key = topic，value = 子 params
            - compare_route 标记为 "section"
        """
        parts: list[str] = []
        all_sources: list[dict] = []
        seen_source_ids: set[str] = set()
        all_image_chunks: list[dict] = []
        seen_image_ids: set[str] = set()
        params_a_merged: dict[str, Any] = {}
        params_b_merged: dict[str, Any] = {}
        for r in section_results:
            if not r:
                continue
            topic = r.get("topic", "")
            label = r.get("topic_label", topic)
            ans = r.get("answer", "")
            if ans:
                parts.append(f"## {label} 对比\n\n{ans}")
            for s in r.get("sources", []):
                sid = s.get("chunk_id") or s.get("id") or ""
                if sid and sid not in seen_source_ids:
                    seen_source_ids.add(sid)
                    all_sources.append(s)
            for img in r.get("image_chunks", []):
                iid = img.get("id") or img.get("chunk_id") or ""
                if iid and iid not in seen_image_ids:
                    seen_image_ids.add(iid)
                    all_image_chunks.append(img)
            if r.get("params_a"):
                params_a_merged[topic] = r["params_a"]
            if r.get("params_b"):
                params_b_merged[topic] = r["params_b"]
        final_answer = "\n\n---\n\n".join(parts) if parts else "未生成对比内容。"
        return {
            "answer": final_answer,
            "sources": all_sources[:30],
            "image_chunks": all_image_chunks[:20],
            "params_a": params_a_merged,
            "params_b": params_b_merged,
            "doc_a": doc_a_name or doc_a,
            "doc_b": doc_b_name or doc_b,
            "language": language,
            "used_rerank": False,
            "compare_route": "section",
            "section_topics": [r.get("topic", "") for r in section_results if r],
        }

    @staticmethod
    def _resolve_doc_name(hits: list[dict], fallback: str) -> str:
        """从检索结果中解析文档名，找不到时用 fallback。"""
        for h in hits:
            name = h.get("file_name", "")
            if name:
                return name
        return fallback
