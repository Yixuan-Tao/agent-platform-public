"""EMC 表格可视化工具

将 RAG 检索到的表格 chunk 自动检测类型、归一化并生成 ECharts 可视化 HTML。
支持单文档可视化和双文档叠加对比。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .emc_type_detector import detect_table_type_from_chunks, get_type_confidence
from .emc_types import CONVERTERS, TYPE_A_CONFIG_COMPUTER
from .emc_viewer import generate_html, build_emc_data, to_inline_bundle, DEFAULT_FREQUENCY_GROUPS
from .emc_chart_policy import frequency_extent
from utils.logger import log


class EmcVisualizeTool:
    """EMC 表格可视化工具

    用法::

        tool = EmcVisualizeTool(rag_service)
        result = tool.visualize("辐射发射限值", doc_id="doc_123")
        # result = {"html": "...", "datasets": [...], "type": "Type_A", ...}
    """

    def __init__(self, rag_service) -> None:
        """初始化可视化工具

        Args:
            rag_service: 需提供 query() 和 retrieve_all_by_doc() 方法的服务对象。
                         通常是 RagService 实例。
        """
        self._rag = rag_service

    def visualize(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """单文档 EMC 表格可视化

        完整流程：
            1. 从 RAG 检索 table chunks
            2. 用 emc_type_detector 检测类型
            3. 用对应的 Type converter 归一化
            4. 用 emc_viewer 生成 ECharts HTML
            5. 返回 HTML bundle + 数据集

        Args:
            query: 检索查询语句，如"辐射发射限值"。
            doc_id: 可选的文档 ID，指定后只从该文档检索。
            top_k: 检索 top_k 个 chunk。

        Returns:
            dict: {
                "type": "Type_A",
                "confidence": 0.85,
                "datasets": [...],
                "html": "...",           # generate_html 格式
                "bundle": {...},         # to_inline_bundle 格式
            }
        """
        # step1: 检索 table chunks
        chunks = self._retrieve_table_chunks(query, doc_id, top_k)
        if not chunks:
            return {"type": "Unknown", "error": "未检索到相关 table chunks"}

        table_chunks = [c for c in chunks if c.get("modality") == "table"]
        if not table_chunks:
            table_chunks = chunks  # fallback: 如果没有明确标记 table，用全部

        # step2: 检测类型
        type_info = get_type_confidence(table_chunks)
        table_type = type_info["type"]
        if table_type == "Unknown":
            return {"type": "Unknown", "error": "无法检测表格类型"}

        # step3: 用 converter 归一化
        dataset = self._convert_chunks_to_dataset(table_chunks, table_type, query)
        if dataset is None:
            return {"type": table_type, "error": "数据转换失败"}

        # step4: 生成 HTML
        emc_data = {"datasets": [dataset]}
        try:
            html_result = generate_html(emc_data)
            bundle_result = to_inline_bundle(emc_data)
        except Exception as e:
            log.warning("可视化 HTML 生成失败: %s", e)
            return {
                "type": table_type,
                "datasets": [dataset],
                "error": f"HTML 生成失败: {e}",
            }

        return {
            "type": table_type,
            "confidence": type_info["confidence"],
            "type_detail": type_info,
            "datasets": [dataset],
            "html": html_result,
            "bundle": bundle_result,
        }

    def visualize_overlay(
        self,
        doc_a: str,
        doc_b: str,
        query: Optional[str] = None,
        table_type: Optional[str] = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """双文档叠加对比可视化

        从两份文档中分别提取 EMC 表格数据，叠加在同一图表中比较。

        Args:
            doc_a: 文档A 的 doc_id 或文件名。
            doc_b: 文档B 的 doc_id 或文件名。
            query: 可选检索查询，为空时全量拉取。
            table_type: 可选指定类型（如 "Type_A"），不指定则自动检测。
            top_k: 每份文档检索 chunk 数。

        Returns:
            dict: {
                "type": "Type_A",
                "doc_a_name": "...",
                "doc_b_name": "...",
                "datasets": [dataset_a, dataset_b],
                "html": {...},
                "bundle": {...},
            }
        """
        # step1: 分别拉取两份文档的数据
        chunks_a = self._retrieve_table_chunks(query, doc_a, top_k)
        chunks_b = self._retrieve_table_chunks(query, doc_b, top_k)

        if not chunks_a or not chunks_b:
            missing = []
            if not chunks_a:
                missing.append("文档A")
            if not chunks_b:
                missing.append("文档B")
            return {"error": f"以下文档无数据: {', '.join(missing)}"}

        # step2: 检测类型（或使用指定类型）
        if table_type is None:
            all_chunks = chunks_a + chunks_b
            type_info = get_type_confidence(all_chunks)
            table_type = type_info["type"]
            if table_type == "Unknown":
                return {"error": "无法自动检测表格类型，请手动指定 table_type"}

        # step3: 分别归一化
        dataset_a = self._convert_chunks_to_dataset(chunks_a, table_type, query or doc_a)
        dataset_b = self._convert_chunks_to_dataset(chunks_b, table_type, query or doc_b)

        if dataset_a is None or dataset_b is None:
            failed = "文档A" if dataset_a is None else "文档B"
            return {"error": f"{failed} 数据转换失败"}

        # step4: 生成对比 HTML
        emc_data = {"datasets": [dataset_a, dataset_b]}
        try:
            html_result = generate_html(emc_data)
            bundle_result = to_inline_bundle(emc_data)
        except Exception as e:
            log.warning("叠加对比 HTML 生成失败: %s", e)
            return {
                "type": table_type,
                "datasets": [dataset_a, dataset_b],
                "error": f"HTML 生成失败: {e}",
            }

        return {
            "type": table_type,
            "doc_a_name": dataset_a.get("name", doc_a),
            "doc_b_name": dataset_b.get("name", doc_b),
            "datasets": [dataset_a, dataset_b],
            "html": html_result,
            "bundle": bundle_result,
        }

    # ---------- 内部方法 ----------

    def _retrieve_table_chunks(
        self,
        query: Optional[str],
        doc_id: Optional[str],
        top_k: int,
    ) -> list[dict]:
        """从 RAG 检索 table chunks

        支持三种模式：
        1. 有 doc_id + rag 有 retrieve_all_by_doc：全量拉取整份文档
        2. 有 doc_id + rag 只有 query（旧 QueryPipeline）：filter_expr 检索
        3. 无 doc_id：调用 retrieve 检索相关 chunk
        """
        try:
            if doc_id:
                # 按文档拉取
                if hasattr(self._rag, "retrieve_all_by_doc"):
                    return self._rag.retrieve_all_by_doc(doc_id) or []
                # fallback: 旧 QueryPipeline 兼容
                if hasattr(self._rag, "query"):
                    result = self._rag.query(
                        query or "",
                        top_k=top_k * 2,
                        filter_expr=f'doc_id == "{doc_id}"',
                    )
                    return result.get("sources", [])
            else:
                # 普通检索（RagService 使用 retrieve，旧 QueryPipeline 走不到这里）
                if query:
                    if hasattr(self._rag, "retrieve"):
                        return self._rag.retrieve(query, top_k=top_k) or []
                    if hasattr(self._rag, "query"):
                        result = self._rag.query(query, top_k=top_k)
                        return result.get("sources", [])
        except Exception as e:
            log.warning("检索 table chunks 失败: %s", e)
        return []

    def _convert_chunks_to_dataset(
        self,
        chunks: list[dict],
        table_type: str,
        source_name: str,
    ) -> Optional[dict]:
        """将检索到的 chunks 转换为对应的 Type dataset 格式

        需要 chunks 中包含 "structured_data" 或 "json_data" 字段（由 chunking 阶段生成的 JSON），
        否则尝试从 content 文本解析。
        """
        # 从 chunks 中收集结构化数据
        json_items = []
        for chunk in chunks:
            # 优先使用结构化数据字段
            structured = chunk.get("structured_data") or chunk.get("json_data")
            if structured and isinstance(structured, (dict, list)):
                if isinstance(structured, list):
                    json_items.extend(structured)
                else:
                    json_items.append(structured)
                continue

            # 尝试从 content 解析 JSON
            content = chunk.get("content", "")
            if content and content.strip().startswith("{"):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        json_items.extend(parsed)
                    else:
                        json_items.append(parsed)
                except (json.JSONDecodeError, TypeError):
                    pass

        if not json_items:
            log.warning("chunks 中未找到结构化数据，无法转换")
            return None

        # 构造标准输入格式
        data = {
            "type": table_type,
            "data": json_items,
            "title": source_name,
        }

        converter = CONVERTERS.get(table_type)
        if converter is None:
            log.warning("未知类型 %s，无可用的 converter", table_type)
            return None

        try:
            return converter(data, source_name)
        except Exception as e:
            log.warning("数据转换失败 (type=%s): %s", table_type, e)
            return None
