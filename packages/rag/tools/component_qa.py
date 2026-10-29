#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
元器件问答工具 — 检索命中 datasheet 时按参数表逐参数组织回答。

背景：
    RAG 库中同时存在 EMC 标准文档与元器件规格书（datasheet）。当用户询问
    元器件（如"ESD24VD3BHE3 的钳位电压"）时，普通 RAG 只会零散地给出
    检索到的片段，缺少对"汽车ECU电子元器件替代评估关键参数表"的覆盖。

本工具在 QueryPipeline 普通 RAG 分支生成答案前介入：
    1. 判定检索结果是否主要来自 datasheet 文档（非 EMC 标准）
    2. 识别元器件种类（复用 param_template 的类型检测）
    3. 从参数表（CSV/XLSX）加载该类型的评估参数
    4. 构建"按参数逐项回答"的 system_prompt，注入 Generator
"""
from __future__ import annotations

import re
from typing import Any, Optional

from config import config as _app_config
from tools.param_template import (
    _detect_component_type,
    parse_xlsx_template,
)
from utils.logger import log

# 产品型号提取正则（匹配类似 PESD1IVN27-A、ESD24VD3BHE3、BUK6D120-60PX 等型号）
_PRODUCT_MODEL_RE = re.compile(
    r"\b([A-Z]{2,}\d[\w-]*\d|\d{3}\s*\d{3}\s*\d{3})\b", re.IGNORECASE
)

# datasheet 文件名特征词（命中任一即视为元器件文档）
_DATASHEET_KEYWORDS = [
    "规格书", "datasheet", "spec", "手册", "器件",
    "tvs", "esd", "mosfet", "mlcc", "zener", "schottky",
    "bjt", "npn", "pnp", "diode", "二极管",
    "sot-", "sod-", "dfn", "son", "qfn",
    "buk", "sqa", "电阻", "电容", "电感", "晶振", "连接器",
]

# EMC 标准文件名特征词（命中任一优先判为标准，避免误判）
_STANDARD_KEYWORDS = [
    "标准", "规范", "emc", "电磁兼容", "demoalpha", "demobeta",
    "测试要求", "测试规范", "test specification", "requirements",
]

# 元器件问答触发阈值：datasheet 文档的 chunk 占比达到该比例才启用
_DATASHEET_CHUNK_RATIO = 0.5


class ComponentQATool:
    """元器件问答工具：命中 datasheet 时按参数表逐参数组织回答。"""

    def __init__(
        self,
        rag_service: Any = None,
        config: Any = None,
        template_path: str = "",
    ) -> None:
        """
        Args:
            rag_service: RagService 实例（当前未使用，保留兼容接口）。
            config: 配置对象（默认使用全局 config）。
            template_path: 参数表文件路径；为空时使用 param_template 默认（example 下 CSV）。
        """
        self._rag = rag_service
        self._cfg = config or _app_config
        self._template_path = template_path

    # ============================================================
    # 公开方法
    # ============================================================

    def is_datasheet_query(self, candidates: list[dict]) -> bool:
        """判定检索结果是否主要来自 datasheet（元器件规格书）文档。

        规则：按 doc_id 聚合 chunk 数，逐文档判定类型（文件名优先，
        无法判定时用 chunk 内容特征词兜底）；datasheet chunk 占比 >= 阈值
        且 > 0 时返回 True。

        Args:
            candidates: 检索结果（retriever.retrieve 归一化后的 hit 列表）。

        Returns:
            True 表示主要命中元器件文档，应启用元器件问答模式。
        """
        doc_stats: dict[str, dict[str, Any]] = {}
        for c in candidates:
            did = c.get("doc_id") or ""
            fname = c.get("file_name") or ""
            if not did:
                continue
            stat = doc_stats.setdefault(did, {"file_name": fname, "chunks": []})
            stat["chunks"].append(c)
            if not stat["file_name"] and fname:
                stat["file_name"] = fname

        if not doc_stats:
            return False

        total = sum(len(s["chunks"]) for s in doc_stats.values())
        datasheet_chunks = 0
        for did, stat in doc_stats.items():
            if self._doc_is_datasheet(stat["file_name"], stat["chunks"]):
                datasheet_chunks += len(stat["chunks"])

        ratio = datasheet_chunks / total if total > 0 else 0.0
        log.debug(
            "datasheet 命中判定: %d/%d (%.0f%%), 阈值 %.0f%%",
            datasheet_chunks, total, ratio * 100, _DATASHEET_CHUNK_RATIO * 100,
        )
        return ratio >= _DATASHEET_CHUNK_RATIO

    def detect_component_type(self, query: str, candidates: list[dict]) -> str:
        """识别用户所询问元器件的种类（如 TVS / MOSFET / MLCC）。

        综合 query（型号+类型词）、命中的 datasheet 文档名进行检测。
        """
        # 从 query 提取产品型号
        model = ""
        if query:
            matches = _PRODUCT_MODEL_RE.findall(query)
            if matches:
                model = matches[0]

        # 取命中的第一个 datasheet 文档名参与检测
        doc_name = ""
        for c in candidates:
            fname = c.get("file_name") or ""
            if fname and self._doc_is_datasheet(fname, [c]):
                doc_name = fname
                break

        return _detect_component_type(doc_name, model, query or "")

    def build_qa_prompt(
        self, component_type: str, params_template: dict,
    ) -> str:
        """构建元器件问答的 system_prompt（覆盖默认 prompt）。

        将参数模板按类别格式化为"评估参数清单"，要求 LLM 逐项回答：
        - 参考资料中能找到的参数 → 给出具体数值（保留原文单位）
        - 找不到的参数 → 明确标注"【文档未提供】"
        """
        category_names = {
            "electrical": "电气特性",
            "package": "封装/物理特性",
            "thermal": "热特性",
            "reliability": "可靠性",
            "compliance": "合规认证",
            "other": "其他",
        }
        lines: list[str] = []
        for category, params in params_template.items():
            cat_name = category_names.get(category, str(category))
            lines.append(f"【{cat_name}】")
            if not isinstance(params, dict):
                continue
            for param_name, spec in params.items():
                if not isinstance(spec, dict):
                    spec = {}
                unit = spec.get("unit", "") or ""
                desc = (spec.get("description") or "")[:80]
                unit_str = f"（{unit}）" if unit else ""
                lines.append(f"- {param_name}{unit_str}：{desc}")
        param_list = "\n".join(lines)

        return f"""你是专业的电子元器件文档助手。用户询问的元器件类型为【{component_type}】。

【回答规则 - 必须遵守】
1. 严格基于下方"参考资料"中给出的内容回答问题，禁止编造未出现的事实或参数。
2. 请按照以下评估参数清单【逐项回答】，覆盖清单中的全部参数：
   - 参考资料中能找到的参数 → 给出具体数值，保留原文单位（如 27V、1.0µA、SOD-323）
   - 参考资料中找不到的参数 → 明确标注"【文档未提供】"，不要编造
   - 同一参数出现多个值时，优先取典型值（Typical）或标称值（Nominal）
3. 在正文中使用方括号标注引用来源，例如 [1]、[2]；同一来源可多次复用。
4. 答案末尾必须另起一行输出"参考来源："列表，格式为：[1] 文件名 p.页码, [2] 文件名 p.页码。
5. 用户问题特别关注的参数请优先详细回答。
6. 请用中文回答。

【元器件类型】{component_type}

【评估参数清单（按此逐项回答）】
{param_list}"""

    def try_build_prompt(
        self, query: str, candidates: list[dict],
    ) -> Optional[str]:
        """元器件问答入口：命中 datasheet 时返回增强 system_prompt，否则返回 None。

        - 未命中 datasheet → None（走普通 RAG）
        - 类型检测失败或参数表无对应类型 → None（降级普通 RAG，不阻断回答）
        """
        if not query or not query.strip() or not candidates:
            return None
        if not self.is_datasheet_query(candidates):
            return None

        component_type = self.detect_component_type(query, candidates)
        try:
            params = parse_xlsx_template(
                component_type=component_type,
                query=query,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("元器件参数表加载失败，降级普通 RAG: %s", e)
            return None

        if not params:
            log.info(
                "参数表中无 %s 类型数据，降级普通 RAG", component_type,
            )
            return None

        log.info(
            "触发元器件问答模式: 类型=%s, 参数类别=%s",
            component_type, list(params.keys()),
        )
        return self.build_qa_prompt(component_type, params)

    # ============================================================
    # 内部方法
    # ============================================================

    def _doc_is_datasheet(self, file_name: str, chunks: list[dict]) -> bool:
        """判定单个文档是否为元器件规格书。

        判定优先级：
            1. 文件名命中标准特征词 → 非 datasheet
            2. 文件名命中 datasheet 特征词 → datasheet
            3. 文件名无法判定 → 用 chunk 内容特征词兜底（复用 EMC 检测）
        """
        if not file_name:
            return False
        name = file_name.lower()

        # 标准特征词优先（防止 datasheet 关键词误中标准文档）
        if any(kw in name for kw in _STANDARD_KEYWORDS):
            return False
        if any(kw in name for kw in _DATASHEET_KEYWORDS):
            return True

        # 文件名无法判定 → 内容兜底：非 EMC 文档视为元器件文档
        if chunks:
            try:
                from tools.emc_compare import _detect_is_emc_doc
                return not _detect_is_emc_doc(chunks)
            except Exception:  # noqa: BLE001
                return False
        return False


__all__ = ["ComponentQATool"]
