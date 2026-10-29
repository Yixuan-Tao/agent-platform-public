#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
元器件替换判断工具 (Task 5)

功能: 接收用户传入的参数表模板，对两份 datasheet 文档做逐字段严格对比
输出: 可替换 / 有条件替换 / 不可替换 三档结论

核心流程:
    1. 通过 rag_service 拉取两份文档的全量 chunks
    2. 对模板中的每个参数在 chunks 中定位并提取值（关键词匹配 + LLM 提取）
    3. 逐参数对比（含容差计算）
    4. 聚合所有结果，生成可替换性结论
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

from config import config as _app_config


class ComponentCheckTool:
    """元器件替换判断工具"""

    # 被视为关键参数的类别：此类参数不匹配时直接判"不可替换"
    _CRITICAL_CATEGORIES = {"electrical"}

    def __init__(self, rag_service, generator, config=None):
        """
        参数:
            rag_service: RagService 实例，提供 get_document_chunks 等方法
            generator: Generator 实例，用于调用 LLM 提取参数值
            config: 配置对象（默认使用全局 config）
        """
        self._rag = rag_service
        self._generator = generator
        self._cfg = config or _app_config

    def compare_with_template(
        self, doc_a: str, doc_b: str,
        params_template: dict, strict_mode: bool = True,
    ) -> dict:
        """按参数模板对比两份文档

        参数:
            doc_a: 文档 A 的 ID
            doc_b: 文档 B 的 ID
            params_template: 参数表模板
            strict_mode: True 表示超出容差即判不可替换

        返回值:
            {
                "conclusion": "可替换 | 有条件替换 | 不可替换",
                "conclusion_detail": "详细说明",
                "params_comparison": [...],  # 逐参数对比详情
                "failed_params": [...]       # 未定位到的参数
            }
        """
        # 1. 拉取两份文档的全量 chunks
        chunks_a = self._rag.get_document_chunks(doc_a) or []
        chunks_b = self._rag.get_document_chunks(doc_b) or []

        # 2. 从模板中提取参数定义列表
        params_spec = self._flatten_template(params_template)

        # 3. 定位并提取参数值
        values_a, failed_a = self._locate_params_in_chunks(chunks_a, params_spec)
        values_b, failed_b = self._locate_params_in_chunks(chunks_b, params_spec)
        failed_params = list(set(failed_a + failed_b))

        # 4. 逐参数对比
        comparisons = []
        for param_name, spec in params_spec:
            comp = self._compare_params(
                values_a.get(param_name),
                values_b.get(param_name),
                spec,
                strict_mode,
            )
            comp["param"] = param_name
            comp["category"] = spec.get("category", "other")
            comparisons.append(comp)

        # 5. 生成结论
        conclusion, detail = self._generate_conclusion(comparisons, len(params_spec))

        return {
            "conclusion": conclusion,
            "conclusion_detail": detail,
            "params_comparison": comparisons,
            "failed_params": failed_params,
        }

    # ------------------------------------------------------------------
    # 内部：参数模板扁平化
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_template(template: dict) -> list[tuple[str, dict]]:
        """将嵌套的参数模板扁平化为 (name, spec) 列表

        输入:
            {"electrical": {"VBR": {"unit": "V", ...}}, "package": {"type": ...}}

        输出:
            [("VBR", {"category": "electrical", "unit": "V", ...}), ...]
        """
        items: list[tuple[str, dict]] = []
        for category, params in template.items():
            if not isinstance(params, dict):
                continue
            for param_name, spec in params.items():
                if not isinstance(spec, dict):
                    spec = {}
                # 注入 category 字段，便于下游按类别判定严重程度
                spec["category"] = category
                items.append((param_name, spec))
        return items

    # ------------------------------------------------------------------
    # 内部：在 chunks 中定位参数值
    # ------------------------------------------------------------------

    def _locate_params_in_chunks(
        self, chunks: list[dict], params_spec: list[tuple[str, dict]],
    ) -> tuple[dict[str, str], list[str]]:
        """在 chunks 中定位每个参数的值

        对每个参数名做关键词匹配 + LLM 提取

        返回:
            (values_dict, failed_params)
            - values_dict: {param_name: "提取到的值字符串"}
            - failed_params: 未找到的参数名列表
        """
        # 预构建所有 chunk 的文本内容，方便检索
        chunk_texts = [
            {
                "idx": i,
                "text": c.get("searchable_text") or c.get("content", ""),
                "content": c.get("content", ""),
                "page": c.get("page", "?"),
                "section": c.get("section", ""),
            }
            for i, c in enumerate(chunks)
        ]

        values: dict[str, str] = {}
        failed: list[str] = []

        for param_name, spec in params_spec:
            # 对当前参数做关键词匹配，找出最相关的 chunks
            candidates = self._keyword_rank_chunks(chunk_texts, param_name, top_k=5)
            if not candidates:
                # 尝试用 LLM 做更大范围的搜索
                candidates = self._llm_guided_search(
                    chunk_texts, param_name, spec, top_k=3,
                )

            if not candidates:
                failed.append(param_name)
                continue

            # 用 LLM 从候选 chunk 中提取精确值
            value = self._extract_value_with_llm(param_name, spec, candidates)
            if value:
                values[param_name] = value
            else:
                failed.append(param_name)

        return values, failed

    @staticmethod
    def _keyword_rank_chunks(
        chunk_texts: list[dict], keyword: str, top_k: int = 5,
    ) -> list[dict]:
        """基于关键词频率对 chunks 排序（简易 BM25 近似）

        计算 keyword 在 chunk 文本中的出现次数作为相关性得分，
        返回得分最高的 top_k 个 chunk。
        """
        scored: list[tuple[float, int]] = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        for ct in chunk_texts:
            text = ct.get("text", "")
            if not text:
                continue
            # 关键词出现次数作为原始得分
            matches = pattern.findall(text)
            if matches:
                # 简短文本中匹配次数多 => 相关度更高，除以文本长度做归一化
                score = len(matches) / max(len(text), 1) * 1000
                scored.append((score, ct["idx"]))

        # 按得分降序排列
        scored.sort(key=lambda x: -x[0])
        top_idxs = {idx for _, idx in scored[:top_k]}
        return [ct for ct in chunk_texts if ct["idx"] in top_idxs]

    def _llm_guided_search(
        self, chunk_texts: list[dict], param_name: str,
        spec: dict, top_k: int = 3,
    ) -> list[dict]:
        """LLM 引导的宽泛搜索——用于关键词匹配失败时的降级策略

        将所有 chunk 的 section 标题 + 前 200 字拼接后让 LLM 判断
        哪些 chunk 可能包含该参数信息，再取 top_k 个返回。
        """
        # 只取每个 chunk 的 section + 开头片段做摘要
        summaries = []
        for ct in chunk_texts:
            section = ct.get("section", "")
            preview = ct.get("text", "")[:200]
            summaries.append({
                "idx": ct["idx"],
                "hint": f"[{section}] {preview}" if section else preview,
            })

        if not summaries:
            return []

        prompt = (
            f"你是一个硬件参数检索助手。请在以下文档片段中，找出最可能包含参数"
            f"「{param_name}」（{spec.get('unit', '')}）的 {top_k} 个片段。\n"
            f"仅返回片段编号（idx），用逗号分隔，不要输出其他内容。\n\n"
            f"文档片段：\n"
        )
        for s in summaries:
            prompt += f"\nidx={s['idx']}: {s['hint']}"

        try:
            raw = self._generator.generate(
                query=prompt,
                contexts=[],
                language="zh",
            )
            # 解析返回的 idx 列表
            idxs = re.findall(r"\d+", raw)
            matched_idxs = {int(i) for i in idxs if i.isdigit()}
            return [
                ct for ct in chunk_texts if ct["idx"] in matched_idxs
            ][:top_k]
        except Exception:
            return []

    def _extract_value_with_llm(
        self, param_name: str, spec: dict, candidates: list[dict],
    ) -> Optional[str]:
        """用 LLM 从候选 chunks 中提取指定参数的精确值

        返回:
            提取到的值字符串，如 "27V"；提取失败返回 None
        """
        if not candidates:
            return None

        # 构建上下文
        context_parts = []
        for c in candidates:
            page = c.get("page", "?")
            section = c.get("section", "")
            text = c.get("content") or c.get("text", "")
            header = f"--- 第 {page} 页"
            if section:
                header += f" 章节: {section}"
            context_parts.append(f"{header}\n{text[:1500]}")

        context = "\n\n".join(context_parts)
        unit_hint = spec.get("unit", "")
        unit_str = f"（单位：{unit_hint}）" if unit_hint else ""

        prompt = (
            f"你是一个硬件参数提取助手。请从以下 datasheet 内容中，"
            f"找出参数「{param_name}」{unit_str}的值。\n"
            f"规则：\n"
            f"1. 仅返回该参数的精确数值+单位（如 27V, 1.0uA, SOD-323），不要解释\n"
            f"2. 如果出现多个值，取典型值（Typical）或标称值（Nominal）\n"
            f"3. 如果完全找不到该参数，请仅输出「NOT_FOUND」\n\n"
            f"datasheet 内容：\n{context}"
        )

        try:
            raw = self._generator.generate(
                query=prompt,
                contexts=[],
                language="zh",
            )
            result = raw.strip()
            if "NOT_FOUND" in result.upper():
                return None
            # 清洗：去掉可能的引号、多余文本
            result = result.strip("\"' .,;:。，；：")
            return result if result else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 内部：单个参数对比
    # ------------------------------------------------------------------

    def _compare_params(
        self, val_a: Optional[str], val_b: Optional[str],
        spec: dict, strict_mode: bool,
    ) -> dict:
        """对比单个参数值，考虑容差

        返回:
            {
                "doc_a_value": str | None,
                "doc_b_value": str | None,
                "match": bool,
                "within_tolerance": bool,
                "note": str,
            }
        """
        result = {
            "doc_a_value": val_a,
            "doc_b_value": val_b,
            "match": False,
            "within_tolerance": False,
            "note": "",
        }

        # 两个文档都无此参数值
        if not val_a and not val_b:
            result["note"] = "两份文档均未找到该参数"
            return result

        # 只有一份文档有值
        if not val_a:
            result["note"] = f"仅在文档 B 中找到: {val_b}"
            return result
        if not val_b:
            result["note"] = f"仅在文档 A 中找到: {val_a}"
            return result

        # 两份都有值，尝试数值对比
        parsed_a = self._parse_value_with_unit(val_a)
        parsed_b = self._parse_value_with_unit(val_b)

        # 如果两者都能解析为数值，做数值对比 + 容差判断
        if parsed_a is not None and parsed_b is not None:
            num_a, unit_a = parsed_a
            num_b, unit_b = parsed_b

            # 单位不一致
            if unit_a and unit_b and unit_a.upper() != unit_b.upper():
                result["note"] = f"单位不一致: {unit_a} vs {unit_b}"
                try:
                    # 尝试单位换算后对比数值
                    converted = self._try_unit_convert(num_a, unit_a, unit_b)
                    if converted is not None:
                        num_a = converted
                        unit_a = unit_b
                    else:
                        return result
                except Exception:
                    return result

            # 数值完全相等
            if math.isclose(num_a, num_b, rel_tol=1e-9):
                result["match"] = True
                result["within_tolerance"] = True
                result["note"] = f"完全匹配 ({val_a})"
                return result

            # 检查容差
            tolerance = spec.get("tolerance", "")
            if tolerance and self._check_tolerance(num_a, num_b, tolerance):
                result["match"] = True
                result["within_tolerance"] = True
                result["note"] = f"容差内匹配: {val_a} vs {val_b} ({tolerance})"
                return result

            # 数值不匹配
            diff_pct = abs(num_a - num_b) / max(abs(num_a), abs(num_b), 1e-9) * 100
            result["note"] = (
                f"数值不匹配: {val_a} vs {val_b} (偏差 {diff_pct:.1f}%)"
            )
            if strict_mode:
                return result
            # 非严格模式下，偏差小于 20% 算容忍
            if diff_pct <= 20:
                result["match"] = True
                result["within_tolerance"] = False
                result["note"] += "（非严格模式，偏差可接受）"
            return result

        # 字符串值对比（如封装类型）
        if val_a.strip().upper() == val_b.strip().upper():
            result["match"] = True
            result["within_tolerance"] = True
            result["note"] = f"完全匹配 ({val_a})"
        else:
            result["note"] = f"字符串不匹配: {val_a} vs {val_b}"
        return result

    @staticmethod
    def _parse_value_with_unit(value_str: str) -> Optional[tuple[float, str]]:
        """解析带单位的数值字符串

        支持格式: "27V", "1.0uA", "±5%", "1.7x1.25mm" 等

        返回:
            (数值, 单位) 或 None（无法解析时）
        """
        if not value_str or not isinstance(value_str, str):
            return None

        s = value_str.strip()

        # 处理范围值 "26~30V" 或 "26-30V" → 取中间值 28V
        range_match = re.match(
            r"([\d.]+)\s*[~\-–]\s*([\d.]+)\s*([a-zA-Zµμ%°]+)", s,
        )
        if range_match:
            lo, hi, unit = range_match.groups()
            avg = (float(lo) + float(hi)) / 2.0
            return (avg, unit)

        # 处理数值+单位 "27V", "1.0uA"
        value_match = re.match(
            r"^([±±]?\s*[\d.]+(?:[eE][+-]?\d+)?)\s*([a-zA-Zµμ%°]+)", s,
        )
        if value_match:
            num_str, unit = value_match.groups()
            num_str = num_str.replace("±", "").strip()
            return (float(num_str), unit)

        # 纯数值（无单位）
        pure_num = re.match(r"^([±±]?\s*[\d.]+(?:[eE][+-]?\d+)?)$", s)
        if pure_num:
            return (float(pure_num.group(1).replace("±", "").strip()), "")

        return None

    @staticmethod
    def _check_tolerance(val_a: float, val_b: float, tolerance: str) -> bool:
        """检查两个数值是否在容差范围内

        支持的容差格式:
            - "±5%"  → 百分比容差
            - "±1V"  → 绝对值容差
            - "±0.5" → 绝对值容差（无单位）
        """
        tol_str = tolerance.strip()
        # 去掉 ± 前缀
        if tol_str.startswith("±") or tol_str.startswith("+/-"):
            tol_str = tol_str.lstrip("±+/- ")
        else:
            return False

        # 百分比容差
        if tol_str.endswith("%"):
            pct = float(tol_str.rstrip("%"))
            limit = abs(val_a) * pct / 100.0
            return abs(val_a - val_b) <= limit

        # 绝对值容差（可能带单位，如 "1V"）
        tol_value = re.match(r"([\d.]+(?:[eE][+-]?\d+)?)", tol_str)
        if tol_value:
            limit = float(tol_value.group(1))
            return abs(val_a - val_b) <= limit

        return False

    @staticmethod
    def _try_unit_convert(
        value: float, from_unit: str, to_unit: str,
    ) -> Optional[float]:
        """尝试在兼容单位间做换算（如 A ↔ mA ↔ uA）"""
        # 单位归一化
        unit_map = {
            "A": 1.0, "MA": 1e-3, "UA": 1e-6, "NA": 1e-9,
            "PA": 1e-12, "KA": 1e3,
            "V": 1.0, "MV": 1e-3, "UV": 1e-6, "KV": 1e3,
            "W": 1.0, "MW": 1e-3, "UW": 1e-6, "KW": 1e3,
            "OHM": 1.0, "MOHM": 1e-3, "UOHM": 1e-6, "KOHM": 1e3,
            "F": 1.0, "MF": 1e-3, "UF": 1e-6, "NF": 1e-9, "PF": 1e-12,
            "H": 1.0, "MH": 1e-3, "UH": 1e-6, "NH": 1e-9,
            "S": 1.0, "MS": 1e-3, "US": 1e-6, "NS": 1e-9,
        }

        fu = from_unit.upper().replace("µ", "U").replace("Μ", "U")
        tu = to_unit.upper().replace("µ", "U").replace("Μ", "U")

        base_fu = fu.rstrip("S") + "S" if fu.rstrip("S") + "S" in unit_map else fu
        base_tu = tu.rstrip("S") + "S" if tu.rstrip("S") + "S" in unit_map else tu

        if base_fu in unit_map and base_tu in unit_map:
            return value * unit_map[base_fu] / unit_map[base_tu]

        # 处理常见别名：u→µ 微 prefix
        aliases = {"U": "U", "µ": "U", "Μ": "U"}
        return None

    # ------------------------------------------------------------------
    # 内部：结论生成
    # ------------------------------------------------------------------

    def _generate_conclusion(
        self, comparisons: list[dict], total_params: int,
    ) -> tuple[str, str]:
        """聚合所有参数对比结果，生成最终结论

        返回:
            (conclusion, conclusion_detail)
        """
        if not comparisons:
            return "不可替换", "参数模板为空，无法进行比较"

        matched = sum(1 for c in comparisons if c.get("match"))
        within_tol = sum(1 for c in comparisons if c.get("within_tolerance"))
        failed_in_doc = sum(
            1 for c in comparisons
            if not c.get("doc_a_value") or not c.get("doc_b_value")
        )
        critical_failed = [
            c for c in comparisons
            if not c.get("match")
            and c.get("category") in self._CRITICAL_CATEGORIES
        ]

        # 构建详细说明
        parts: list[str] = []
        parts.append(
            f"共 {total_params} 个参数："
            f"匹配 {matched} 个（容差内 {within_tol} 个），"
            f"未匹配 {total_params - matched} 个"
        )

        # 判断结论
        if matched == total_params and within_tol == total_params:
            conclusion = "可替换"
            parts.append("所有参数完全匹配或在容差范围内，可以直接替换。")
        elif critical_failed:
            conclusion = "不可替换"
            critical_names = [c["param"] for c in critical_failed]
            parts.append(
                f"关键参数（电气特性）不匹配: {', '.join(critical_names)}，"
                f"存在性能风险，不建议直接替换。"
            )
        else:
            conclusion = "有条件替换"
            if failed_in_doc > 0:
                parts.append(
                    f"有 {failed_in_doc} 个参数无法在两份文档中都找到，"
                    f"建议人工确认这些参数的值。"
                )
            unmatched = [
                c for c in comparisons if not c.get("match")
            ]
            if unmatched:
                names = [c["param"] for c in unmatched]
                parts.append(
                    f"以下参数不匹配: {', '.join(names)}，"
                    f"需根据具体应用场景评估风险。"
                )
            parts.append("整体功能兼容，但建议在替换前进行充分验证。")

        return conclusion, " ".join(parts)
