"""EMC 表格类型检测器

根据 chunk 文本内容中的关键词，检测 EMC 表格属于 Type_A/B/C/D 中的哪一种。
"""

import re
from typing import Optional

# 类型关键词映射（按匹配优先级排列）
TYPE_KEYWORDS = {
    "Type_A": [
        "频率/MHz", "限值/dB", "PK", "QP", "AV", "辐射发射", "辐射骚扰",
        "Radiated Emission", "RE", "频率范围", "准峰值", "平均值",
        "dBµV/m", "dBμV/m", "limit_value", "限值线",
    ],
    "Type_B": [
        "BCI", "抗扰度", "注入电流", "大电流注入", "Bulk Current Injection",
        "Immunity", "RS", "辐射抗扰", "CS", "传导抗扰",
        "RI", "radiated immunity", "conducted immunity",
        "测试等级", "test level", "功能状态", "function_status",
    ],
    "Type_C": [
        "脉冲", "瞬态", "Us", "tr/td", "供电线", "Transient Pulse",
        "Pulse 1", "Pulse 2a", "Pulse 2b", "Pulse 3a", "Pulse 3b",
        "瞬态脉冲", "pulse_", "test_requirement",
    ],
    "Type_D": [
        "ESD", "静电放电", "接触放电", "空气放电", "静电",
        "Electrostatic Discharge", "接触/空气", "放电电压",
        "discharge_type", "test_level_kV", "人体模型",
    ],
}


def detect_table_type(chunk_text: str) -> str:
    """检测单个 chunk 的 EMC 表格类型

    对每种类型计算关键词匹配得分（区分大小写不敏感），
    返回得分最高的类型。

    Args:
        chunk_text: chunk 的文本内容。

    Returns:
        "Type_A" / "Type_B" / "Type_C" / "Type_D" / "Unknown"
    """
    if not chunk_text or not chunk_text.strip():
        return "Unknown"

    text_lower = chunk_text.lower()
    scores: dict[str, int] = {}

    for type_name, keywords in TYPE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            # 精确子串匹配（不区分大小写）
            if kw.lower() in text_lower:
                score += 1
            # 正则宽松匹配：处理单词边界情况
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
                score += 2  # 单词边界匹配权重更高
        if score > 0:
            scores[type_name] = score

    if not scores:
        return "Unknown"

    # 返回得分最高的类型
    best_type = max(scores, key=scores.get)
    return best_type


def detect_table_type_from_chunks(chunks: list) -> str:
    """从多个 chunk 中检测类型（加权投票）

    收集每个 chunk 的检测结果，按类型累加得分（detect_table_type 的得分），
    返回总分最高的类型。平局时返回第一个达到最高分的类型。

    Args:
        chunks: chunk 字典列表，每个字典需包含 "content" 字段。

    Returns:
        "Type_A" / "Type_B" / "Type_C" / "Type_D" / "Unknown"
    """
    if not chunks:
        return "Unknown"

    # step1: 收集每类在所有 chunk 中的总得分
    total_scores: dict[str, float] = {}
    for chunk in chunks:
        content = ""
        if isinstance(chunk, dict):
            content = chunk.get("content") or chunk.get("text") or ""
        elif isinstance(chunk, str):
            content = chunk

        if not content:
            continue

        text_lower = content.lower()
        for type_name, keywords in TYPE_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw.lower() in text_lower:
                    score += 1
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
                    score += 2
            if score > 0:
                total_scores[type_name] = total_scores.get(type_name, 0) + score

    if not total_scores:
        return "Unknown"

    best_type = max(total_scores, key=total_scores.get)
    return best_type


def get_type_confidence(chunks: list) -> dict:
    """获取每个类型的置信度详情

    返回各类型的得分和占比，用于调试和展示。

    Args:
        chunks: chunk 字典列表。

    Returns:
        dict: {
            "type": "Type_A",
            "scores": {"Type_A": 15, "Type_B": 3, ...},
            "total": 18,
            "confidence": 0.83,
            "chunk_count": len(chunks),
        }
    """
    if not chunks:
        return {"type": "Unknown", "scores": {}, "total": 0, "confidence": 0.0, "chunk_count": 0}

    total_scores: dict[str, float] = {}
    for chunk in chunks:
        content = ""
        if isinstance(chunk, dict):
            content = chunk.get("content") or chunk.get("text") or ""
        elif isinstance(chunk, str):
            content = chunk
        if not content:
            continue
        text_lower = content.lower()
        for type_name, keywords in TYPE_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw.lower() in text_lower:
                    score += 1
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
                    score += 2
            if score > 0:
                total_scores[type_name] = total_scores.get(type_name, 0) + score

    if not total_scores:
        return {"type": "Unknown", "scores": {}, "total": 0, "confidence": 0.0, "chunk_count": len(chunks)}

    grand_total = sum(total_scores.values())
    best_type = max(total_scores, key=total_scores.get)
    best_score = total_scores[best_type]
    confidence = best_score / grand_total if grand_total > 0 else 0.0

    return {
        "type": best_type,
        "scores": {k: int(v) for k, v in total_scores.items()},
        "total": int(grand_total),
        "confidence": round(confidence, 4),
        "chunk_count": len(chunks),
    }
