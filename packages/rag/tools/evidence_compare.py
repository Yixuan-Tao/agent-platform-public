"""基于人工确认 EvidenceItem 的确定性 EMC 比较。"""
from __future__ import annotations

from typing import Any


EMC_MATRIX_FIELDS = (
    "标准标识", "适用范围", "测试项", "频段", "等级/限值", "测试方法", "测试布置", "判定准则",
)


def _index(items: list[dict], subject_id: str, test_item: str) -> dict[str, list[dict]]:
    indexed: dict[str, list[dict]] = {}
    for item in items:
        if item.get("subject_id") != subject_id:
            continue
        if test_item and item.get("test_item", "").upper() != test_item.upper():
            continue
        indexed.setdefault(item.get("field_name", ""), []).append(item)
    return indexed


def _citation(item: dict) -> dict:
    return {
        "subject_id": item.get("subject_id", ""), "field": item.get("field_name", ""),
        "file_name": item.get("source_file", ""), "page": item.get("page", 0),
        "section": item.get("section", ""),
    }


def compare_emc_evidence(
    items_a: list[dict], items_b: list[dict], subject_a: str, subject_b: str, test_item: str = "",
) -> dict[str, Any]:
    """按固定 EMC 矩阵比较两个对象；缺失证据不推断。"""
    indexed_a, indexed_b = _index(items_a, subject_a, test_item), _index(items_b, subject_b, test_item)
    matrix, citations = [], []
    for field in EMC_MATRIX_FIELDS:
        evidence_a, evidence_b = indexed_a.get(field, []), indexed_b.get(field, [])
        values_a = [entry.get("value_raw", "") for entry in evidence_a]
        values_b = [entry.get("value_raw", "") for entry in evidence_b]
        if not evidence_a or not evidence_b:
            status = "missing"
        elif values_a == values_b:
            status = "same"
        else:
            status = "different"
        matrix.append({"field": field, "a": values_a, "b": values_b, "status": status})
        citations.extend(_citation(item) for item in evidence_a + evidence_b)
    found = sum(len(values) for values in indexed_a.values()) + sum(len(values) for values in indexed_b.values())
    if test_item:
        status = "ok" if indexed_a and indexed_b else "insufficient_evidence"
    else:
        status = "ok" if all(row["status"] != "missing" for row in matrix) else "insufficient_evidence"
    summary = "；".join(f"{row['field']}：{'一致' if row['status'] == 'same' else '存在差异' if row['status'] == 'different' else '证据不足'}" for row in matrix)
    return {
        "answer": f"基于人工确认的 EMC 证据比较结果：{summary}。",
        "answer_status": status,
        "comparison_matrix": matrix,
        "coverage": {"required": len(EMC_MATRIX_FIELDS), "found": found},
        "citations": citations,
    }
