"""TVS/MOSFET 人工证据的确定性双向替代规则。"""
from __future__ import annotations

from typing import Any


_UNIT_SCALE = {"V": 1.0, "MV": 1e-3, "KV": 1e3, "A": 1.0, "MA": 1e-3, "UA": 1e-6, "MΩ": 1e-3, "KΩ": 1e3, "Ω": 1.0, "MΩ": 1e-3}


def _by_subject(items: list[dict], subject: str) -> dict[str, dict]:
    return {item.get("field_name", ""): item for item in items if item.get("subject_id") == subject}


def _value(item: dict, unit: str) -> float | None:
    value = item.get("value_normalized")
    if not isinstance(value, (int, float)):
        return None
    from_unit = str(item.get("unit", "")).upper()
    target = str(unit or from_unit).upper()
    if from_unit == target:
        return float(value)
    if from_unit in _UNIT_SCALE and target in _UNIT_SCALE:
        return float(value) * _UNIT_SCALE[from_unit] / _UNIT_SCALE[target]
    return None


def _citation(item: dict) -> dict:
    return {"file_name": item.get("source_file", ""), "page": item.get("page", 0), "section": item.get("section", "")}


def _direction(base: dict[str, dict], candidate: dict[str, dict], rules: list[dict], name: str) -> dict:
    failures, pending, rows, citations = [], [], [], []
    for rule in rules:
        parameter = rule["parameter"]
        source, replacement = base.get(parameter), candidate.get(parameter)
        if not source or not replacement:
            (failures if rule.get("critical") else pending).append(parameter)
            rows.append({"parameter": parameter, "status": "missing", "reason": "缺少字段级证据"})
            continue
        citations.extend([_citation(source), _citation(replacement)])
        if rule.get("require_condition") and (not source.get("test_condition") or not replacement.get("test_condition") or source.get("test_condition") != replacement.get("test_condition")):
            pending.append(parameter)
            rows.append({"parameter": parameter, "status": "conditional", "reason": "测试条件缺失或不可比"})
            continue
        kind = rule.get("rule", "equal")
        if kind == "equal":
            passed = str(source.get("value_normalized")) == str(replacement.get("value_normalized"))
        else:
            base_value = _value(source, source.get("unit", ""))
            candidate_value = _value(replacement, source.get("unit", ""))
            if base_value is None or candidate_value is None:
                pending.append(parameter)
                rows.append({"parameter": parameter, "status": "conditional", "reason": "数值或单位不可比"})
                continue
            candidate_value *= float(rule.get("derating_factor", 1.0))
            passed = candidate_value >= base_value if kind == "candidate_gte" else candidate_value <= base_value
        if passed:
            rows.append({"parameter": parameter, "status": "pass", "reason": "规则满足"})
        else:
            failures.append(parameter)
            rows.append({"parameter": parameter, "status": "fail", "reason": "规则不满足"})
    conclusion = "不可替换" if failures else "有条件替换" if pending else "可替换"
    return {"direction": name, "conclusion": conclusion, "failed_params": failures, "pending_params": pending, "rule_results": rows, "citations": citations}


def compare_component_evidence(items_a: list[dict], items_b: list[dict], subject_a: str, subject_b: str, rules: list[dict]) -> dict[str, Any]:
    """按照人工发布规则输出 A→B 与 B→A；不使用 LLM 推断。"""
    a, b = _by_subject(items_a, subject_a), _by_subject(items_b, subject_b)
    return {"A_to_B": _direction(a, b, rules, "A→B"), "B_to_A": _direction(b, a, rules, "B→A")}
