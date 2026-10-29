"""失败标签体系：把未通过用例映射到统一标签，便于按环节聚合定位问题。"""

from __future__ import annotations

from typing import Any

FAILURE_LABELS = (
    "RETRIEVAL_MISS",
    "WRONG_EVIDENCE",
    "CITATION_MISSING",
    "HALLUCINATION",
    "REFUSAL_ERROR",
    "FORMAT_FAIL",
)


def classify_failure(item: dict[str, Any], res: dict[str, Any]) -> str | None:
    """按规则映射失败标签；通过/不适用用例返回 None。

    规则：
    - 应诚实说"未提及"却作答 → HALLUCINATION
    - 应网关降级未降级 → REFUSAL_ERROR
    - 执行错误/其余 → FORMAT_FAIL
    - 证据数不足 → CITATION_MISSING
    - 无来源且关键词未中 → RETRIEVAL_MISS
    - 有来源但关键词未中 → WRONG_EVIDENCE
    """
    status = res.get("status")
    if status in ("OK", "N/A", None):
        return None
    expected = item.get("expected_answer_status")
    actual = res.get("answer_status")
    if expected == "insufficient_evidence" and actual == "ok":
        return "HALLUCINATION"
    if expected == "gateway_unavailable" and actual != "gateway_unavailable":
        return "REFUSAL_ERROR"
    if status == "ERR":
        return "FORMAT_FAIL"
    if res.get("evidence_ok") is False:
        return "CITATION_MISSING"
    if not res.get("sources"):
        return "RETRIEVAL_MISS"
    if res.get("keyword_hit") is False:
        return "WRONG_EVIDENCE"
    return "FORMAT_FAIL"
