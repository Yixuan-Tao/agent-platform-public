# Test values are synthetic fixtures for software behavior.
"""失败标签分类测试。"""

from __future__ import annotations

from evaluation.labels import classify_failure


def test_labels_ok_none():
    assert classify_failure({}, {"status": "OK"}) is None
    assert classify_failure({}, {"status": "N/A"}) is None


def test_labels_hallucination():
    item = {"expected_answer_status": "insufficient_evidence"}
    res = {"status": "MISS", "answer_status": "ok", "sources": 2, "keyword_hit": False}
    assert classify_failure(item, res) == "HALLUCINATION"


def test_labels_refusal():
    item = {"expected_answer_status": "gateway_unavailable"}
    res = {"status": "MISS", "answer_status": "ok", "sources": 0, "keyword_hit": False}
    assert classify_failure(item, res) == "REFUSAL_ERROR"


def test_labels_retrieval_miss():
    res = {"status": "MISS", "answer_status": "ok", "sources": 0, "keyword_hit": False}
    assert classify_failure({}, res) == "RETRIEVAL_MISS"


def test_labels_wrong_evidence():
    res = {"status": "MISS", "answer_status": "ok", "sources": 3, "keyword_hit": False}
    assert classify_failure({}, res) == "WRONG_EVIDENCE"


def test_labels_citation_missing():
    res = {
        "status": "MISS",
        "answer_status": "ok",
        "sources": 1,
        "keyword_hit": True,
        "evidence_ok": False,
    }
    assert classify_failure({}, res) == "CITATION_MISSING"


def test_labels_format_fail():
    assert classify_failure({}, {"status": "ERR"}) == "FORMAT_FAIL"
