# Test values are synthetic fixtures for software behavior.
"""评测快照统计对比测试。"""

from __future__ import annotations

from evaluation.stats import compare_judge_snapshots


def _snapshot(scores: list[float]) -> dict:
    return {
        "results": [
            {
                "judge": {
                    "score": s,
                    "faithfulness": 4.0,
                    "relevance": 4.0,
                    "completeness": 4.0,
                    "citation_quality": 4.0,
                }
            }
            for s in scores
        ]
    }


def test_compare_regression():
    prev = _snapshot([4.5, 4.4, 4.6, 4.5, 4.4])
    cur = _snapshot([3.1, 3.2, 3.0, 3.1, 3.2])
    out = compare_judge_snapshots(prev, cur)
    assert out["overall"] == "regression"
    assert out["dimensions"]["score"]["verdict"] == "regression"
    assert out["dimensions"]["score"]["delta"] < 0


def test_compare_no_change():
    prev = _snapshot([4.0, 4.1, 4.0])
    cur = _snapshot([4.05, 4.0, 4.1])
    out = compare_judge_snapshots(prev, cur)
    assert out["overall"] == "no_change"


def test_compare_improvement():
    prev = _snapshot([3.0, 3.1, 3.0, 3.1])
    cur = _snapshot([4.5, 4.4, 4.6, 4.5])
    out = compare_judge_snapshots(prev, cur)
    assert out["overall"] == "improvement"
