# Test values are synthetic fixtures for software behavior.
"""裁判确定性测试：temperature=0、模型版本留痕、中位数/双裁判。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock


def _fake_response(content: dict):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": json.dumps(content)}}]}
    return resp


def test_judge_temperature_zero_and_version(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-x")
    monkeypatch.setenv("DEEPSEEK_MODEL_VERSION", "2026-08-01")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["body"] = json
        return _fake_response(
            {
                "score": 4.0,
                "faithfulness": 4,
                "relevance": 4,
                "completeness": 4,
                "citation_quality": 4,
                "pass": True,
                "feedback": "ok",
            }
        )

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    from evaluation.judge import DeepSeekJudge

    judge = DeepSeekJudge()
    item = {
        "question": "q",
        "answer": "a",
        "contexts": ["c"],
        "ground_truth": "g",
        "doc_group": "device",
    }
    result = judge._judge_stable(item)
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["model"] == "deepseek-x"
    assert result["judge_model"] == "deepseek-x"
    assert result["judge_model_version"] == "2026-08-01"
    assert result["judge_runs"] == 1
    assert result["arbitrated"] is True


def test_judge_median_runs(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("JUDGE_MEDIAN_RUNS", "3")
    from evaluation.judge import DeepSeekJudge

    judge = DeepSeekJudge()
    scores = iter([3.0, 5.0, 4.0])

    def fake_one(item, model=None):
        return {
            "score": next(scores),
            "faithfulness": 4.0,
            "relevance": 4.0,
            "completeness": 4.0,
            "citation_quality": 4.0,
            "pass": False,
            "feedback": "",
        }

    judge._judge_one = fake_one
    result = judge._judge_stable({"question": "q"})
    assert result["score"] == 4.0  # 中位数
    assert result["judge_runs"] == 3


def test_judge_second_model_arbitration(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("JUDGE_SECOND_MODEL", "deepseek-y")
    from evaluation.judge import DeepSeekJudge

    judge = DeepSeekJudge()
    calls = {"n": 0}

    def fake_one(item, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"score": 4.5, "pass": True, "faithfulness": 4, "relevance": 4,
                    "completeness": 4, "citation_quality": 4, "feedback": ""}
        return {"score": 2.0, "pass": False, "faithfulness": 2, "relevance": 2,
                "completeness": 2, "citation_quality": 2, "feedback": ""}

    judge._judge_one = fake_one
    result = judge._judge_stable({"question": "q"})
    assert result["judge_second_model"] == "deepseek-y"
    assert result["arbitrated"] is False


def test_summarize_judge_stddev():
    from evaluation.judge import summarize_judge

    results = [
        {"score": 4.0, "faithfulness": 4, "relevance": 4, "completeness": 4,
         "citation_quality": 4, "pass": True, "judge_model": "m"},
        {"score": 5.0, "faithfulness": 5, "relevance": 5, "completeness": 5,
         "citation_quality": 5, "pass": True, "judge_model": "m"},
    ]
    summary = summarize_judge(results)
    assert summary["std"]["score"] is not None
    assert summary["judge_models"] == ["m"]
