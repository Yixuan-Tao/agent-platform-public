# Test values are synthetic fixtures for software behavior.
"""裁判结果回填测试：修复 N/A 与 ERR 题导致的评分错位。"""

from importlib import util
from pathlib import Path

_RUNNER = Path(__file__).resolve().parents[2] / "evaluation" / "run_testset.py"
spec = util.spec_from_file_location("run_testset_backfill", _RUNNER)
module = util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_na_and_err_cases_do_not_consume_judge_results():
    results = [
        {"id": "A", "status": "OK", "question": "a"},
        {"id": "B", "status": "N/A", "question": "b"},
        {"id": "C", "status": "ERR", "question": "c"},
        {"id": "D", "status": "OK", "question": "d"},
    ]
    targets = module.judge_targets_of(results)
    assert [t["id"] for t in targets] == ["A", "D"]

    module.backfill_judge(targets, [{"score": 5.0}, {"score": 3.0}])

    assert results[0]["judge"] == {"score": 5.0}
    assert "judge" not in results[1], "N/A 题不应获得评分"
    assert "judge" not in results[2], "ERR 题不应获得评分"
    assert results[3]["judge"] == {"score": 3.0}


def test_backfill_pads_missing_results_with_empty_dict():
    results = [
        {"id": "A", "status": "OK"},
        {"id": "B", "status": "OK"},
    ]
    targets = module.judge_targets_of(results)
    module.backfill_judge(targets, [{"score": 4.0}])
    assert results[0]["judge"] == {"score": 4.0}
    assert results[1]["judge"] == {}


def test_to_judge_items_keeps_case_identity_fields():
    items = module.to_judge_items(
        [
            {
                "question": "q",
                "answer": "a",
                "expected_evidence_min": 2,
                "expected_answer_status": "ok",
            }
        ]
    )
    assert items[0]["question"] == "q"
    assert items[0]["answer"] == "a"
    assert items[0]["expected_evidence_min"] == 2
    assert items[0]["expected_answer_status"] == "ok"
