# Test values are synthetic fixtures for software behavior.
"""裁判样本量校验脚本测试（对应 CI 空跑防护）。"""

import json
from pathlib import Path

from evaluation.check_judge_sample import load_judged_count, main


def _write(path: Path, judged: int) -> Path:
    path.write_text(
        json.dumps({"judge_summary": {"judged": judged}, "results": []}),
        encoding="utf-8",
    )
    return path


def test_load_judged_count(tmp_path):
    assert load_judged_count(_write(tmp_path / "r.json", 96)) == 96


def test_missing_summary_counts_as_zero(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"results": []}), encoding="utf-8")
    assert load_judged_count(path) == 0


def test_main_fails_when_sample_too_small(tmp_path, monkeypatch):
    path = _write(tmp_path / "small.json", 3)
    monkeypatch.setattr(
        "sys.argv", ["check_judge_sample.py", str(path), "--min", "90"]
    )
    assert main() == 1


def test_main_passes_when_sample_sufficient(tmp_path, monkeypatch):
    path = _write(tmp_path / "ok.json", 96)
    monkeypatch.setattr(
        "sys.argv", ["check_judge_sample.py", str(path), "--min", "90"]
    )
    assert main() == 0


def test_main_reports_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["check_judge_sample.py", str(tmp_path / "nope.json")]
    )
    assert main() == 2
