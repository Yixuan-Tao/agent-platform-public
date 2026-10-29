"""裁判校准：跑黄金集，计算 Cohen's kappa（pass/fail）对齐门槛 >= 0.7。

用法（packages/rag 下，需 DEEPSEEK_API_KEY）：
    python evaluation/calibrate_judge.py [--golden evaluation/golden_set_v1.json]
                                         [--eval-file evaluation/run_result_v6_judged.json]
输出 metrics/judge_calibration.json；kappa < 0.7 时以非零退出码告警（可接入门禁）。
建议每月校准一次，捕获裁判漂移。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

KAPPA_THRESHOLD = 0.7
_EVAL_DIR = Path(__file__).resolve().parent
_METRICS_DIR = _EVAL_DIR.parent / "metrics"


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa（二元标签）。"""
    n = len(a)
    if n == 0:
        return 1.0
    p0 = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a) / n
    pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return 1.0 if p0 == 1.0 else 0.0
    return (p0 - pe) / (1 - pe)


def _judge_item_from_result(r: dict) -> dict:
    return {
        "question": r.get("question", ""),
        "answer": r.get("answer", ""),
        "contexts": r.get("contexts") or [],
        "ground_truth": r.get("ground_truth", ""),
        "expected_keywords": r.get("expected_keywords") or [],
        "judge_criteria": r.get("judge_criteria", ""),
        "doc_group": r.get("doc_group", ""),
        "expected_answer_status": r.get("expected_answer_status"),
        "expected_evidence_min": r.get("expected_evidence_min"),
        "keyword_variants": r.get("keyword_variants") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(_EVAL_DIR / "golden_set_v1.json"))
    parser.add_argument(
        "--eval-file", default=str(_EVAL_DIR / "run_result_v6_judged.json")
    )
    parser.add_argument(
        "--output", default=str(_METRICS_DIR / "judge_calibration.json")
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    except Exception:  # noqa: BLE001
        pass

    from evaluation.judge import DeepSeekJudge

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    saved = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    by_question = {r.get("question"): r for r in saved.get("results") or []}

    judge = DeepSeekJudge()
    if not judge.available:
        print("缺少 DEEPSEEK_API_KEY，无法校准")
        return 1

    matched: list[dict] = []
    missing: list[str] = []
    for item in golden:
        r = by_question.get(item.get("question"))
        if r is None:
            missing.append(item.get("question", ""))
            continue
        matched.append({**item, "_result": r})
    if not matched:
        print("黄金集与评测结果无交集，无法校准")
        return 1
    if missing:
        print(f"警告：{len(missing)} 条黄金集样本未在评测结果中找到，跳过")

    items = [_judge_item_from_result(m["_result"]) for m in matched]
    results = judge.judge_many(items)

    per_item = []
    expected = []
    actual = []
    for m, jr in zip(matched, results):
        exp = 1 if m["expected_pass"] else 0
        act = 1 if jr.get("pass") else 0
        expected.append(exp)
        actual.append(act)
        lo, hi = m["expected_score_range"]
        score = jr.get("score")
        in_range = bool(score is not None and lo <= score <= hi)
        per_item.append(
            {
                "id": m["id"],
                "question": m["question"],
                "expected_pass": m["expected_pass"],
                "judge_pass": jr.get("pass"),
                "score": score,
                "score_in_range": in_range,
                "feedback": jr.get("feedback", ""),
            }
        )

    kappa = cohen_kappa(expected, actual)
    agreement = sum(1 for x, y in zip(expected, actual) if x == y) / len(expected)
    passed = kappa >= KAPPA_THRESHOLD
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "judge_model": judge.model,
        "judge_model_version": judge.model_version,
        "golden_size": len(matched),
        "kappa": round(kappa, 3),
        "agreement": round(agreement, 3),
        "kappa_threshold": KAPPA_THRESHOLD,
        "passed": passed,
        "per_item": per_item,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n校准结果：kappa={kappa:.3f}（门槛 {KAPPA_THRESHOLD}）→ {'通过' if passed else '未通过'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
