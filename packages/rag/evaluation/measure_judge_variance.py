"""裁判方差基线测量：对冻结配置（同 prompt/模型/样本）连续跑 N 次，输出各维度均值/标准差。

用途：把标准差作为「正常波动区间」写入基线，后续分数变化先与本区间比较，
避免把噪音当回归。用法（packages/rag 下）：
    python evaluation/measure_judge_variance.py [--eval-file ...] [--limit 20] [--repeats 5]
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
_METRICS_DIR = _EVAL_DIR.parent / "metrics"


def _judge_items_from_results(results: list[dict]) -> list[dict]:
    items = []
    for r in results:
        if r.get("status") == "ERR":
            continue
        items.append(
            {
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
        )
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-file",
        default=str(_EVAL_DIR / "run_result_v6_judged.json"),
        help="已保存的评测结果（含 answer/contexts），取前 --limit 条做测量",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output", default=str(_METRICS_DIR / "judge_variance_baseline.json")
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    except Exception:  # noqa: BLE001
        pass

    from evaluation.judge import DeepSeekJudge

    saved = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    items = _judge_items_from_results(saved.get("results") or [])[: args.limit]
    judge = DeepSeekJudge()
    if not judge.available:
        print("缺少 DEEPSEEK_API_KEY，无法测量方差基线")
        return 1
    if not items:
        print("没有可评测的样本")
        return 1

    dims = ("score", "faithfulness", "relevance", "completeness", "citation_quality")
    print(f"[方差基线] {judge.model} 评测 {len(items)} 条 x {args.repeats} 次 ...")
    runs: list[list[dict]] = []
    for run in range(args.repeats):
        print(f"  第 {run + 1}/{args.repeats} 轮 ...")
        runs.append(judge.judge_many(items))

    per_item: dict[str, dict] = {}
    for i, item in enumerate(items):
        q = (item.get("question") or "")[:40]
        entry: dict[str, Any] = {"question": q}
        for d in dims:
            vals = [
                runs[run][i].get(d)
                for run in range(args.repeats)
                if runs[run][i].get(d) is not None
            ]
            if vals:
                entry[d] = {
                    "mean": round(statistics.mean(vals), 3),
                    "stddev": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                }
        per_item[str(i)] = entry

    # 整体维度统计（全部轮次合并）
    all_vals: dict[str, list[float]] = {d: [] for d in dims}
    for run in runs:
        for r in run:
            for d in dims:
                v = r.get(d)
                if v is not None:
                    all_vals[d].append(float(v))
    per_dimension = {
        d: {
            "mean": round(statistics.mean(vals), 3) if vals else None,
            "stddev": round(statistics.stdev(vals), 3) if len(vals) > 1 else None,
            "n": len(vals),
        }
        for d, vals in all_vals.items()
    }

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "testset": Path(args.eval_file).name,
        "sample_size": len(items),
        "repeats": args.repeats,
        "judge_model": judge.model,
        "judge_model_version": judge.model_version,
        "per_dimension": per_dimension,
        "per_item": per_item,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n方差基线已写入:", args.output)
    print(json.dumps(per_dimension, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
