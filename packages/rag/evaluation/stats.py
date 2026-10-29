"""评测快照统计对比：Welch's t 检验 + 绝对阈值，输出「回归/改进/噪音/无变化」三态。"""

from __future__ import annotations

import math
import statistics
from typing import Any

ALPHA = 0.05
DELTA_THRESHOLD = 0.03
_DIMS = ("score", "faithfulness", "relevance", "completeness", "citation_quality")


def _welch_t(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Welch's t 统计量与自由度；样本不足或方差为 0 时返回 None。"""
    if len(xs) < 2 or len(ys) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx, vy = statistics.variance(xs), statistics.variance(ys)
    denom = vx / len(xs) + vy / len(ys)
    if denom <= 0:
        return None
    t = (mx - my) / math.sqrt(denom)
    df = denom**2 / (
        (vx / len(xs)) ** 2 / (len(xs) - 1) + (vy / len(ys)) ** 2 / (len(ys) - 1)
    )
    return t, max(df, 1e-6)


def _t_pvalue(t: float, df: float) -> float:
    """双侧 t 检验 p 值（Simpson 数值积分，无 scipy 依赖）。"""
    if not math.isfinite(t) or df <= 0:
        return 1.0
    abs_t = abs(t)
    if abs_t > 30:
        return 0.0
    const = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(
        df * math.pi
    )

    def density(x: float) -> float:
        return const * (1 + x * x / df) ** (-(df + 1) / 2)

    lo, hi = abs_t, abs_t + 40.0
    n = 4000
    h = (hi - lo) / n
    s = density(lo) + density(hi)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * density(lo + i * h)
    tail = s * h / 3
    return min(1.0, 2.0 * tail)


def _extract_scores(snapshot: dict[str, Any]) -> dict[str, list[float]]:
    """从评测结果快照提取各维度分值列表（judge 字段在 results[i]['judge']）。"""
    out: dict[str, list[float]] = {d: [] for d in _DIMS}
    for r in snapshot.get("results") or []:
        judge = r.get("judge") or {}
        for d in _DIMS:
            v = judge.get(d)
            if isinstance(v, (int, float)) and v is not None:
                out[d].append(float(v))
    return out


def compare_judge_snapshots(
    prev: dict[str, Any],
    cur: dict[str, Any],
    *,
    delta_threshold: float = DELTA_THRESHOLD,
) -> dict[str, Any]:
    """对比两轮裁判快照，逐维度判定：regression / improvement / noise / no_change。"""
    prev_vals = _extract_scores(prev)
    cur_vals = _extract_scores(cur)
    dims_report: dict[str, Any] = {}
    verdicts: list[str] = []
    for d in _DIMS:
        xs, ys = prev_vals[d], cur_vals[d]
        if not xs or not ys:
            dims_report[d] = {"prev_avg": None, "cur_avg": None, "delta": None, "p_value": None, "verdict": "no_change"}
            verdicts.append("no_change")
            continue
        mx, my = statistics.mean(xs), statistics.mean(ys)
        delta = my - mx
        welch = _welch_t(xs, ys)
        p = _t_pvalue(*welch) if welch else None
        if abs(delta) < delta_threshold:
            verdict = "no_change"
        elif delta < 0 and (p is None or p < ALPHA):
            verdict = "regression"
        elif delta > 0 and (p is None or p < ALPHA):
            verdict = "improvement"
        else:
            verdict = "noise"
        verdicts.append(verdict)
        dims_report[d] = {
            "prev_avg": round(mx, 3),
            "cur_avg": round(my, 3),
            "delta": round(delta, 3),
            "p_value": round(p, 4) if p is not None else None,
            "verdict": verdict,
        }
    if "regression" in verdicts:
        overall = "regression"
    elif "improvement" in verdicts:
        overall = "improvement"
    elif "noise" in verdicts:
        overall = "noise"
    else:
        overall = "no_change"
    return {"overall": overall, "dimensions": dims_report}


def main() -> int:
    """CLI：对比两轮裁判快照，回归时以非零退出码告警。"""
    import argparse
    import json as _json
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(description="裁判快照回归检测")
    parser.add_argument("--prev", required=True, help="上一轮裁判快照 JSON")
    parser.add_argument("--cur", required=True, help="本轮裁判快照 JSON")
    args = parser.parse_args()
    prev = _json.loads(_Path(args.prev).read_text(encoding="utf-8"))
    cur = _json.loads(_Path(args.cur).read_text(encoding="utf-8"))
    report = compare_judge_snapshots(prev, cur)
    print(_json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report.get("overall") == "regression" else 0


if __name__ == "__main__":
    raise SystemExit(main())
