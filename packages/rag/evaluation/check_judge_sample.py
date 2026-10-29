"""校验裁判结果样本量，防止"未裁判却判定通过"的空跑。

用法（在 packages/rag 目录）：
    python evaluation/check_judge_sample.py evaluation/run_result_v6_judged.json --min 90
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_judged_count(path: str | Path) -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return int(data.get("judge_summary", {}).get("judged", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description="校验裁判样本量")
    parser.add_argument("path", help="已裁判的评测结果 JSON")
    parser.add_argument("--min", type=int, default=90, dest="minimum")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"[错误] 结果文件不存在: {target}")
        return 2
    judged = load_judged_count(target)
    if judged < args.minimum:
        print(f"[错误] 裁判样本不足: {judged} < {args.minimum}")
        return 1
    print(f"裁判样本数: {judged}（门槛 {args.minimum}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
