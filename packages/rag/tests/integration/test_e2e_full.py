# Test values are synthetic fixtures for software behavior.
"""
全链路端到端测试（终极版）。

流程：
    1. 清理数据库（删除 chroma_data 目录 + 重置单例）
    2. 摄入所有文档（example/*.pdf）
    3. 运行全部 42 题评估（含对比题，沙箱外模式）
    4. 生成 HTML 报告

用法：
    python tests/integration/test_e2e_full.py              # 完整流程
    python tests/integration/test_e2e_full.py --ingest-only  # 仅摄入，不评估
    python tests/integration/test_e2e_full.py --eval-only    # 仅评估（跳过摄入）
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# 项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import config
from db.chroma_client import ChromaClientSingleton
from pipeline.ingest import IngestPipeline
from utils.logger import log

# ---- 路径常量 ----
EXAMPLE_DIR = _PROJECT_ROOT / "example"
CHROMA_DB_PATH = Path(config.CHROMA_DB_PATH).resolve()
OUT_JSON = _PROJECT_ROOT / "qa_eval_full.json"


# ============================================================
# 阶段 1: 清理数据库
# ============================================================

def clean_database() -> None:
    """清空 ChromaDB 数据：删除持久化目录 + 重置单例。"""
    print("=" * 60)
    print("[阶段 1/3] 清理数据库")
    print("=" * 60)

    # 重置单例（关闭已有连接 + 置空引用）
    ChromaClientSingleton.reset_instance()

    # 删除 chroma_data 目录
    if CHROMA_DB_PATH.exists():
        print(f"  删除 {CHROMA_DB_PATH} ...")
        shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
        print("  [OK] 数据库目录已删除")
    else:
        print("  [SKIP] 数据库目录不存在，无需清理")

    # 删除旧的评估结果，确保全新评估
    if OUT_JSON.exists():
        print(f"  删除旧评估结果 {OUT_JSON.name} ...")
        OUT_JSON.unlink()
        print("  [OK] 旧评估结果已删除")

    print()


# ============================================================
# 阶段 2: 摄入文档
# ============================================================

def ingest_all_documents() -> list[dict]:
    """摄入 example/ 下所有 PDF，返回每个文件的摄入结果。"""
    print("=" * 60)
    print("[阶段 2/3] 摄入文档（解析 + 向量化 + 写入 ChromaDB）")
    print("=" * 60)

    pdf_files = sorted(EXAMPLE_DIR.glob("*.pdf"))
    if not pdf_files:
        print("  [错误] example/ 下没有找到 PDF 文件！")
        return []

    print(f"  找到 {len(pdf_files)} 个 PDF 文件:\n")
    for f in pdf_files:
        print(f"    - {f.name}")

    pipeline = IngestPipeline()
    results: list[dict] = []
    failed: list[str] = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n  [{idx}/{len(pdf_files)}] 摄入: {pdf_path.name}")
        t0 = time.time()
        result = pipeline.ingest_file(str(pdf_path))
        elapsed = time.time() - t0

        if result.get("status") == "ok":
            print(f"    [OK] {result.get('chunk_count', 0)} chunks, "
                  f"doc_id={result.get('doc_id', '?')}, "
                  f"耗时 {elapsed:.1f}s")
        else:
            print(f"    [FAIL] {result.get('error', '未知错误')}")
            failed.append(pdf_path.name)

        results.append(result)

    print(f"\n  摄入完成: {len(pdf_files) - len(failed)}/{len(pdf_files)} 成功")
    if failed:
        print(f"  失败文件: {', '.join(failed)}")

    print()
    return results


# ============================================================
# 阶段 3: 运行评估
# ============================================================

def run_full_evaluation() -> None:
    """运行全部 42 题评估 + 生成 HTML 报告。"""
    print("=" * 60)
    print("[阶段 3/3] 运行全链路评估（42 题，含对比题）")
    print("=" * 60)

    # 导入评估模块（延迟导入，避免阶段 1 前就初始化 ChromaDB）
    from tests.integration.test_eval import generate_reports, run_evaluation

    results = run_evaluation(force=True, no_skip=True)
    generate_reports(results)

    # 打印汇总
    total = len(results)
    avg_ans = sum(r.get("answer_score", 0) for r in results) / max(total, 1)
    avg_ret = sum(r.get("retrieval_score", 0) for r in results) / max(total, 1)
    avg_lat = sum(r.get("latency_ms", 0) for r in results) / max(total, 1)
    errors = [r for r in results if r.get("error") and r["error"] != "SKIPPED"]

    print(f"\n  评估完成: {total} 题")
    print(f"  平均 answer_score: {avg_ans:.2%}")
    print(f"  平均 retrieval_score: {avg_ret:.2%}")
    print(f"  平均延迟: {avg_lat:.0f}ms")
    if errors:
        print(f"  异常题数: {len(errors)}")
        for e in errors:
            print(f"    - {e['question'][:50]}... | {e['error']}")

    print()


# ============================================================
# 获取数据库统计
# ============================================================

def show_db_stats() -> None:
    """打印当前 ChromaDB 数据库统计信息。"""
    try:
        client = ChromaClientSingleton.get_instance()
        if not client.is_connected:
            client.connect()
        stats = client.get_collection_stats()
        docs = client.list_documents()
        print(f"  ChromaDB 状态: {stats['names']}, 共 {stats['row_count']} 条 chunk")
        for d in docs:
            print(f"    {d['file_name']}: {d['chunk_count']} chunks")
    except Exception as e:
        print(f"  [警告] 无法获取数据库统计: {e}")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG 全链路端到端测试（终极版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tests/integration/test_e2e_full.py               # 完整流程（清理 → 摄入 → 30 题评估）
  python tests/integration/test_e2e_full.py --ingest-only  # 仅摄入文档
  python tests/integration/test_e2e_full.py --eval-only    # 仅评估（跳过摄入）
  python tests/integration/test_e2e_full.py --skip-clean   # 保留数据库，直接摄入+评估
        """,
    )
    parser.add_argument("--ingest-only", action="store_true",
                        help="仅摄入文档，不运行评估")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅运行评估（跳过清理和摄入）")
    parser.add_argument("--skip-clean", action="store_true",
                        help="跳过清理数据库步骤（保留已有数据）")
    args = parser.parse_args()

    t_start = time.time()

    # ---- 仅评估模式 ----
    if args.eval_only:
        run_full_evaluation()
        elapsed = time.time() - t_start
        print(f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
        return

    # ---- 完整流程 ----
    if not args.skip_clean:
        clean_database()

    ingest_results = ingest_all_documents()

    if args.ingest_only:
        print("=" * 60)
        print("[完成] 摄入完毕，数据库状态:")
        show_db_stats()
        elapsed = time.time() - t_start
        print(f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
        return

    # 摄入后展示数据库状态
    show_db_stats()
    print()

    run_full_evaluation()

    elapsed = time.time() - t_start
    print(f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
