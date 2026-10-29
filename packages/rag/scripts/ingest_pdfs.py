"""
全链路测试 — Step 1: 直接摄入 4 份 PDF
使用 IngestPipeline 直接调用，绕过 HTTP API，更可靠
"""
import os, time, sys, json
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ingest import IngestPipeline
from db.chroma_client import ChromaClientSingleton
from utils.logger import log

PDF_DIR = Path(__file__).resolve().parent.parent / "example"
LOG_PATH = Path(__file__).resolve().parent.parent / "ingest_result.json"

def main():
    pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')])
    print(f"找到 {len(pdfs)} 份 PDF")
    print("=" * 60)

    pipeline = IngestPipeline()
    results = []
    total_start = time.time()

    for i, pdf_name in enumerate(pdfs):
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        print(f"\n[{i+1}/{len(pdfs)}] 摄入: {pdf_name[:50]}...")
        t0 = time.time()

        result = pipeline.ingest_file(str(pdf_path))
        elapsed = time.time() - t0

        result['elapsed_seconds'] = round(elapsed, 1)
        result['pdf_name'] = pdf_name
        results.append(result)

        status = result.get('status', 'unknown')
        if status == 'ok':
            print(f"  成功 | chunks={result.get('chunk_count', '?')} | 耗时 {elapsed:.1f}s")
        else:
            print(f"  失败 | {result.get('error', '')[:80]} | 耗时 {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"全部摄入完成，总耗时 {total_elapsed:.1f}s")

    # 汇总
    chroma = ChromaClientSingleton.get_instance()
    stats = chroma.get_stats()
    print(f"ChromaDB 统计: {json.dumps(stats, ensure_ascii=False)}")

    # 保存结果
    report = {
        'total_time_seconds': round(total_elapsed, 1),
        'chroma_stats': stats,
        'results': results,
    }
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {LOG_PATH}")

    # 打印摘要
    ok = sum(1 for r in results if r.get('status') == 'ok')
    fail = len(results) - ok
    total_chunks = sum(r.get('chunk_count', 0) for r in results)
    print(f"\n摘要: {ok} 成功, {fail} 失败, 共 {total_chunks} 个 chunk")

if __name__ == '__main__':
    main()
