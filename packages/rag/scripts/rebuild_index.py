"""重建本地索引：备份 → 清空 → 摄入 example/ 全部文档（PDF/DOCX/TXT/MD）→ 校验。

用法（在 packages/rag 目录下）:
    python scripts/rebuild_index.py --confirm [--backup-dir 自定义备份目录]

清空操作为破坏性动作，必须先备份；本脚本默认自动生成带时间戳的备份目录。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.chroma_client import ChromaClientSingleton  # noqa: E402
from pipeline.ingest import IngestPipeline  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = _ROOT / "example"
SUPPORTED_EXT = {".pdf", ".docx", ".doc", ".txt", ".md"}


def main() -> int:
    parser = argparse.ArgumentParser(description="重建本地索引（备份→清空→摄入 example/）")
    parser.add_argument("--confirm", action="store_true", help="确认清空当前索引")
    parser.add_argument("--backup-dir", default=None, help="自定义备份目录（默认自动时间戳）")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("索引重建会清空当前 chroma_data；请追加 --confirm")

    chroma_dir = _ROOT / "chroma_data"
    backup = Path(args.backup_dir) if args.backup_dir else _ROOT / f"chroma_data_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    if chroma_dir.exists():
        shutil.copytree(chroma_dir, backup)
        print(f"已备份: {chroma_dir} -> {backup}")

    chroma = ChromaClientSingleton.get_instance()
    removed = chroma.clear_all()
    print(f"已清空 {removed} 个旧 chunk")

    pipeline = IngestPipeline()
    files = sorted(p for p in EXAMPLE_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXT)
    print(f"待摄入 {len(files)} 份文档")
    results: list[dict] = []
    for i, path in enumerate(files, start=1):
        t0 = time.time()
        result = pipeline.ingest_file(str(path))
        result["file"] = path.name
        result["elapsed_s"] = round(time.time() - t0, 1)
        results.append(result)
        status = "OK" if result.get("status") == "ok" else "FAIL"
        print(f"[{i}/{len(files)}] {status} {path.name[:52]} "
              f"chunks={result.get('chunk_count', '?')} {result['elapsed_s']}s "
              f"{str(result.get('error', ''))[:90]}")
        sys.stdout.flush()

    docs = chroma.list_documents()
    stats = chroma.get_collection_stats()
    report = {
        "results": results,
        "documents": docs,
        "total_chunks": stats.get("row_count", 0),
    }
    (_ROOT / "index_rebuild_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(json.dumps({
        "ingested": ok,
        "total": len(results),
        "documents": len(docs),
        "chunks": report["total_chunks"],
        "report": "index_rebuild_report.json",
    }, ensure_ascii=False, indent=2))
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
