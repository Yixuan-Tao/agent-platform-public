"""修复 VL 描述缺失：删除指定文档旧数据 + 清除缓存 + 重新摄入。

用法：
    python scripts/fix_vl_descriptions.py                                  # 默认 PESD1IVN27-A
    python scripts/fix_vl_descriptions.py --pdf path/to/file.pdf           # 指定其他 PDF
    python scripts/fix_vl_descriptions.py --skip-cache                    # 跳过缓存清理
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ingest import IngestPipeline
from utils.logger import log

CACHE_DIR = ROOT / "cache" / "parse_cache"
DEFAULT_PDF = ROOT / "example" / "C19170884_静电和浪涌保护[TVS-ESD]_PESD1IVN27-A-ES_规格书_p8B69T.pdf"


def main() -> int:
    parser = argparse.ArgumentParser(description="修复 VL 描述缺失：清缓存 + 重新摄入")
    parser.add_argument(
        "--pdf", type=str, default=str(DEFAULT_PDF),
        help="目标 PDF 路径（默认 PESD1IVN27-A 规格书）",
    )
    parser.add_argument(
        "--skip-cache", action="store_true",
        help="跳过解析缓存清理（保留缓存可加快摄入）",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        log.error(f"PDF 不存在: {pdf_path}")
        return 1

    # 1) 清除解析缓存（强制重新解析，触发 VL 重新生成描述）
    if not args.skip_cache:
        if CACHE_DIR.exists():
            for pkl in CACHE_DIR.glob("*.pkl"):
                pkl.unlink()
                log.info(f"删除缓存: {pkl.name}")
        else:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 2) 删除 ChromaDB 旧数据
    pipeline = IngestPipeline()
    try:
        pipeline.delete_document(pdf_path.name)
        log.info(f"删除 ChromaDB 旧数据: {pdf_path.name}")
    except Exception as e:
        log.warning(f"删除旧数据警告（可能不存在）: {e}")

    # 3) 重新摄入（VL 会重新生成描述）
    log.info(f"重新摄入: {pdf_path.name}")
    pipeline.ingest_file(str(pdf_path))
    log.info("完成！VL 描述已重新生成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
