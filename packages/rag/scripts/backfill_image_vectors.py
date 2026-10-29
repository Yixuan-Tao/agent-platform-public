"""
图片向量回填脚本：为已入库的 image chunk 生成真实多模态向量并写入 image_chunks 集合。

背景：
    旧索引中图片只有 VL 识图描述文本（dense_vector），真实图片向量为空
    （image_chunks 集合为空）。本脚本不重新解析文档、不重复调用 VL，
    直接从 document_chunks 读取已保存的图片 bytes（chroma_data/images/*），
    用百炼原生多模态 Embedding 编码后 upsert 到 image_chunks。

用法（在 packages/rag 目录下运行）：
    python scripts/backfill_image_vectors.py                 # 全量回填
    python scripts/backfill_image_vectors.py --limit 10      # 只处理前 10 张（小样本验证）
    python scripts/backfill_image_vectors.py --only-missing  # 跳过 image_chunks 已有的 id
    python scripts/backfill_image_vectors.py --dry-run       # 只统计，不调用 API

输出：
    metrics/image_backfill_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import config  # noqa: E402
from core.embedder import ImageEmbedder  # noqa: E402
from db.chroma_client import ChromaClientSingleton  # noqa: E402
from db.schemas import IMAGE_COLLECTION_NAME, MODALITY_IMAGE  # noqa: E402
from utils.logger import log  # noqa: E402

REPORT_FILE = _PROJECT_ROOT / "metrics" / "image_backfill_report.json"
_PAGE_SIZE = 100


def _collect_image_rows(chroma: ChromaClientSingleton) -> list[dict[str, Any]]:
    """从 document_chunks 分页拉取全部 modality=image 的记录。"""
    doc_coll = chroma._doc_collection
    if doc_coll is None:
        return []
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        batch = doc_coll.get(
            where={"modality": MODALITY_IMAGE},
            include=["metadatas", "documents"],
            limit=_PAGE_SIZE,
            offset=offset,
        )
        ids = batch.get("ids", []) or []
        if not ids:
            break
        metadatas = batch.get("metadatas", []) or []
        documents = batch.get("documents", []) or []
        for i, cid in enumerate(ids):
            if not cid:
                continue
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            rows.append({"id": str(cid), "metadata": meta or {}, "document": doc or ""})
        if len(ids) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


def _resolve_image_bytes(row: dict[str, Any]) -> Optional[bytes]:
    """从 metadata.image_path 或默认 images/ 目录解析图片 bytes。"""
    meta = row.get("metadata", {}) or {}
    cid = row["id"]
    candidates: list[Path] = []
    img_path = meta.get("image_path") or ""
    if img_path:
        candidates.append(Path(str(img_path)))
    # 兜底：默认 chroma_data/images/{chunk_id}.*
    images_dir = Path(config.CHROMA_DB_PATH) / "images"
    if images_dir.is_dir():
        candidates.extend(sorted(images_dir.glob(f"{cid}.*")))
    for p in candidates:
        try:
            if p.is_file():
                return p.read_bytes()
        except OSError as exc:
            log.warning("读取图片失败 %s: %s", p, exc)
    return None


def _upsert_vectors(chroma: ChromaClientSingleton, items: list[dict[str, Any]]) -> None:
    """把编码成功的图片向量 upsert 到 image_chunks。"""
    img_coll = chroma._image_collection
    if img_coll is None:
        return
    if not items:
        return
    img_coll.upsert(
        ids=[it["id"] for it in items],
        embeddings=[it["vector"] for it in items],
        metadatas=[it["metadata"] for it in items],
        documents=[it["document"] for it in items],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="图片向量回填（百炼原生多模态 Embedding）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理 N 张（0=全量）")
    parser.add_argument("--only-missing", action="store_true", help="跳过 image_chunks 已有的 id")
    parser.add_argument("--dry-run", action="store_true", help="只统计不调用 API")
    parser.add_argument("--workers", type=int, default=4, help="并发数（默认 4）")
    args = parser.parse_args()

    chroma = ChromaClientSingleton.get_instance()
    chroma.ensure_loaded()

    rows = _collect_image_rows(chroma)
    log.info("document_chunks 中共 %d 条 image chunk", len(rows))

    existing_ids: set[str] = set()
    if args.only_missing and chroma._image_collection is not None:
        existing = chroma._image_collection.get(include=[])
        existing_ids = {str(x) for x in (existing.get("ids", []) or [])}
        log.info("image_chunks 已有 %d 条，跳过", len(existing_ids))

    pending = [r for r in rows if r["id"] not in existing_ids]
    if args.limit and args.limit > 0:
        pending = pending[: args.limit]

    if not pending:
        print(f"无可回填图片（共 {len(rows)} 条，已有 {len(existing_ids)} 条）")
        return

    print(
        f"待回填: {len(pending)} 张 | model={config.IMAGE_EMBED_MODEL} "
        f"| dim={config.IMAGE_EMBED_DIM} | url={config.DASHSCOPE_MULTIMODAL_URL}"
    )
    if args.dry_run:
        missing_bytes = sum(
            1 for r in pending if _resolve_image_bytes(r) is None
        )
        print(f"dry-run: 共 {len(pending)} 张，缺图片文件 {missing_bytes} 张")
        return

    embedder = ImageEmbedder()
    if embedder._disabled or not embedder._native:
        print(
            "ImageEmbedder 不可用（native=%s, disabled=%s）：请确认 DASHSCOPE_API_KEY / "
            "LLM_API_KEY 与 DASHSCOPE_WORKSPACE_ID 配置，且未设置 DISABLE_IMAGE_EMBED=1",
            embedder._native, embedder._disabled,
        )
        return

    # 预解析图片 bytes（不占用 API 配额）
    for row in pending:
        row["raw"] = _resolve_image_bytes(row)

    ok_items: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="img-backfill") as pool:
        futures = {
            pool.submit(embedder.embed, row["raw"]): row
            for row in pending
            if row.get("raw")
        }
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                vec = fut.result()
            except Exception as exc:  # noqa: BLE001
                vec = None
                row["error"] = f"{type(exc).__name__}: {exc}"
            if vec and len(vec) == config.IMAGE_EMBED_DIM:
                ok_items.append({
                    "id": row["id"],
                    "vector": list(vec),
                    "metadata": row["metadata"],
                    "document": row["document"],
                })
            else:
                failed.append({"id": row["id"], "error": row.get("error", "empty_vector")})
        for row in pending:
            if not row.get("raw"):
                failed.append({"id": row["id"], "error": "missing_image_file"})

    _upsert_vectors(chroma, ok_items)
    chroma.invalidate_list_cache()

    elapsed = time.time() - t0
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": config.IMAGE_EMBED_MODEL,
        "dim": config.IMAGE_EMBED_DIM,
        "endpoint": config.DASHSCOPE_MULTIMODAL_URL,
        "total_chunks": len(rows),
        "requested": len(pending),
        "success": len(ok_items),
        "failed": len(failed),
        "elapsed_s": round(elapsed, 1),
        "failed_items": failed[:50],
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"回填完成: 成功 {len(ok_items)} / {len(pending)} 张，"
        f"耗时 {elapsed:.1f}s → image_chunks 现有 "
        f"{chroma._image_collection.count() if chroma._image_collection else 0} 条"
    )
    print(f"报告已保存: {REPORT_FILE}")


if __name__ == "__main__":
    main()
