"""
多模态 RAG 系统 — 摄入 Pipeline (Task 7)

职责：
    1. 把 parser.DocumentChunk 转成 db.schemas.DocumentChunk
       （字段名差异：chunk_id → id，position/child_images/parent_chunk_id 合并到 metadata）
    2. 调用 embedder.embed_chunks 完成向量化
    3. 写入 ChromaDB，返回标准化的摄入结果
    4. 暴露纯文本快速摄入与按文件名删除两个辅助方法
"""
from __future__ import annotations

import time
import hashlib
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from core.embedder import Embedder
from core.parser import DocumentChunk as _ParserChunk
from core.parser import DocumentParser
from db.chroma_client import ChromaClientSingleton
from db.schemas import MODALITY_IMAGE, MODALITY_TEXT, DocumentChunk
from utils.logger import log


class IngestPipeline:
    """文档摄入编排：parse → 类型转换 → embed → 写入 ChromaDB。"""

    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        embedder: Optional[Embedder] = None,
        chroma_client: Optional[ChromaClientSingleton] = None,
    ) -> None:
        # 三件套懒加载：测试可注入，生产用默认单例
        self._parser = parser
        self._embedder = embedder
        self._chroma = chroma_client

    # ---------- 懒加载属性 ----------

    @property
    def parser(self) -> DocumentParser:
        if self._parser is None:
            self._parser = DocumentParser()
        return self._parser

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def chroma(self) -> ChromaClientSingleton:
        if self._chroma is None:
            self._chroma = ChromaClientSingleton.get_instance()
        return self._chroma

    # ---------- 公开方法 ----------

    def ingest_file(self, file_path: str, file_name: str | None = None) -> dict[str, Any]:
        """单文件摄入：parse → 转换 → embed → insert。

        参数 ``file_name`` 用于覆盖入库元数据的文件名：上传流程会把文件先落盘为
        ``{task_id}_{原名}``，若直接使用路径名，文档列表与按名删除都会带随机前缀。

        失败统一返回 ``{"status": "error", "error": str}``。
        """
        doc_id = ""
        try:
            source = Path(file_path)
            source_hash = None
            if source.is_file():
                with source.open("rb") as stream:
                    source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            # 1) 解析
            parser_chunks: list[_ParserChunk] = self.parser.parse(file_path, file_name=file_name)
            if not parser_chunks:
                return {
                    "status": "error",
                    "error": f"解析为空或失败: {file_path}",
                }

            # 2) 类型转换：parser.DocumentChunk → db.schemas.DocumentChunk
            db_chunks: list[DocumentChunk] = [
                self._convert_parser_chunk(pc) for pc in parser_chunks
            ]
            if source_hash is not None:
                with source.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != source_hash:
                        raise ValueError("源文件在解析期间发生变化，取消本次摄入")
                for chunk in db_chunks:
                    chunk.metadata["source_content_sha256"] = source_hash
            doc_id = db_chunks[0].doc_id

            # 2.5) 若文档包含图片，VL 调用消耗了 OneAPI 全局令牌桶，
            #       用指数退避等待限流窗口恢复，避免紧接着的 embed_dense 被连坐限流
            has_images = any(
                pc.modality == "image" or pc.metadata.get("image_bytes")
                for pc in parser_chunks
            )
            if has_images:
                self._wait_for_rate_limit_recovery(parser_chunks)

            # 3) 向量化（in-place 填充 dense / sparse / image）
            log.info("[ingest_file] 进入向量化阶段: %d chunks", len(db_chunks))
            t0 = time.time()
            self.embedder.embed_chunks(db_chunks)
            log.info("[ingest_file] 向量化完成, 耗时=%.1fs", time.time() - t0)

            # 4) 写入 ChromaDB
            log.info("[ingest_file] 进入 ChromaDB 写入阶段: %d chunks", len(db_chunks))
            t0 = time.time()
            insert_result = self.chroma.insert(db_chunks)
            inserted = int((insert_result or {}).get("insert_count", 0))
            log.info("[ingest_file] ChromaDB 写入完成, 耗时=%.1fs", time.time() - t0)

            # 向量化失败（例如 embedding 网关不可用/返回 404）时插入会静默为 0 条，
            # 若在这里当成成功返回，任务会显示 done 并给出一个查不到的 doc_id。
            if inserted <= 0:
                if doc_id:
                    try:
                        self.chroma.delete_by_doc_id(doc_id)
                    except Exception as rollback_error:  # noqa: BLE001
                        log.error("摄入回滚失败 doc_id=%s: %s", doc_id, rollback_error)
                return {
                    "status": "error",
                    "error": (
                        "向量化或写入失败：0 条 chunk 进入索引。"
                        "请检查 EMBED_MODEL / ONEAPI_BASE_URL / ONEAPI_API_KEY 配置"
                        "（对话网关通常不提供 /embeddings 接口）。"
                    ),
                }

            # 5) 返回标准化结果
            first = db_chunks[0]
            return {
                "file_name": first.file_name,
                "doc_id": first.doc_id,
                "chunk_count": len(db_chunks),
                "inserted_count": inserted,
                "status": "ok",
            }
        except Exception as e:  # noqa: BLE001
            # Chroma 的批量写入并非跨 collection 原子操作。任何中途失败都删除
            # 本次导入的 doc_id，避免查询到不完整文档或资料包版本混杂。
            if doc_id:
                try:
                    self.chroma.delete_by_doc_id(doc_id)
                except Exception as rollback_error:  # noqa: BLE001
                    log.error("摄入回滚失败 doc_id=%s: %s", doc_id, rollback_error)
            log.exception("摄入失败 %s: %s", file_path, e)
            return {"status": "error", "error": str(e)}

    def ingest_text(
        self,
        text: str,
        file_name: str = "inline.txt",
    ) -> dict[str, Any]:
        """直接摄入纯文本：构造单 chunk，模态 = text。"""
        try:
            if not text or not text.strip():
                return {"status": "error", "error": "空文本，跳过"}

            # 构造单 chunk；id/doc_id 用随机 hex 保证幂等性（同一内容多次摄入会重复）
            chunk = DocumentChunk(
                id=uuid4().hex[:16],
                doc_id=uuid4().hex[:12],
                file_name=file_name,
                modality=MODALITY_TEXT,
                content=text,
                searchable_text=text,
                page=0,
                section="",
                metadata={"source": "inline_text"},
            )
            self.embedder.embed_chunks([chunk])
            self.chroma.insert([chunk])
            return {
                "file_name": file_name,
                "doc_id": chunk.doc_id,
                "chunk_count": 1,
                "status": "ok",
            }
        except Exception as e:  # noqa: BLE001
            log.exception("摄入文本失败: %s", e)
            return {"status": "error", "error": str(e)}

    def ingest_image(
        self,
        image_bytes: bytes,
        file_name: str,
        description: str,
        page: int = 0,
        section: str = "",
    ) -> dict[str, Any]:
        """摄入单张图片：构造 image 模态 chunk，调用 ImageEmbedder 编码。

        Args:
            image_bytes: 原始图片 bytes（PNG / JPEG 等）。
            file_name: 原始文件名（建议带后缀）。
            description: 文本描述（VL 模型生成 或 人工标注），同时作为 searchable_text。
            page: 页码（图片所在 PDF 页），默认 0。
            section: 章节标题，默认空。

        Returns:
            ``{"status": "ok"|"error", "doc_id": ..., "chunk_count": int, ...}``
        """
        try:
            if not image_bytes:
                return {"status": "error", "error": "image_bytes 为空"}
            if not description or not description.strip():
                return {"status": "error", "error": "description 为空"}

            chunk = DocumentChunk(
                id=uuid4().hex[:16],
                doc_id=uuid4().hex[:12],
                file_name=file_name,
                modality=MODALITY_IMAGE,
                content=description,
                searchable_text=description,
                page=int(page),
                section=section,
                metadata={
                    "source": "inline_image",
                    "image_bytes": bytes(image_bytes),
                },
            )
            # 触发 image embedding（写到 chunk.image_vector 与 metadata.embedding_status）
            self.embedder.embed_chunks([chunk])
            self.chroma.insert([chunk])
            return {
                "file_name": file_name,
                "doc_id": chunk.doc_id,
                "chunk_count": 1,
                "chunk_id": chunk.id,
                "embedding_status": chunk.metadata.get("embedding_status", "unknown"),
                "image_vector_dim": len(chunk.image_vector),
                "status": "ok",
            }
        except Exception as e:  # noqa: BLE001
            log.exception("摄入图片失败: %s", e)
            return {"status": "error", "error": str(e)}

    def delete_document(self, file_name: str) -> dict[str, Any]:
        """按文件名删除该文档所有 chunk。"""
        try:
            count = self.chroma.delete_by_file_name(file_name)
            return {"status": "ok", "file_name": file_name, "deleted_count": count}
        except Exception as e:  # noqa: BLE001
            log.exception("删除文档失败 %s: %s", file_name, e)
            return {"status": "error", "error": str(e)}

    # ---------- 内部转换 ----------

    def _wait_for_rate_limit_recovery(self, parser_chunks: list[_ParserChunk]) -> None:
        """VL 调用后的限流恢复等待，按图片数量做比例退避。

        图片 chunk 数量越多，VL 调用越密集，需要更长的冷却时间。
        上限 10s，避免过度阻塞。
        """
        if not parser_chunks:
            return
        image_count = sum(1 for pc in parser_chunks if pc.modality == "image")
        if image_count <= 0:
            return
        # 每张图片等待 1.5s，上限 10s
        wait = min(1.5 * image_count, 10.0)
        log.info("文档包含 %d 个图片 chunk，等待 %.1fs 让 OneAPI 限流窗口恢复 ...",
                 image_count, wait)
        time.sleep(wait)

    @staticmethod
    def _convert_parser_chunk(pc: _ParserChunk) -> DocumentChunk:
        """把 parser.DocumentChunk 转成 db.schemas.DocumentChunk。

        字段映射要点：
            - ``chunk_id`` → ``id``
            - ``position`` 写入 ``metadata["position"]``
            - ``child_images`` 写入 ``metadata["child_images"]``
            - ``parent_chunk_id`` 写入 ``metadata["parent_chunk_id"]``
            - 保留原有 ``metadata`` 字段
        """
        # 防御性复制 metadata，避免修改上游对象
        meta = dict(pc.metadata or {})
        if pc.position:
            meta["position"] = dict(pc.position)
        if pc.child_images:
            meta["child_images"] = list(pc.child_images)
        if pc.parent_chunk_id:
            meta["parent_chunk_id"] = pc.parent_chunk_id

        return DocumentChunk(
            id=pc.chunk_id,
            doc_id=pc.doc_id,
            file_name=pc.file_name,
            modality=pc.modality,
            content=pc.content,
            searchable_text=pc.searchable_text,
            page=int(pc.page or 0),
            section=pc.section or "",
            metadata=meta,
            # 章节结构字段透传（章节路由对比依赖，丢失会导致 retrieve_by_section 失配）
            section_id=pc.section_id or "",
            section_path=pc.section_path or "",
            section_topic=pc.section_topic or "",
        )


__all__ = ["IngestPipeline"]
