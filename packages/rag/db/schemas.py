"""
ChromaDB Collection 定义 + DocumentChunk 数据类。

统一管理多模态 RAG 系统中所有需要存入 ChromaDB 的文档切片的字段结构、
向量维度以及模态常量。

采用两 collection 设计：
    - ``document_chunks``：text/table 模态，使用 ``dense_vector`` (1024 dim)
    - ``image_chunks``：image 模态，使用 ``image_vector`` (1024 dim)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from config import config

# Python 3.13 移除了 imghdr，用魔数检测替代
_IMG_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpg",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"RIFF": "webp",  # RIFF....WEBP
    b"BM": "bmp",
}


def _detect_image_ext(data: bytes) -> str:
    """通过魔数检测图片格式，返回扩展名。"""
    for magic, ext in _IMG_MAGIC.items():
        if data.startswith(magic):
            if ext == "webp" and len(data) >= 12 and data[8:12] == b"WEBP":
                return "webp"
            if ext == "webp":
                continue  # RIFF 但不是 WEBP
            return ext
    return "png"


# ============================================================
# 常量定义
# ============================================================

# 文本/表格 collection 名称
COLLECTION_NAME: str = "document_chunks"
# 图片 collection 名称
IMAGE_COLLECTION_NAME: str = "image_chunks"

# 模态常量（与 DocumentChunk.modality 字段对齐）
MODALITY_TEXT: str = "text"
MODALITY_IMAGE: str = "image"
MODALITY_TABLE: str = "table"

# 向量维度（从 config 读取，保持单一来源）
_DENSE_DIM: int = config.DENSE_EMBED_DIM
_IMAGE_DIM: int = config.IMAGE_EMBED_DIM


# ============================================================
# Pipeline 内统一使用的 dataclass
# ============================================================

# Chunk 角色常量（父子双层结构）
CHUNK_ROLE_PARENT: str = "parent"  # 父 chunk（page/heading 级别上下文，不入向量库）
CHUNK_ROLE_CHILD: str = "child"    # 子 chunk（按 token 切分后入向量库）


@dataclass
class DocumentChunk:
    """在 pipeline 各阶段之间传递的文档切片数据类。

    字段说明：
        - ``dense_vector``：list[float]，长度 == config.DENSE_EMBED_DIM
        - ``sparse_vector``：dict[int, float]，BM25 词项 id -> 权重（保留在内存中，不写入 DB）
        - ``image_vector``：list[float]，长度 == config.IMAGE_EMBED_DIM（仅 image 模态使用）
        - ``chunk_role``：``"parent"`` / ``"child"``，父子双层结构角色
        - ``is_table_only``：表格独立 chunk 标记
        - ``parent_chunk_id``：父 chunk 的 id（child chunk 通过此字段挂回父）

    提示：使用 ``field(default_factory=...)`` 避免可变默认值的共享陷阱。
    """

    id: str
    doc_id: str
    file_name: str
    modality: str
    content: str
    searchable_text: str
    page: int = 0
    section: str = ""
    # 向量字段使用 default_factory，避免所有实例共享同一个 list/dict
    dense_vector: list[float] = field(default_factory=list)
    sparse_vector: dict[int, float] = field(default_factory=dict)
    image_vector: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # 父子双层结构（向后兼容：旧数据无此字段时默认为 child）
    parent_chunk_id: str = ""
    chunk_role: str = CHUNK_ROLE_CHILD
    is_table_only: bool = False
    # 章节结构字段（标准类文档对比路由用，默认空）
    section_id: str = ""        # 层级化编号，如 "5", "5.3", "5.3.2"
    section_path: str = ""      # 完整路径，如 "5.3.2 / BCI Test"
    section_topic: str = ""     # 章节主题（从 SECTION_TOPIC_KEYWORDS 路由表匹配）

    # ---------- ChromaDB 实体转换 ----------

    def to_chroma_entity(self) -> tuple[str, list[float], dict[str, Any], str] | None:
        """根据模态返回 ChromaDB 的 (id, embedding, metadata, document) 元组。

        - text/table：使用 ``dense_vector``
        - image：使用 ``image_vector``
        - parent chunk：返回 None（不入向量库，仅作为 child 的上下文锚点）
        """
        # 父 chunk 不直接入向量库（仅作为 child 的上下文来源）
        if self.chunk_role == CHUNK_ROLE_PARENT:
            return None
        meta = self._build_chroma_metadata()
        if self.modality == MODALITY_IMAGE:
            vec = self.image_vector if self.image_vector else [0.0] * _IMAGE_DIM
            return (self.id, list(vec), meta, self.content)
        else:
            vec = self.dense_vector if self.dense_vector else [0.0] * _DENSE_DIM
            return (self.id, list(vec), meta, self.content)

    def to_text_entity(self) -> tuple[str, list[float], dict[str, Any], str]:
        """返回使用 ``dense_vector`` 的 ChromaDB 实体元组。

        用于将 image 模态 chunk 同时写入 document_chunks collection，
        使文本检索能通过 VL 描述向量命中图片 chunk。
        """
        meta = self._build_chroma_metadata()
        vec = self.dense_vector if self.dense_vector else [0.0] * _DENSE_DIM
        return (self.id, list(vec), meta, self.content)

    def _build_chroma_metadata(self) -> dict[str, Any]:
        """构建 ChromaDB 兼容的扁平 metadata 字典。

        ChromaDB metadata 不支持嵌套 dict，因此将原始 metadata 序列化为 JSON 字符串。
        image_bytes 写入磁盘（chroma_data/images/{chunk_id}.{ext}），metadata 只存路径。

        父子双层结构相关字段（chunk_role / parent_chunk_id / is_table_only）写入顶层
        metadata，便于检索时按 chunk_role 过滤或回查 parent context。
        """
        meta = dict(self.metadata)
        image_bytes = meta.pop("image_bytes", None)

        chroma_meta = {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "modality": self.modality,
            "searchable_text": (self.searchable_text or "")[:8192],
            "page": int(self.page),
            "section": self.section or "",
            "chunk_role": self.chunk_role or CHUNK_ROLE_CHILD,
            "parent_chunk_id": self.parent_chunk_id or "",
            "is_table_only": bool(self.is_table_only),
            "section_id": self.section_id or "",
            "section_path": self.section_path or "",
            "section_topic": self.section_topic or "",
            "metadata_json": json.dumps(meta, ensure_ascii=False),
        }

        # 图片 bytes 存磁盘，metadata 只存路径
        if image_bytes and isinstance(image_bytes, bytes):
            img_dir = os.path.join(config.CHROMA_DB_PATH, "images")
            os.makedirs(img_dir, exist_ok=True)
            ext = _detect_image_ext(image_bytes)
            img_path = os.path.join(img_dir, f"{self.id}.{ext}")
            try:
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                chroma_meta["image_path"] = img_path
            except OSError:
                pass  # 磁盘写入失败，不阻塞入库

        return chroma_meta
