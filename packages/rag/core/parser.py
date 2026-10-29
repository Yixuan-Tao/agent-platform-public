"""
多模态 RAG 系统 — 文档解析层 (Task 3)

职责：
  1. 用 LiteParse 提取 PDF/DOCX/图片中的文本、表格、图片区域（带坐标 + 页码）
  2. 按标题深度分块（H1/H2/H3...），同一标题下的文本/表格聚合成一个 chunk
  3. 对图片区域调用 Qwen3-VL-Plus 视觉理解，生成中文语义描述
  4. 对表格调用 Qwen3.7-Max 生成语义摘要
  5. 输出 DocumentChunk 列表，供 ingest pipeline 写入 ChromaDB

设计说明：
  - 本文件顶部定义本地 DocumentChunk 数据类，与 db/schemas.py 同名同字段，
    主流程在 ingest pipeline 里做转换，避免循环依赖。
  - LiteParse v2.0.0（已安装）：基于 Rust + pdfium，对 PDF 最优；
    DOCX/图片经内部转换或走图像分支。
  - 网络调用全部 try/except 降级，解析层永远返回可用 chunk 列表。
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

try:
    from PIL import Image  # 用于 VL 前压缩图片，降低 token 消耗
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from openai import OpenAI

from config import config
from db.schemas import (
    CHUNK_ROLE_CHILD,
    CHUNK_ROLE_PARENT,
    MODALITY_TABLE,
)
from utils.language_detector import detect_language
from utils.logger import log
from utils.tokenizer import count_tokens, split_by_tokens

# 尝试导入 LiteParse（v2.x 原生绑定）；失败时不抛错，在 parse 时降级
try:
    from liteparse import LiteParse as _LiteParse  # type: ignore
    _LITEPARSE_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    _LITEPARSE_AVAILABLE = False
    log.warning(f"LiteParse 不可用，解析层将返回空列表: {_e}")

# PyMuPDF：用于提取 PDF 中的矢量图形（LiteParse 只能检测光栅图，无法检测矢量图）
try:
    import fitz as _fitz  # type: ignore
    _FITZ_AVAILABLE = True
except Exception:
    _FITZ_AVAILABLE = False


# ============================================================
# 数据类定义（解析层本地版，与 db/schemas.py 字段名保持一致）
# ============================================================

@dataclass
class DocumentChunk:
    """统一的文档块数据结构，跨解析/入库/检索层流转。"""
    chunk_id: str
    doc_id: str
    file_name: str
    modality: str                # "text" | "table" | "image" | "mixed"
    content: str                 # 原始内容（文本/表格 markdown/[IMAGE]占位）
    searchable_text: str         # 用于 BM25 + 稠密向量检索的语义文本
    page: int
    section: str = ""            # 所属章节标题（最近一级标题）
    position: dict = field(default_factory=dict)   # {"x":..., "y":..., "w":..., "h":...}
    metadata: dict = field(default_factory=dict)   # {"language": "zh", "heading_level": 2, ...}
    child_images: list[str] = field(default_factory=list)  # 关联的图像 chunk_id
    parent_chunk_id: str = ""
    # 父子双层结构（v2 切块改造新增，向后兼容：旧数据无此字段时默认 child）
    chunk_role: str = "child"
    is_table_only: bool = False
    # 章节结构字段（标准类文档对比路由用，默认空）
    section_id: str = ""        # 层级化编号，如 "5", "5.3", "5.3.2"
    section_path: str = ""      # 完整路径，如 "5.3.2 / BCI Test"
    section_topic: str = ""     # 章节主题（从 SECTION_TOPIC_KEYWORDS 路由表匹配）

    def to_chroma_entity(self):
        """父 chunk 不入向量库；child 返回占位 entity（向量由 ingest 阶段填充）。

        解析层尚未计算 embedding，因此这里对 child 返回一个占位 entity；
        真实写入由 ingest pipeline 在拿到 dense_vector/image_vector 后调用。
        与 db/schemas.DocumentChunk 行为对齐：parent→None, child→(id, vec, meta, content)。
        """
        if self.chunk_role == CHUNK_ROLE_PARENT:
            return None
        # 占位：解析层不持有向量，统一用 0 向量占位（ingest 阶段会用真实向量重建 entity）
        from db.schemas import _DENSE_DIM  # 复用统一向量维度
        meta = {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "modality": self.modality,
            "page": int(self.page),
            "section": self.section or "",
            "chunk_role": self.chunk_role,
            "parent_chunk_id": self.parent_chunk_id or "",
            "is_table_only": bool(self.is_table_only),
            "section_id": self.section_id or "",
            "section_path": self.section_path or "",
            "section_topic": self.section_topic or "",
        }
        return (self.chunk_id, [0.0] * _DENSE_DIM, meta, self.content)


@dataclass
class _RawElement:
    """LiteParse 提取后的中间元素，供分块使用。"""
    type: str            # "heading" | "text" | "table" | "image"
    level: int           # 标题层级 0=正文；H1=1, H2=2, ...
    content: str         # 文本/表格 markdown
    page: int
    position: dict = field(default_factory=dict)
    image_bytes: Optional[bytes] = None
    image_format: str = ""
    image_id: str = ""


# ============================================================
# 图片压缩工具（降低 VL token 消耗）
# ============================================================

def _resize_image_for_vl(image_bytes: bytes, max_pixels: int = 768) -> bytes:
    """将图片缩放到 max_pixels 以内，降低 VL 模型 token 消耗。

    Qwen-VL 按图片分辨率计 token，压缩到 768px 可将 token 消耗降低 50-75%。
    返回 JPEG 格式（更小体积），失败时返回原始 bytes。
    """
    if not _HAS_PIL:
        return image_bytes
    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        if max(w, h) <= max_pixels:
            return image_bytes  # 已经足够小，无需压缩
        # 等比缩放
        ratio = max_pixels / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        # 转 RGB（避免 RGBA/CMYK 问题）
        if img.mode in ("RGBA", "P", "CMYK"):
            img = img.convert("RGB")
        img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return buf.getvalue()
    except Exception:
        return image_bytes


# ============================================================
# 主类
# ============================================================

class DocumentParser:
    """
    文档解析器：将 PDF/DOCX/图片 转换为 DocumentChunk 列表。

    用法：
        parser = DocumentParser()
        chunks = parser.parse("/path/to/file.pdf")
    """

    # 支持的扩展名（白名单），其余类型记录 warning 并返回空列表
    _SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".txt", ".md", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    # 解析结果缓存目录（基于文件哈希跳过重复解析）
    _CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "cache" / "parse_cache"

    # VL 并行调用的最大并发数（避免 API 限流）
    # 默认 1：OneAPI 全局限制极严，2 并发也会触发 429，串行最安全。
    # 可通过环境变量 VL_MAX_WORKERS 覆盖
    _VL_MAX_WORKERS: int = 1

    def __init__(self) -> None:
        # 允许环境变量覆盖并发数（无需改代码就能调优）
        _env_vl_workers = os.getenv("VL_MAX_WORKERS")
        if _env_vl_workers and _env_vl_workers.isdigit() and int(_env_vl_workers) >= 1:
            self._VL_MAX_WORKERS = int(_env_vl_workers)
        # OneAPI 客户端（OpenAI 兼容）。base_url/api_key 来自 config
        self._client: Optional[OpenAI] = None
        if config.ONEAPI_API_KEY:
            try:
                self._client = OpenAI(
                    base_url=config.ONEAPI_BASE_URL,
                    api_key=config.ONEAPI_API_KEY,
                    timeout=config.CHAT_TIMEOUT,  # 避免 VL/Chat 调用无限等待
                    max_retries=0,
                )
            except Exception as e:
                log.warning(f"OneAPI 客户端初始化失败，VL/Chat 调用将降级: {e}")
        else:
            log.warning("ONEAPI_API_KEY 为空，VL/Chat 调用将降级到占位文本")

        # Chat-specific client (table/image enhancement): default = VL gateway;
        # set CHAT_BASE_URL to route generation through a separate gateway (e.g. DeepSeek official).
        self._chat_client: Optional[OpenAI] = None
        if config.CHAT_API_KEY:
            try:
                self._chat_client = OpenAI(
                    base_url=config.CHAT_BASE_URL,
                    api_key=config.CHAT_API_KEY,
                    timeout=config.CHAT_TIMEOUT,
                    max_retries=0,
                )
            except Exception as e:
                log.warning(f"Chat client init failed, enhancement will degrade: {e}")
        if self._chat_client is None:
            self._chat_client = self._client

        # LiteParse 单例：启用 markdown 输出 + 嵌入图片
        # 默认关闭 OCR — 文本型 PDF 已有 native text layer，OCR 反而会在受限环境下报错（os error 5）
        # 可通过 .env 中 LITEPARSE_OCR_ENABLED=true 重新开启（扫描件场景）
        self._liteparse = None
        if _LITEPARSE_AVAILABLE:
            try:
                self._liteparse = _LiteParse(
                    output_format="markdown",
                    image_mode="embed",
                    quiet=True,
                    ocr_enabled=config.LITEPARSE_OCR_ENABLED,
                    ocr_failure_fatal=config.LITEPARSE_OCR_FAILURE_FATAL,
                )
            except Exception as e:
                log.warning(f"LiteParse 初始化失败: {e}")
                self._liteparse = None

    # --------------------------------------------------------
    # 公开入口
    # --------------------------------------------------------

    def parse(self, file_path: str, file_name: str | None = None) -> list[DocumentChunk]:
        """
        解析单个文档，返回 DocumentChunk 列表。

        流程：校验后缀 → 缓存检查 → LiteParse 提取 → 按标题分块 → 视觉/文本增强 → 语种检测。

        Args:
            file_path: 磁盘上的文件路径（上传场景可能是带 task_id 前缀的临时文件）。
            file_name: 入库元数据使用的文件名。默认取 ``file_path`` 的文件名；
                上传场景应传入用户原始文件名，避免随机前缀泄漏到文档列表与删除接口。
        """
        path = Path(file_path)
        if not path.exists():
            log.warning(f"文件不存在: {file_path}")
            return []

        ext = path.suffix.lower()
        if ext not in self._SUPPORTED_EXTS:
            log.warning(f"不支持的文件类型 {ext}，跳过: {file_path}")
            return []

        # 使用局部变量避免并发覆盖（不再用 self.doc_id / self._file_name）
        doc_id = uuid4().hex[:12]
        file_name = file_name or path.name

        # ---- 解析缓存：基于文件哈希跳过重复解析 ----
        file_hash = self._compute_file_hash(str(path))
        if file_hash:
            cached = self._load_cache(file_hash)
            if cached is not None:
                log.info(
                    f"命中解析缓存: {file_path} -> {len(cached)} 个 chunk "
                    f"(hash={file_hash[:8]}...)"
                )
                # 缓存内容复用解析结果，但必须为本次导入分配新身份并重写内部引用，
                # 避免复用旧 doc_id / chunk_id（对齐 intern_rag 的缓存身份恢复）。
                return self._restore_cached_identity(cached, doc_id, file_name)

        log.info(f"开始解析: {file_path} (doc_id={doc_id})")
        t0 = time.time()

        # 1) LiteParse 提取原始元素
        try:
            elements = self._extract_with_liteparse(path)
            if not elements and path.suffix.lower() in {".docx", ".doc"}:
                # 本机未装 LibreOffice 时 LiteParse 无法转换 DOCX，启用纯 Python 兜底
                elements = self._extract_from_docx(path)
        except Exception as e:
            log.error(f"LiteParse 提取失败 {file_path}: {e}")
            return []

        # #region debug-point H2:elements-extracted（已清理）
        if not elements:
            log.warning(f"未提取到任何元素: {file_path}")
            return []

        # 2) 按标题深度分块（传入 doc_id / file_name 作为参数，避免并发竞态）
        chunks = self._build_heading_chunks(elements, doc_id, file_name)

        # 3) 整篇语种检测，写入每个 chunk 的 metadata
        if chunks:
            aggregated = " ".join(c.searchable_text for c in chunks)[:4000]
            lang = detect_language(aggregated)
            for c in chunks:
                c.metadata["language"] = lang
                c.metadata.setdefault("file_ext", ext)

        # ---- 写入解析缓存 ----
        if file_hash and chunks:
            self._save_cache(file_hash, chunks)

        log.info(
            f"解析完成: {file_path} -> {len(chunks)} 个 chunk "
            f"({time.time() - t0:.2f}s)"
        )
        # #region debug-point H3/H4/H5:chunks-summary（已清理）
        return chunks

    # --------------------------------------------------------
    # 解析缓存辅助方法
    # --------------------------------------------------------

    @staticmethod
    def _restore_cached_identity(
        chunks: list[DocumentChunk], doc_id: str, file_name: str,
    ) -> list[DocumentChunk]:
        """为缓存内容分配本次导入的身份，并同步重写内部引用。"""
        id_map = {chunk.chunk_id: uuid4().hex[:12] for chunk in chunks}
        for chunk in chunks:
            chunk.chunk_id = id_map[chunk.chunk_id]
            chunk.doc_id = doc_id
            chunk.file_name = file_name
            if chunk.parent_chunk_id:
                chunk.parent_chunk_id = id_map.get(chunk.parent_chunk_id, "")
            chunk.child_images = [
                id_map[child_id] for child_id in chunk.child_images if child_id in id_map
            ]
        return chunks

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """计算完整文件的 SHA-256 哈希，用于解析缓存去重。"""
        try:
            sha = hashlib.sha256()
            with open(file_path, "rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    sha.update(block)
            return sha.hexdigest()
        except Exception:
            return ""

    @classmethod
    def _load_cache(cls, file_hash: str) -> Optional[list[DocumentChunk]]:
        """从磁盘加载缓存的解析结果。"""
        cache_path = cls._CACHE_DIR / f"{file_hash}.json"
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, encoding="utf-8") as f:
                result = [DocumentChunk(**_cache_restore(item)) for item in json.load(f)]
                # 校验返回值类型，防止缓存损坏导致下游崩溃
                if not isinstance(result, list):
                    log.warning(f"解析缓存类型异常: 期望 list, 实际 {type(result).__name__}")
                    return None
                return result
        except Exception as e:
            log.warning(f"解析缓存读取失败: {e}")
            return None

    @classmethod
    def _save_cache(cls, file_hash: str, chunks: list[DocumentChunk]) -> None:
        """将解析结果持久化到磁盘缓存。"""
        cls._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = cls._CACHE_DIR / f"{file_hash}.json"
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    [_cache_sanitize(asdict(chunk)) for chunk in chunks],
                    f, ensure_ascii=False,
                )
        except Exception as e:
            log.warning(f"解析缓存写入失败: {e}")

    # --------------------------------------------------------
    # LiteParse 提取
    # --------------------------------------------------------

    def _extract_with_liteparse(self, path: Path) -> list[_RawElement]:
        """
        调用 LiteParse 解析文档，转为 _RawElement 列表。

        处理策略：
          - PDF：交给 LiteParse 走原生解析（最优路径）
          - DOCX/DOC：LiteParse 内部会自动尝试转换；失败则走纯文本兜底
          - 图片：直接当单页处理
          - TXT/MD：直接读文本，按段落切分
        """
        if self._liteparse is None and path.suffix.lower() not in {".txt", ".md"}:
            log.warning("LiteParse 未就绪，跳过解析")
            return []

        ext = path.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
            return self._extract_from_image(path)
        if ext in {".txt", ".md"}:
            return self._extract_from_text(path)

        # PDF / DOCX：统一交给 LiteParse
        try:
            result = self._liteparse.parse(str(path.absolute()))
            elements = self._convert_parse_result(result)
        except Exception as e:
            # 关键：避免回退到 _extract_from_text（会把 PDF/DOCX 二进制当 UTF-8 文本读，产生垃圾 chunk）
            # 仅当 LiteParse 失败时记录 warning 并返回空列表，让上层感知失败
            log.warning(
                f"LiteParse 解析失败 {path}: {e}。不进行纯文本回退（避免把二进制文件当文本处理）"
            )
            return []

        # PDF 额外处理：用 PyMuPDF 一次扫描提取矢量图 + 漏掉的光栅图
        if ext == ".pdf" and _FITZ_AVAILABLE:
            try:
                supplement_elements = self._extract_pymupdf_supplements(path, elements)
                elements.extend(supplement_elements)
                elements.sort(key=lambda e: (e.page, e.position.get("y", 0)))
            except Exception as e:
                log.warning(f"PyMuPDF 补充提取失败（非致命）: {e}")

        # PDF 文本兜底：LiteParse 未提取到任何文本/表格时，用 PyMuPDF 页面文本补全
        # （部分 PDF 在 liteparse 2.x 下 markdown/text 为空，但 PyMuPDF 可正常读出文本层）
        if (
            ext == ".pdf"
            and _FITZ_AVAILABLE
            and not any(e.type in ("text", "table") for e in elements)
        ):
            text_elements = self._extract_pymupdf_text_fallback(path)
            if text_elements:
                log.info("PyMuPDF 文本兜底: %s -> %d 页", path.name, len(text_elements))
                elements.extend(text_elements)
                elements.sort(key=lambda e: (e.page, e.position.get("y", 0)))

        return elements

    def _extract_pymupdf_text_fallback(self, path: Path) -> list[_RawElement]:
        """PyMuPDF 页面文本兜底：按页构造 text 元素，供 PDF 文本层提取失败时使用。"""
        if not _FITZ_AVAILABLE:
            return []
        elements: list[_RawElement] = []
        try:
            doc = _fitz.open(str(path))
            for i, page in enumerate(doc):
                text = page.get_text().strip()
                if text:
                    elements.append(
                        _RawElement(type="text", level=0, content=text, page=i + 1)
                    )
        except Exception as e:  # noqa: BLE001
            log.warning(f"PyMuPDF 文本兜底失败 {path}: {e}")
            return []
        return elements

    def _extract_from_text(self, path: Path) -> list[_RawElement]:
        """读取纯文本/Markdown，按行构造 text 元素；遇到空行分段。"""
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            log.error(f"读取文本失败 {path}: {e}")
            return []
        if not text.strip():
            return []
        return [
            _RawElement(
                type="text",
                level=0,
                content=line.strip(),
                page=1,
            )
            for line in text.splitlines()
            if line.strip()
        ]

    def _extract_from_docx(self, path: Path) -> list[_RawElement]:
        """DOCX 纯 Python 兜底（本机无 LibreOffice 时启用）：
        按文档顺序提取段落与表格，识别标题层级，page 统一为 1。"""
        try:
            from docx import Document as _DocxDocument
            from docx.oxml.ns import qn as _qn
            from docx.table import Table as _DocxTable
            from docx.text.paragraph import Paragraph as _DocxParagraph
        except Exception as e:
            log.warning(f"python-docx 不可用，无法解析 DOCX: {e}")
            return []

        elements: list[_RawElement] = []

        def _level(p: _DocxParagraph) -> int:
            style_name = (p.style.name or "") if p.style else ""
            m = re.search(r"(?:heading|标题)\s*(\d+)", style_name, re.IGNORECASE)
            return int(m.group(1)) if m else 0

        try:
            doc = _DocxDocument(str(path))
            for child in doc.element.body.iterchildren():
                if child.tag == _qn("w:p"):
                    para = _DocxParagraph(child, doc)
                    text = para.text.strip()
                    if text:
                        elements.append(_RawElement(
                            type="text", level=_level(para), content=text, page=1,
                        ))
                elif child.tag == _qn("w:tbl"):
                    table = _DocxTable(child, doc)
                    rows = []
                    for row in table.rows:
                        cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                        rows.append(" | ".join(cells))
                    if rows:
                        elements.append(_RawElement(
                            type="table", level=0, content="\n".join(rows), page=1,
                        ))
        except Exception as e:
            log.warning(f"python-docx 提取失败 {path}: {e}")
            return []

        if not elements:
            log.warning(f"DOCX 未提取到内容: {path}")
            return []
        log.info(f"DOCX 兜底解析成功: {path} -> {len(elements)} 个元素")
        return elements

    def _extract_from_image(self, path: Path) -> list[_RawElement]:
        """处理单张图片：读 bytes，构造一个 image 元素（不做文本提取）。"""
        try:
            data = path.read_bytes()
        except Exception as e:
            log.error(f"读取图片失败 {path}: {e}")
            return []
        return [
            _RawElement(
                type="image",
                level=0,
                content="",
                page=1,
                image_bytes=data,
                image_format=path.suffix.lstrip(".").lower() or "png",
                image_id=uuid4().hex[:8],
            )
        ]

    def _convert_parse_result(self, result: Any) -> list[_RawElement]:
        """
        将 LiteParse 的 ParseResult 转为 _RawElement 列表。

        核心策略：逐页处理 markdown 行，按 `#` 数量识别标题层级；
        图片引用（![](id)）单独抽出。
        """
        elements: list[_RawElement] = []

        # 建立 image_id -> (bytes, format) 索引，便于从 markdown 中反查
        image_index: dict[str, tuple[bytes, str]] = {}
        for img in getattr(result, "images", []) or []:
            try:
                image_index[img.id] = (img.bytes, img.format)
            except Exception:
                continue

        for page in result.pages:
            md = getattr(page, "markdown", "") or page.text or ""
            page_num = page.page_num
            if not md.strip():
                continue

            page_elements = self._parse_markdown_page(
                md, page_num, image_index, page
            )
            elements.extend(page_elements)

        return elements

    def _parse_markdown_page(
        self,
        md: str,
        page_num: int,
        image_index: dict[str, tuple[bytes, str]],
        page_obj: Any,
    ) -> list[_RawElement]:
        """解析单页 markdown，产出 _RawElement 列表。"""
        out: list[_RawElement] = []
        # 表格缓冲区（LiteParse 输出的 markdown 表格是连续行）
        table_buf: list[str] = []

        def flush_table() -> None:
            if table_buf:
                out.append(_RawElement(
                    type="table",
                    level=0,
                    content="\n".join(table_buf),
                    page=page_num,
                ))
                table_buf.clear()

        for raw_line in md.splitlines():
            line = raw_line.rstrip()
            stripped = line.lstrip()

            # ---- 表格行：含 | 且不是代码块 ----
            if stripped.startswith("|") and stripped.endswith("|"):
                table_buf.append(line)
                continue
            else:
                flush_table()

            # ---- 图片引用 ![](id) ----
            m = re.match(r"!\[.*?\]\(([^)]+)\)", stripped)
            if m:
                ref = m.group(1)
                img_bytes_fmt = image_index.get(ref)
                # LiteParse 的 markdown 引用格式为 ![](image_{id}.{fmt})，
                # 而 image_index 的 key 是原始 id（如 p3_0），需要做格式兼容
                if not img_bytes_fmt:
                    # 尝试去掉 "image_" 前缀和扩展名（如 image_p3_0.png → p3_0）
                    clean = ref.replace("image_", "").rsplit(".", 1)[0]
                    img_bytes_fmt = image_index.get(clean)
                if img_bytes_fmt:
                    bts, fmt = img_bytes_fmt
                    # 过滤装饰性小图标（如信封/logo，通常 < 2KB）
                    if len(bts) < 2000:
                        continue
                    out.append(_RawElement(
                        type="image",
                        level=0,
                        content="",
                        page=page_num,
                        image_bytes=bts,
                        image_format=fmt or "png",
                        image_id=ref,
                    ))
                continue

            # ---- 标题 # / ## / ### ... ----
            heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading:
                level = len(heading.group(1))
                title = heading.group(2).strip()
                if title:
                    out.append(_RawElement(
                        type="heading",
                        level=level,
                        content=title,
                        page=page_num,
                    ))
                continue

            # ---- 普通文本行（非空才保留） ----
            if stripped:
                out.append(_RawElement(
                    type="text",
                    level=0,
                    content=stripped,
                    page=page_num,
                ))

        flush_table()
        return out

    # --------------------------------------------------------
    # PyMuPDF 补充提取（矢量图 + 漏掉的光栅图，一次 PDF 扫描）
    # --------------------------------------------------------

    # 线条图（stroke）聚类的最小面积阈值（像素²）
    _VECTOR_MIN_AREA: int = 500

    # 线条图聚类的合并距离（点）
    _VECTOR_MERGE_GAP: float = 300.0

    # 宽高比阈值：超过此值的视为装饰性线条
    _VECTOR_MAX_ASPECT_RATIO: float = 15.0

    # 光栅图最小面积阈值（像素²）
    _RASTER_MIN_AREA: int = 20000

    # 矢量图渲染时，附近文字块的扩展距离（点）
    _VECTOR_TEXT_EXPAND: float = 60.0

    def _extract_pymupdf_supplements(
        self, path: Path, existing_elements: list[_RawElement]
    ) -> list[_RawElement]:
        """一次打开 PDF，同时提取矢量图形和 LiteParse 漏掉的光栅图。

        合并了原 _extract_vector_drawings 和 _extract_raster_images_pymupdf，
        避免重复打开 PDF 文件，减少 I/O 开销。
        """
        elements: list[_RawElement] = []
        doc = _fitz.open(str(path.absolute()))
        try:
            # 收集已有图片 bytes 用于光栅图去重（LiteParse 图片无 position 信息）
            existing_bytes: set[bytes] = set()
            for el in existing_elements:
                if el.type == "image" and el.image_bytes:
                    existing_bytes.add(el.image_bytes)

            for i in range(doc.page_count):
                pn = i + 1
                page = doc[i]

                # ---- 矢量图形提取 ----
                drawings = page.get_drawings()
                if drawings:
                    strokes = [d for d in drawings if d.get("fill") is None]
                    if strokes:
                        clusters = self._cluster_drawings(strokes)
                        if clusters:
                            text_blocks = page.get_text("dict")["blocks"]
                            text_rects = [
                                _fitz.Rect(b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3])
                                for b in text_blocks if b["type"] == 0
                            ]
                            for j, cluster in enumerate(clusters):
                                bbox = cluster["bbox"]
                                margin = 4.0
                                clip = _fitz.Rect(
                                    max(0, bbox[0] - margin),
                                    max(0, bbox[1] - margin),
                                    min(page.rect.width, bbox[2] + margin),
                                    min(page.rect.height, bbox[3] + margin),
                                )
                                # 扩展 clip 包含下方连续文字块
                                below_texts = [
                                    tr for tr in text_rects
                                    if abs(tr.x0 - clip.x0) < self._VECTOR_TEXT_EXPAND * 2
                                    and tr.y0 > clip.y1 - margin
                                ]
                                below_texts.sort(key=lambda tr: tr.y0)
                                prev_bottom = clip.y1
                                for tr in below_texts:
                                    if tr.y0 - prev_bottom < self._VECTOR_TEXT_EXPAND:
                                        clip.y1 = max(clip.y1, tr.y1 + margin)
                                        clip.x0 = min(clip.x0, tr.x0 - margin)
                                        clip.x1 = max(clip.x1, tr.x1 + margin)
                                        prev_bottom = clip.y1
                                    else:
                                        break
                                pix = page.get_pixmap(clip=clip, matrix=_fitz.Matrix(2, 2))
                                img_bytes = pix.tobytes("png")
                                elements.append(_RawElement(
                                    type="image", level=0, content="", page=pn,
                                    image_bytes=img_bytes, image_format="png",
                                    image_id=f"vector_p{pn}_{j}",
                                    position={
                                        "x": clip.x0, "y": clip.y0,
                                        "w": clip.width, "h": clip.height,
                                    },
                                ))

                # ---- 光栅图补充提取 ----
                blocks = page.get_text("dict")["blocks"]
                img_blocks = [b for b in blocks if b["type"] == 1]
                if img_blocks:
                    xref_map: dict[tuple[int, int], int] = {}
                    for img in page.get_images():
                        xref_map[(img[2], img[3])] = img[0]
                    for b in img_blocks:
                        w, h = int(b["width"]), int(b["height"])
                        area = w * h
                        if area < self._RASTER_MIN_AREA:
                            continue
                        xref = xref_map.get((w, h))
                        if xref is None:
                            continue
                        try:
                            base = doc.extract_image(xref)
                            img_bytes = base["image"]
                            img_format = base["ext"]
                        except Exception:
                            continue
                        if img_bytes in existing_bytes:
                            continue
                        existing_bytes.add(img_bytes)
                        x0, y0, x1, y1 = b["bbox"]
                        elements.append(_RawElement(
                            type="image", level=0, content="", page=pn,
                            image_bytes=img_bytes, image_format=img_format or "png",
                            image_id=f"raster_p{pn}_{len(elements)}",
                            position={"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
                        ))
        finally:
            doc.close()
        return elements

    def _cluster_drawings(
        self, drawings: list[dict]
    ) -> list[dict]:
        """
        锚点聚类：以大面积线条为锚点，附近小线条自动纳入。

        算法：
          1. 按面积降序排列，面积 > _VECTOR_MIN_AREA 的为锚点
          2. 对每个锚点，检查是否与已有聚类重叠/相邻
          3. 若重叠，合并；否则新建聚类
          4. 第二轮：将小线条（面积 <= _VECTOR_MIN_AREA）纳入最近的聚类
             — 解决 Page 6 封装图顶部细线、Page 7 虚线被过滤的问题

        返回: [{"bbox": (x0, y0, x1, y1), "count": N}, ...]
        """
        if not drawings:
            return []

        # 过滤极窄线条
        valid = []
        for d in drawings:
            r = d["rect"]
            w, h = r.width, r.height
            if w > 0 and h > 0:
                ratio = max(w, h) / min(w, h)
                if ratio > self._VECTOR_MAX_ASPECT_RATIO:
                    continue
            valid.append(d)

        if not valid:
            return []

        # 按面积降序排列
        valid.sort(key=lambda d: d["rect"].width * d["rect"].height, reverse=True)

        # 第一轮：大线条（锚点）聚类
        clusters: list[dict] = []
        for d in valid:
            area = d["rect"].width * d["rect"].height
            if area < self._VECTOR_MIN_AREA:
                break  # 后面的更小，全部留给第二轮

            r = d["rect"]
            dx0, dy0, dx1, dy1 = r.x0, r.y0, r.x1, r.y1
            merged = False
            for c in clusters:
                cx0, cy0, cx1, cy1 = c["bbox"]
                if (
                    dx0 - self._VECTOR_MERGE_GAP < cx1
                    and dx1 + self._VECTOR_MERGE_GAP > cx0
                    and dy0 - self._VECTOR_MERGE_GAP < cy1
                    and dy1 + self._VECTOR_MERGE_GAP > cy0
                ):
                    c["bbox"] = (
                        min(cx0, dx0), min(cy0, dy0),
                        max(cx1, dx1), max(cy1, dy1),
                    )
                    c["count"] += 1
                    merged = True
                    break
            if not merged:
                clusters.append({"bbox": (dx0, dy0, dx1, dy1), "count": 1})

        # 第二轮：小线条纳入最近的聚类
        for d in valid:
            area = d["rect"].width * d["rect"].height
            if area >= self._VECTOR_MIN_AREA:
                continue  # 已在第一轮处理

            r = d["rect"]
            dx0, dy0, dx1, dy1 = r.x0, r.y0, r.x1, r.y1
            merged = False
            for c in clusters:
                cx0, cy0, cx1, cy1 = c["bbox"]
                if (
                    dx0 - self._VECTOR_MERGE_GAP < cx1
                    and dx1 + self._VECTOR_MERGE_GAP > cx0
                    and dy0 - self._VECTOR_MERGE_GAP < cy1
                    and dy1 + self._VECTOR_MERGE_GAP > cy0
                ):
                    c["bbox"] = (
                        min(cx0, dx0), min(cy0, dy0),
                        max(cx1, dx1), max(cy1, dy1),
                    )
                    c["count"] += 1
                    merged = True
                    break
            # 小线条不创建新聚类（避免孤立噪声）

        return clusters

    # --------------------------------------------------------
    # 标题分块（heading-aware + parent-child）
    # --------------------------------------------------------

    @dataclass
    class _SectionGroup:
        """按 heading 分组后的 Section 单元（跨页同标题合并后）。"""
        section: str           # 最近一级标题
        level: int             # 标题层级
        start_page: int        # 起始页
        end_page: int          # 结束页
        elements: list         # 包含的 _RawElement 列表
        # 章节结构识别结果（默认空，向后兼容；由 _build_heading_chunks 填充）
        section_id: str = ""        # 层级化编号，如 "5", "5.3", "5.3.2"
        section_path: str = ""      # 完整路径，如 "5.3.2 / BCI Test"
        section_topic: str = ""     # 章节主题（从 SECTION_TOPIC_KEYWORDS 路由表匹配）

    def _group_by_heading(
        self, elements: list, doc_id: str, file_name: str,
    ) -> list["DocumentParser._SectionGroup"]:
        """按 heading 深度分组 elements，跨页同标题合并为同一 Section。

        规则：
          - 遇到 H1 (level==1) → 开启新 Section
          - 遇到 H2+ (level>=2) → 累积到当前 Section（作为子节标记）
          - 跨页同 H1 标题 → 合并为同一 Section
        """
        if not config.ENABLE_HEADING_MERGE:
            # 不合并：每个 heading 都按 level 独立 Section
            groups: list[DocumentParser._SectionGroup] = []
            for el in elements:
                if el.type == "heading" and el.level == 1:
                    groups.append(DocumentParser._SectionGroup(
                        section=el.content, level=el.level,
                        start_page=el.page, end_page=el.page,
                        elements=[el],
                    ))
                else:
                    if groups:
                        groups[-1].elements.append(el)
                        groups[-1].end_page = el.page
            return groups

        # 跨页合并模式：H1 切分；H2+ 累积
        groups = []
        current: Optional[DocumentParser._SectionGroup] = None
        for el in elements:
            if el.type == "heading" and el.level == 1:
                if current and current.section == el.content:
                    # 跨页同 H1 标题：合并
                    current.elements.append(el)
                    current.end_page = el.page
                else:
                    # 新 H1 Section
                    current = DocumentParser._SectionGroup(
                        section=el.content, level=el.level,
                        start_page=el.page, end_page=el.page,
                        elements=[el],
                    )
                    groups.append(current)
            else:
                # H2+ heading 或 text/table/image：累积到当前 group
                if current is None:
                    # 文档开头没有 H1 的情况：建一个虚拟 group
                    current = DocumentParser._SectionGroup(
                        section="", level=0,
                        start_page=el.page, end_page=el.page,
                        elements=[],
                    )
                    groups.append(current)
                current.elements.append(el)
                current.end_page = el.page
        return groups

    # ---------- 章节结构识别 ----------

    @staticmethod
    def _chinese_to_int(s: str) -> int:
        """中文章节数字转阿拉伯数字（支持 一-九 + 十/百，最大 999）。

        示例："五"→5, "十"→10, "十二"→12, "二十三"→23, "一百二十三"→123。
        仅支持 Task 1 范围内需要识别的中文章节格式。
        """
        digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        if not s:
            return 0
        total = 0
        if "百" in s:
            idx = s.index("百")
            hundreds_char = s[:idx]
            hundreds = digits.get(hundreds_char, 1) if hundreds_char else 1
            rest = s[idx + 1:]
            tens = 0
            ones = 0
            if "十" in rest:
                jdx = rest.index("十")
                tens_char = rest[:jdx]
                tens = digits.get(tens_char, 1) if tens_char else 1
                ones_char = rest[jdx + 1:]
                ones = digits.get(ones_char, 0) if ones_char else 0
            total = hundreds * 100 + tens * 10 + ones
            return total
        if "十" in s:
            idx = s.index("十")
            if idx == 0:
                # "十X" 形式 → 10 + ones
                ones_char = s[idx + 1:]
                ones = digits.get(ones_char, 0) if ones_char else 0
                return 10 + ones
            # "X十" / "X十Y" 形式
            tens_char = s[:idx]
            ones_char = s[idx + 1:]
            tens = digits.get(tens_char, 0)
            ones = digits.get(ones_char, 0) if ones_char else 0
            return tens * 10 + ones
        return digits.get(s, 0)

    def _detect_section_path(
        self, heading_text: str, current_stack: list[str]
    ) -> tuple[str, str, str]:
        """从 heading 文本中识别章节编号 / 路径 / 主题。

        识别规则（按优先级）：
            1) 数字编号前缀："5.3.2 BCI Test" → ("5.3.2", "5.3.2 / BCI Test", topic)
            2) "Chapter 5 Test Methods" → ("5", ...)
            3) "Section 5.3" → ("5.3", ...)
            4) 中文章节："第五章 测试方法" → ("5", ...)
            5) 顿号分隔："5.3. 测试方法" → ("5.3", ...)

        Args:
            heading_text: 标题原文（已 strip）
            current_stack: 当前 heading 栈（暂未用于父章节推断，预留扩展）

        Returns:
            (section_id, section_path, section_topic) 元组。
            任何字段无法识别时返回空字符串。
            当 config.ENABLE_SECTION_RECOGNITION=False 时一律返回 ("", "", "")。
        """
        # 总开关关闭时，行为等价于修改前（旧代码不会写入 section_* 字段）
        if not getattr(config, "ENABLE_SECTION_RECOGNITION", True):
            return ("", "", "")

        if not heading_text:
            return ("", "", "")

        text = heading_text.strip()
        section_id = ""
        title = ""

        # 规则 1: 数字编号前缀 "5.3.2 BCI Test Method"
        m = re.match(r"^(\d+(?:\.\d+){0,4})\s+(.+)$", text)
        if m:
            section_id = m.group(1)
            title = m.group(2).strip()
        else:
            # 规则 2: Chapter 5 [Title] / Chapter 5: Title
            m = re.match(r"^Chapter\s+(\d+)\s*[:\-\.]?\s*(.*)$", text, re.IGNORECASE)
            if m:
                section_id = m.group(1)
                title = m.group(2).strip()
            else:
                # 规则 3: Section 5.3
                m = re.match(r"^Section\s+(\d+(?:\.\d+){0,4})\s*[:\-\.]?\s*(.*)$", text, re.IGNORECASE)
                if m:
                    section_id = m.group(1)
                    title = m.group(2).strip()
                else:
                    # 规则 4: 中文章节 "第五章 测试方法"
                    m = re.match(r"^第([一二三四五六七八九十百]+)章\s*(.*)$", text)
                    if m:
                        section_id = str(self._chinese_to_int(m.group(1)))
                        title = m.group(2).strip()
                    else:
                        # 规则 5: 顿号/句号分隔 "5.3. 测试方法"
                        m = re.match(r"^(\d+(?:\.\d+){0,4})\s*[\.、]\s*(.*)$", text)
                        if m:
                            section_id = m.group(1)
                            title = m.group(2).strip()
                        else:
                            # 无法识别编号：返回空
                            return ("", "", "")

        # section_topic 路由：从识别出的标题匹配 config.SECTION_TOPIC_KEYWORDS
        section_topic = ""
        if title:
            kw_map = getattr(config, "SECTION_TOPIC_KEYWORDS", {}) or {}
            if isinstance(kw_map, dict) and kw_map:
                title_lower = title.lower()
                # 找到第一个命中的 topic（按 dict 声明顺序，py3.7+ 保持插入序）
                for topic, keywords in kw_map.items():
                    if not keywords:
                        continue
                    for kw in keywords:
                        if not kw:
                            continue
                        # 关键词匹配：中文精确子串，英文/数字按大小写不敏感子串
                        if kw.lower() in title_lower:
                            section_topic = topic
                            break
                    if section_topic:
                        break

        # section_path 构造："section_id / 标题"；标题为空时只保留 section_id
        if section_id and title:
            section_path = f"{section_id} / {title}"
        elif section_id:
            section_path = section_id
        else:
            section_path = ""

        return (section_id, section_path, section_topic)

    # ---------- _build_parent_chunk 的辅助方法 ----------

    def _collect_section_parts(
        self,
        group: "_SectionGroup",
        file_name: str,
        image_descriptions: dict[str, str],
        doc_id: str,
    ) -> tuple[list[str], list[str], str, int, list[DocumentChunk], list[str]]:
        """阶段1：遍历 Section 元素，收集 text/table/image 三类内容。

        Returns:
            (text_parts, table_parts, cur_section, cur_page, image_chunks, image_chunk_ids)
        """
        text_parts: list[str] = []
        table_parts: list[str] = []
        image_chunks: list[DocumentChunk] = []
        image_chunk_ids: list[str] = []
        cur_section: str = group.section
        cur_page: int = group.start_page
        sub_heading_lines: list[str] = []

        for el in group.elements:
            if el.type == "heading":
                if el.level > config.CHUNK_HEADING_MIN_LEVEL:
                    sub_heading_lines.append(f"{'#' * el.level} {el.content}")
                continue

            if el.type == "image":
                if not el.image_bytes:
                    continue
                img_chunk, img_chunk_id = self._build_image_child(
                    el, file_name, cur_section, text_parts, doc_id, image_descriptions,
                )
                image_chunks.append(img_chunk)
                image_chunk_ids.append(img_chunk_id)
                continue

            if el.type == "table":
                table_parts.append(el.content)
                cur_page = el.page
                continue

            if el.type == "text":
                text_parts.append(el.content)
                cur_page = el.page

        if sub_heading_lines:
            text_parts = sub_heading_lines + text_parts

        return text_parts, table_parts, cur_section, cur_page, image_chunks, image_chunk_ids

    def _build_image_child(
        self,
        el: "_RawElement",
        file_name: str,
        cur_section: str,
        text_parts: list[str],
        doc_id: str,
        image_descriptions: dict[str, str],
    ) -> tuple[DocumentChunk, str]:
        """构建单个图片 child chunk，含 VL 描述生成。"""
        img_desc = image_descriptions.get(el.image_id)
        if img_desc is None:
            img_desc = self._vision_understand(
                el.image_bytes, context_text=f"{file_name} | {cur_section}", image_format=el.image_format,
            ) or ""
        if el.image_id not in (getattr(self, "_annotated_image_ids", None) or ()):
            img_desc = self._annotate_image(img_desc, context_text=cur_section)

        img_chunk_id = uuid4().hex[:12]
        fallback = f"[IMAGE at page {el.page}]"
        img_content = f"[图片内容] {img_desc if img_desc else fallback}"
        parent_text_preview = " ".join(text_parts)[:200] if text_parts else ""
        img_searchable = (
            f"{file_name} p.{el.page} | {cur_section} | {parent_text_preview} | "
            f"{img_desc if img_desc else fallback}"
        )
        img_chunk = DocumentChunk(
            chunk_id=img_chunk_id,
            doc_id=doc_id, file_name=file_name,
            modality="image", content=img_content,
            searchable_text=img_searchable, page=el.page,
            section=cur_section, position=dict(el.position),
            metadata={
                "image_format": el.image_format, "image_id": el.image_id,
                "image_bytes": el.image_bytes,
                "image_bytes_len": len(el.image_bytes) if el.image_bytes else 0,
            },
            child_images=[],
            chunk_role=CHUNK_ROLE_CHILD,
        )
        return img_chunk, img_chunk_id

    def _build_parent_content(
        self,
        group: "_SectionGroup",
        text_parts: list[str],
        table_parts: list[str],
        has_images: bool,
        image_chunk_ids: list[str],
        doc_id: str,
        file_name: str,
    ) -> DocumentChunk:
        """阶段2：构造 parent chunk（section 标题 + 章节摘要）。"""
        parent_id = uuid4().hex[:12]
        parent_section = f"{'#' * group.level} {group.section}".strip() if group.section else ""
        text_summary = " ".join(text_parts)[:400] if text_parts else ""
        table_summary = table_parts[0][:200] if table_parts else ""

        parent_content_parts = []
        if parent_section:
            parent_content_parts.append(parent_section)
        if text_summary:
            parent_content_parts.append(text_summary)
        if table_summary:
            parent_content_parts.append(f"[TABLE_SUMMARY]\n{table_summary}")
        parent_content = "\n\n".join(parent_content_parts).strip()
        if count_tokens(parent_content) > config.PARENT_MAX_TOKENS:
            parent_content = split_by_tokens(parent_content, config.PARENT_MAX_TOKENS)[0]

        return DocumentChunk(
            chunk_id=parent_id,
            doc_id=doc_id, file_name=file_name,
            modality="mixed" if (text_parts or table_parts or has_images) else "text",
            content=parent_content,
            searchable_text=parent_content,
            page=group.start_page,
            section=group.section,
            position={},
            metadata={"heading_level": group.level, "end_page": group.end_page},
            child_images=list(image_chunk_ids),
            chunk_role=CHUNK_ROLE_PARENT,
        )

    def _build_table_child_chunk(
        self,
        table_parts: list[str],
        text_parts: list[str],
        cur_section: str,
        cur_page: int,
        parent_id: str,
        doc_id: str,
        file_name: str,
        heading_level: int,
        markdown_table: str = "",  # 原始表格 Markdown，用于 ENABLE_TABLE_FULL_CONTENT=True 时保留
    ) -> list[DocumentChunk]:
        """阶段3：表格独立成 child chunk。"""
        if not table_parts or not config.ENABLE_TABLE_ISOLATION:
            return []
        table_raw = "\n\n".join(table_parts).strip()
        text_for_anno = " ".join(text_parts)[:300] if text_parts else ""
        full_for_annotation = "\n\n".join([p for p in [text_for_anno, table_raw] if p])
        table_annotation = self._annotate_table(full_for_annotation)

        # 根据 ENABLE_TABLE_FULL_CONTENT 决定 content 格式
        # True：保留原始表格 Markdown + LLM 摘要，提升检索时对表格结构的可见性
        if config.ENABLE_TABLE_FULL_CONTENT and markdown_table:
            content_parts = []
            content_parts.append(f"[原始表格]\n{markdown_table}")
            if table_annotation:
                content_parts.append(f"[表格摘要]\n{table_annotation}")
            table_content = "\n\n".join(content_parts)
        else:
            # 原有行为：仅 LLM 摘要 + 原始表格
            table_content = ("[TABLE_PARAMS]\n" + table_annotation + "\n\n") if table_annotation else ""
            table_content += table_raw

        # searchable_text 始终保留完整表格内容 + 摘要，确保检索覆盖
        table_searchable = (table_annotation + "\n") if table_annotation else ""
        table_searchable += table_raw

        # 使用 TABLE_CHUNK_MAX_TOKENS 把厚表格内容切分为多个 chunk。
        # 原实现把所有表格合并成一个超大 chunk（可达 3 万字符），单个向量被截断且
        # 无法精确命中具体试验项，严重影响跨标准逐项对比。这里按表格块/行切分，
        # 每个 chunk 不超过 TABLE_CHUNK_MAX_TOKENS。
        table_fragments = split_by_tokens(table_raw, config.TABLE_CHUNK_MAX_TOKENS)
        # split_by_tokens 按英文约 4 字符/token 估算时，3000 token 可能产生
        # ~12000 字符的英文表格块；这里按字符硬上限再切，避免单个向量被截断。
        if config.TABLE_CHUNK_MAX_CHARS > 0:
            _max_chars = config.TABLE_CHUNK_MAX_CHARS
            _extra: list[str] = []
            for _frag in table_fragments:
                if len(_frag) <= _max_chars:
                    _extra.append(_frag)
                    continue
                # 优先按行切分（保留表格行边界），再按字符硬切
                _lines = _frag.split("\n")
                _buf: list[str] = []
                _buf_len = 0
                for _line in _lines:
                    _line_len = len(_line)
                    if _buf and _buf_len + _line_len + 1 > _max_chars:
                        _extra.append("\n".join(_buf))
                        _buf = []
                        _buf_len = 0
                    if _line_len > _max_chars:
                        # 单行超长（如超大表格整行），按字符硬切，避免残留超长块
                        for _start in range(0, _line_len, _max_chars):
                            _extra.append(_line[_start:_start + _max_chars])
                        continue
                    _buf.append(_line)
                    _buf_len += _line_len + 1
                if _buf:
                    _extra.append("\n".join(_buf))
            table_fragments = _extra
        if not table_fragments:
            return []

        chunks: list[DocumentChunk] = []
        for idx, frag in enumerate(table_fragments):
            # 摘要仅附加在第一个 chunk 上，防止成本重复
            if idx == 0 and table_annotation:
                frag_content = f"[原始表格]\n{frag}\n\n[表格摘要]\n{table_annotation}"
                frag_searchable = table_annotation + "\n" + frag
            else:
                frag_content = f"[原始表格]\n{frag}"
                frag_searchable = frag
            chunks.append(DocumentChunk(
                chunk_id=uuid4().hex[:12],
                doc_id=doc_id, file_name=file_name,
                modality=MODALITY_TABLE,
                content=frag_content,
                searchable_text=frag_searchable,
                page=cur_page, section=cur_section,
                position={},
                metadata={
                    "heading_level": heading_level,
                    "is_table": True,  # 标记为表格 chunk
                    "max_tokens_source": "TABLE_CHUNK_MAX_TOKENS",
                    "table_part": idx + 1,
                    "table_part_total": len(table_fragments),
                },
                child_images=[], parent_chunk_id=parent_id,
                chunk_role=CHUNK_ROLE_CHILD, is_table_only=True,
            ))
        return chunks

    @staticmethod
    def _is_emc_section(section_topic: str) -> bool:
        """判断 section_topic 是否属于 EMC 相关章节（emission/immunity/bci）。

        命中时文本 chunk 使用更大的 EMC_CHUNK_MAX_TOKENS，避免参数密集的 EMC 文档被过度切碎。
        """
        if not section_topic:
            return False
        _emc_keywords = ("emission", "immunity", "bci")
        return any(kw in section_topic.lower() for kw in _emc_keywords)

    def _build_text_child_chunks(
        self,
        text_parts: list[str],
        cur_section: str,
        cur_page: int,
        parent_id: str,
        doc_id: str,
        file_name: str,
        heading_level: int,
        section_topic: str = "",  # 章节主题，用于 EMC 文档判定
    ) -> list[DocumentChunk]:
        """阶段4：文本按 token 切分为若干 child chunk。"""
        if not text_parts:
            return []
        text_full = "\n".join(text_parts).strip()
        # EMC 文档使用专用 token 上限，避免参数密集的章节被过度切碎
        _max_tokens = config.EMC_CHUNK_MAX_TOKENS if self._is_emc_section(section_topic) else config.CHUNK_MAX_TOKENS
        text_fragments = split_by_tokens(text_full, _max_tokens)
        return [
            DocumentChunk(
                chunk_id=uuid4().hex[:12],
                doc_id=doc_id, file_name=file_name,
                modality="text", content=frag, searchable_text=frag,
                page=cur_page, section=cur_section,
                position={}, metadata={"heading_level": heading_level},
                child_images=[], parent_chunk_id=parent_id,
                chunk_role=CHUNK_ROLE_CHILD,
            )
            for frag in text_fragments
        ]

    def _build_parent_chunk(
        self,
        group: "_SectionGroup",
        doc_id: str,
        file_name: str,
        image_descriptions: dict[str, str],
    ) -> tuple[DocumentChunk, list[DocumentChunk]]:
        """为一个 Section 构建 parent chunk + 关联的 child chunks。

        Returns:
            (parent_chunk, [table_child, ..., text_child, ..., image_child, ...])
            parent_chunk 自身不入向量库（child 入库；parent 仅做上下文锚点）
        """
        # 阶段1：收集三类内容
        text_parts, table_parts, cur_section, cur_page, image_chunks, image_chunk_ids = (
            self._collect_section_parts(group, file_name, image_descriptions, doc_id)
        )

        # 阶段2：构造 parent chunk
        parent_chunk = self._build_parent_content(
            group, text_parts, table_parts, bool(image_chunks),
            image_chunk_ids, doc_id, file_name,
        )

        # 阶段3：表格独立成 child chunk
        # 预先计算 markdown_table（原始表格 Markdown），传入 _build_table_child_chunk
        _table_raw = "\n\n".join(table_parts).strip() if table_parts else ""
        table_chunks = self._build_table_child_chunk(
            table_parts, text_parts, cur_section, cur_page,
            parent_chunk.chunk_id, doc_id, file_name, group.level,
            markdown_table=_table_raw,  # 传入原始表格 Markdown，供 ENABLE_TABLE_FULL_CONTENT 使用
        )

        # 阶段4：文本按 token 切分为 child chunk（传入 section_topic 用于 EMC 关键词检测）
        text_chunks = self._build_text_child_chunks(
            text_parts, cur_section, cur_page,
            parent_chunk.chunk_id, doc_id, file_name, group.level,
            section_topic=group.section_topic,
        )

        # 阶段5：image chunk 挂回 parent
        for img_chunk in image_chunks:
            img_chunk.parent_chunk_id = parent_chunk.chunk_id

        # ---- 章节结构字段透传（Task 1 新增） ----
        # 从 group 上读取已识别的章节信息，赋给所有由该 group 产出的 chunk
        # （包含 parent / table / text / image 四类）
        for _c in (parent_chunk, *table_chunks, *text_chunks, *image_chunks):
            _c.section_id = group.section_id
            _c.section_path = group.section_path
            _c.section_topic = group.section_topic

        return parent_chunk, text_chunks + table_chunks + image_chunks

    class _ChunkBuilder:
        """封装分块构建过程中的可变状态，避免闭包 + nonlocal 的复杂度。

        将原来 10 个 nonlocal 变量收拢到一个类中，flush() 从嵌套函数
        变为普通方法，可读性和可维护性均得到提升。
        """

        def __init__(self, doc_id: str, file_name: str, parser: "DocumentParser") -> None:
            self.doc_id = doc_id
            self.file_name = file_name
            self._parser = parser  # 用于委托 _annotate_table 调用
            self.chunks: list[DocumentChunk] = []
            # 当前 chunk 的构建状态
            self.current_id: Optional[str] = None
            self.section: str = ""
            self.level: int = 0
            self.page: int = 1
            self.text_parts: list[str] = []
            self.table_parts: list[str] = []
            self.position: dict = {}
            self.modality: set[str] = set()
            self.child_images: list[str] = []
            self.parent_id: str = ""
            # 章节结构识别结果（Task 1 新增；start_new 时刷新）
            self.section_id: str = ""
            self.section_path: str = ""
            self.section_topic: str = ""

        def start_new(self, section: str, level: int, page: int, position: dict = None) -> None:
            """开启一个新的 chunk 构建。"""
            self.current_id = uuid4().hex[:12]
            self.section = section
            self.level = level
            self.page = page
            self.position = dict(position) if position else {}
            # 章节结构识别：在 start_new 时调用，结果缓存在 builder 上
            self.section_id, self.section_path, self.section_topic = (
                self._parser._detect_section_path(section, [])
            )

        def flush(self) -> Optional[DocumentChunk]:
            """将当前缓冲区落盘为 DocumentChunk；空 chunk 跳过。"""
            if not self.current_id:
                return None

            text_part = "\n".join(self.text_parts).strip()
            table_part = "\n".join(self.table_parts).strip()
            has_text = bool(text_part)
            has_table = bool(table_part)

            # 内容拼接（参数优先：结构化数据放前面，LLM 生成时优先看到）
            content_parts: list[str] = []
            table_annotation = ""
            if has_table:
                full_for_annotation = "\n\n".join([p for p in [text_part, table_part] if p])
                table_annotation = self._annotate_table(full_for_annotation)
                if table_annotation:
                    content_parts.append("[TABLE_PARAMS]\n" + table_annotation)
            if text_part:
                content_parts.append(text_part)
            if has_table:
                content_parts.append(table_part)
            content = "\n\n".join(content_parts)

            if not content:
                self.current_id = None
                return None

            # 模态判定
            modalities = set(self.modality)
            if has_text:
                modalities.add("text")
            if has_table:
                modalities.add("table")
            modality = "mixed" if len(modalities) > 1 else (modalities.pop() if modalities else "text")

            # searchable_text
            searchable = text_part
            if has_table:
                searchable += "\n" + (table_annotation or table_part)
            for _img_id in self.child_images:
                searchable += "\n"

            chunk = DocumentChunk(
                chunk_id=self.current_id,
                doc_id=self.doc_id,
                file_name=self.file_name,
                modality=modality,
                content=content,
                searchable_text=searchable.strip(),
                page=self.page,
                section=self.section,
                position=dict(self.position),
                metadata={"heading_level": self.level},
                child_images=list(self.child_images),
                parent_chunk_id=self.parent_id,
            )
            # 章节结构字段写入（Task 1 新增）
            chunk.section_id = self.section_id
            chunk.section_path = self.section_path
            chunk.section_topic = self.section_topic
            self.chunks.append(chunk)

            # 重置
            self.current_id = None
            self.section = ""
            self.level = 0
            self.page = 1
            self.text_parts = []
            self.table_parts = []
            self.position = {}
            self.modality = set()
            self.child_images = []
            self.parent_id = ""
            self.section_id = ""
            self.section_path = ""
            self.section_topic = ""
            return chunk

        def _annotate_table(self, full_content: str) -> str:
            """表格参数提取代理：委托给外层 DocumentParser._annotate_table。"""
            return self._parser._annotate_table(full_content)

    def _build_heading_chunks(
        self, elements: list[_RawElement], doc_id: str, file_name: str
    ) -> list[DocumentChunk]:
        """
        按标题深度聚合元素为 chunk（heading-aware + parent-child）。

        新版逻辑（v2）：
            1) 预扫描所有 image，并行调用 VL 生成描述（带 1s 间隔保护 OneAPI 限流）
            2) _group_by_heading：按 H1+H2 heading 分组，跨页同标题合并
            3) _build_parent_chunk：为每个 Section 构造 parent + 若干 child
               - parent：仅做上下文锚点（不入向量库，由 INCLUDE_PARENT_IN_INDEX 控制）
               - text child：按 CHUNK_MAX_TOKENS=768 切分
               - table child：表格独立成 chunk（由 ENABLE_TABLE_ISOLATION 控制）
               - image child：独立 image chunk，挂回 parent

        Returns:
            child chunk 列表（parent 不入向量库，仅作为锚点）。
            当 ENABLE_PARENT_CHILD=False 时，降级为旧版单层逻辑（无父子结构）。
        """
        # ---- 预扫描：收集所有图片，调用 VL 生成描述 ----
        image_descriptions = self._vl_pre_scan(elements, file_name)

        # ---- 关闭父子双层时：降级为旧版单层 chunk 流程（向后兼容） ----
        if not config.ENABLE_PARENT_CHILD:
            return self._build_legacy_single_chunks(
                elements, doc_id, file_name, image_descriptions,
            )

        # ---- 新版：按 heading 分组 → 每个 group 构造 parent + child ----
        groups = self._group_by_heading(elements, doc_id, file_name)
        if not groups:
            return []

        # 章节结构识别：遍历 groups 调用 _detect_section_path，填充 group 字段
        # （section_stack 暂未用于父章节推断，预留扩展位）
        section_stack: list[tuple[str, str, str]] = []
        for _g in groups:
            _sid, _spath, _stopic = self._detect_section_path(_g.section, section_stack)
            _g.section_id = _sid
            _g.section_path = _spath
            _g.section_topic = _stopic

        all_chunks: list[DocumentChunk] = []
        for group in groups:
            try:
                parent_chunk, children = self._build_parent_chunk(
                    group, doc_id, file_name, image_descriptions,
                )
            except Exception as e:  # noqa: BLE001
                log.warning(f"parent chunk 构建失败，跳过 Section '{group.section}': {e}")
                continue
            # parent 是否入向量库（默认不入；INCLUDE_PARENT_IN_INDEX=True 时入）
            if config.INCLUDE_PARENT_IN_INDEX:
                all_chunks.append(parent_chunk)
            all_chunks.extend(children)

        log.info(
            f"切块完成: {len(groups)} 个 Section → {len(all_chunks)} 个 chunk"
            f"（parent 入库={config.INCLUDE_PARENT_IN_INDEX}）",
        )
        return all_chunks

    def _build_legacy_single_chunks(
        self,
        elements: list[_RawElement],
        doc_id: str,
        file_name: str,
        image_descriptions: dict[str, str],
    ) -> list[DocumentChunk]:
        """旧版单层 chunk 流程（ENABLE_PARENT_CHILD=False 时使用，向后兼容）。"""
        # 当 image_descriptions 为空时（独立调用路径），执行 VL 预扫描
        if not image_descriptions:
            image_descriptions = self._vl_pre_scan(elements, file_name)

        # 将分块构建状态封装为内部类，避免闭包 + nonlocal 复杂度
        builder = self._ChunkBuilder(doc_id, file_name, self)
        for el in elements:
            # ---- 标题：开启新 chunk ----
            if el.type == "heading":
                builder.flush()
                builder.start_new(el.content, el.level, el.page, dict(el.position))
                continue

            # ---- 图片：作为 child + 独立 image chunk ----
            if el.type == "image":
                if not el.image_bytes:
                    continue
                # 获取图片描述（优先使用预扫描并行缓存，降级到同步调用）
                img_desc = image_descriptions.get(el.image_id)
                if img_desc is None:
                    img_desc = self._vision_understand(
                        el.image_bytes, context_text=f"{file_name} | {builder.section}", image_format=el.image_format
                    ) or ""
                if el.image_id not in (getattr(self, "_annotated_image_ids", None) or ()):
                    img_desc = self._annotate_image(img_desc, context_text=builder.section)

                # 独立 image chunk
                img_chunk_id = uuid4().hex[:12]
                fallback = f"[IMAGE at page {el.page}]"
                img_content = f"[图片内容] {img_desc if img_desc else fallback}"
                parent_text_preview = " ".join(builder.text_parts)[:200] if builder.text_parts else ""
                img_searchable = (
                    f"{file_name} p.{el.page} | {builder.section} | {parent_text_preview} | "
                    f"{img_desc if img_desc else fallback}"
                )
                img_chunk = DocumentChunk(
                    chunk_id=img_chunk_id,
                    doc_id=doc_id, file_name=file_name,
                    modality="image", content=img_content,
                    searchable_text=img_searchable, page=el.page,
                    section=builder.section, position=dict(el.position),
                    metadata={
                        "image_format": el.image_format, "image_id": el.image_id,
                        "image_bytes": el.image_bytes,
                        "image_bytes_len": len(el.image_bytes) if el.image_bytes else 0,
                    },
                    child_images=[], parent_chunk_id=builder.current_id or "",
                )
                # 章节结构字段注入（Task 1 新增；继承自当前 builder 状态）
                img_chunk.section_id = builder.section_id
                img_chunk.section_path = builder.section_path
                img_chunk.section_topic = builder.section_topic
                builder.chunks.append(img_chunk)
                # 关联到当前文本 chunk
                if builder.current_id:
                    builder.child_images.append(img_chunk_id)
                    builder.modality.add("image")
                continue

            # ---- 表格：累加 ----
            if el.type == "table":
                if not builder.current_id:
                    builder.start_new("", 0, el.page)
                builder.table_parts.append(el.content)
                builder.page = el.page
                continue

            # ---- 文本：累加 ----
            if el.type == "text":
                if not builder.current_id:
                    builder.start_new("", 0, el.page)
                builder.text_parts.append(el.content)
                builder.page = el.page
                continue

        # 收尾
        builder.flush()
        return builder.chunks

    # --------------------------------------------------------
    # 模型调用（VL / Chat）
    # --------------------------------------------------------

    def _vl_pre_scan(
        self, elements: list[_RawElement], file_name: str,
    ) -> dict[str, str]:
        """收集所有图片元素，并发调用 VL 生成描述。

        限流策略：提交间隔 2s（OneAPI VL ~30次/分钟，2s 间隔兼容限流）。
        实际并发度由 ``VL_MAX_WORKERS`` 环境变量控制（默认 1，最安全）。
        """
        image_descriptions: dict[str, str] = {}
        _pre_section = ""
        _images_to_process: list[tuple[str, bytes, str, str]] = []

        for el in elements:
            if el.type == "heading":
                _pre_section = el.content
            elif el.type == "image" and el.image_bytes:
                _images_to_process.append((
                    el.image_id, el.image_bytes,
                    f"{file_name} | {_pre_section}", el.image_format,
                ))

        if not _images_to_process:
            return image_descriptions

        log.info("VL 并发生成 %d 张图片描述（workers=%d）...",
                 len(_images_to_process), self._VL_MAX_WORKERS)
        with ThreadPoolExecutor(
            max_workers=self._VL_MAX_WORKERS, thread_name_prefix="vl",
        ) as pool:
            _futures = {}
            for img_id, bts, ctx, fmt in _images_to_process:
                _futures[pool.submit(self._vision_understand, bts, ctx, fmt)] = img_id
                # 提交间隔 2s：错峰限流避免触发 OneAPI 429
                time.sleep(2.0)
            _completed = 0
            for _fut in as_completed(_futures):
                _img_id = _futures[_fut]
                try:
                    image_descriptions[_img_id] = _fut.result() or ""
                except Exception as e:
                    log.warning("VL 并发调用失败 %s: %s", _img_id, e)
                    image_descriptions[_img_id] = ""
                _completed += 1
                if _completed % 10 == 0 or _completed == len(_images_to_process):
                    log.info("VL 进度: %d/%d", _completed, len(_images_to_process))

        # Optional: parallel image-description enhancement (VL_ANNOTATE_WORKERS>=2).
        # Keeps the serial path when unset; parallelizes the LLM annotate bottleneck.
        _ann_workers = int(os.getenv("VL_ANNOTATE_WORKERS", "1"))
        if _ann_workers >= 2 and image_descriptions:
            self._annotated_image_ids = set(image_descriptions.keys())
            _ann_items = [
                (
                    _img_id,
                    image_descriptions[_img_id],
                    ctx.split(" | ", 1)[-1] if " | " in ctx else ctx,
                )
                for _img_id, _bts, ctx, _fmt in _images_to_process
                if _img_id in image_descriptions and image_descriptions[_img_id]
            ]
            if _ann_items:
                log.info("Parallel annotate %d images (workers=%d)...",
                         len(_ann_items), _ann_workers)
                with ThreadPoolExecutor(
                    max_workers=_ann_workers, thread_name_prefix="ann",
                ) as _pool:
                    _ann_futures = {
                        _pool.submit(self._annotate_image, desc, section): img_id
                        for img_id, desc, section in _ann_items
                    }
                    for _fut in as_completed(_ann_futures):
                        _img_id = _ann_futures[_fut]
                        try:
                            enhanced = _fut.result()
                            if enhanced:
                                image_descriptions[_img_id] = enhanced
                        except Exception as _exc:  # noqa: BLE001
                            log.warning("Image annotate failed %s: %s", _img_id, _exc)

        return image_descriptions

    def _vision_understand(
        self,
        image_bytes: Optional[bytes],
        context_text: str = "",
        image_format: str = "png",
    ) -> str:
        """
        调用 Qwen3-VL-Plus 生成图片语义描述（中文，2-3 句话）。

        429 限流时退避重试（最多 3 次），其他错误返回空串。
        """
        if not image_bytes or self._client is None:
            return ""
        try:
            # 压缩图片降低 VL token 消耗（768px 能保留足够细节）
            compressed = _resize_image_for_vl(image_bytes, max_pixels=768)
            b64 = base64.b64encode(compressed).decode("ascii")
            # 压缩后统一为 JPEG mime
            mime = "image/jpeg"

            prompt = (
                "请用中文客观描述这张图片的核心内容（2-3 句话）。"
                "如果包含技术图纸、元器件、表格等，请说明其类型与关键信息。" "若图中包含明确的参数标注或数值（如 VBR=27V、IPP=8A、tp=8/20μs、限值等），必须逐项提取并列出，不要遗漏数值。"
                + (f"该图片来自文档 {context_text}，请结合文档名称准确识别图片中的元器件型号与类型。" if context_text else "")
            )

            # 429 退避重试：OneAPI 全局限制，单次失败后等待 2/4/6s 重试
            last_exc = None
            for attempt in range(3):
                try:
                    resp = self._client.chat.completions.create(
                        model=config.QWEN_VL_MODEL,
                        extra_body={"enable_thinking": False},
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                                    },
                                ],
                            }
                        ],
                        max_tokens=150,  # 2-3 句中文约 80-120 tokens，150 足够
                        temperature=0.2,
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    return text
                except Exception as e:
                    last_exc = e
                    code = str(getattr(e, "code", ""))
                    if "429" in code or "rate" in str(e).lower():
                        wait = 2.0 * (attempt + 1)
                        log.warning(f"VL 429 限流，{wait:.0f}s 后重试 ({attempt + 1}/3)")
                        time.sleep(wait)
                        continue
                    raise  # 非 429 错误不重试
            raise last_exc  # noqa: B904
        except Exception as e:
            log.warning(f"VL 视觉理解失败，使用占位: {e}")
            return ""

    def _annotate_table(self, full_content: str) -> str:
        """从完整内容（文本+表格）中提取所有参数名和数值。

        传入 text_part + table_part 的完整拼接，而非仅 table markdown，
        因为文本部分也常包含参数值（如 "Reverse Breakdown Voltage VBR IT=1mA 26.0 28.0 V"）。

        DISABLE_TABLE_ANNOTATE=1 时跳过，直接返回空串（避免 Qwen3.7-Max 耗时）。
        失败时返回空字符串（不阻塞，避免错误数据污染 content）。
        """
        if os.getenv("DISABLE_TABLE_ANNOTATE", "").lower() in ("1", "true", "yes"):
            return ""
        if not full_content.strip():
            return ""
        if self._chat_client is None:
            return ""

        snippet = full_content[:4000]
        try:
            resp = self._chat_client.chat.completions.create(
                model=config.QWEN_CHAT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "从以下内容中提取所有技术参数（参数名和数值），逐行列出。\n"
                            "规则：\n"
                            "1. 每行格式：'参数名: 数值'\n"
                            "2. 只提取有明确数值的参数，不要输出 '未知' 或 '-'\n"
                            "3. 不要遗漏文本段落中的参数（如 'Reverse Breakdown Voltage VBR IT=1mA 26.0 28.0 V'）\n"
                            "4. 不要生成描述性文字，只输出参数列表\n\n"
                            f"{snippet}"
                        ),
                    }
                ],
                max_tokens=1000,
                temperature=0.0,
                extra_body=config.CHAT_EXTRA_BODY,
            )
            summary = (resp.choices[0].message.content or "").strip()
            return summary
        except Exception as e:
            log.warning(f"表格参数提取失败: {e}")
            return ""

    def _annotate_image(self, image_desc: str, context_text: str = "") -> str:
        """
        增强图片描述：把 VL 输出的描述 + 上下文融合，生成更利于检索的文本。

        失败时原样返回。
        """
        if not image_desc:
            return ""
        if self._chat_client is None or not context_text:
            return image_desc
        # 跳过图片描述增强（DISABLE_IMAGE_ANNOTATE=1），直接返回 VL 原始描述
        if os.getenv("DISABLE_IMAGE_ANNOTATE", "").lower() in ("1", "true", "yes"):
            return image_desc
        # 进度日志：串行 LLM 调用是潜在的耗时瓶颈，便于观察真实耗时
        t0 = time.time()
        try:
            resp = self._chat_client.chat.completions.create(
                model=config.QWEN_CHAT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "请把以下图片描述与上下文融合，生成 1-2 句中文检索增强文本：\n"
                            f"上下文：{context_text}\n图片描述：{image_desc}"
                        ),
                    }
                ],
                max_tokens=150,
                temperature=0.2,
                extra_body=config.CHAT_EXTRA_BODY,
            )
            enhanced = (resp.choices[0].message.content or "").strip()
            log.info(
                "[_annotate_image] 成功 耗时=%.2fs context='%s'",
                time.time() - t0, context_text[:40],
            )
            return enhanced or image_desc
        except Exception as e:
            log.warning(
                "[_annotate_image] 失败 耗时=%.2fs err=%s context='%s'",
                time.time() - t0, type(e).__name__, context_text[:40],
            )
            return image_desc


__all__ = ["DocumentParser", "DocumentChunk"]


# ------------------------------------------------------------
# 解析缓存序列化辅助：bytes（图片）↔ base64，
# 避免 json.dump 遇到 bytes 中途失败而留下损坏的半截缓存文件。
# 定义在文件末尾，方法运行时模块已加载完毕，可正常引用。
# ------------------------------------------------------------

def _cache_sanitize(obj: Any) -> Any:
    """递归把 bytes 编码为 base64 标记字典，供 JSON 序列化。"""
    if isinstance(obj, dict):
        return {k: _cache_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cache_sanitize(v) for v in obj]
    if isinstance(obj, bytes):
        return {"__bytes_b64__": base64.b64encode(obj).decode("ascii")}
    return obj


def _cache_restore(obj: Any) -> Any:
    """把 _cache_sanitize 的产物还原为原始结构。"""
    if isinstance(obj, dict):
        if "__bytes_b64__" in obj and len(obj) == 1:
            return base64.b64decode(obj["__bytes_b64__"])
        return {k: _cache_restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cache_restore(v) for v in obj]
    return obj
