"""
多模态 RAG 系统 — 向量化层 (Task 4)

职责：
    1. 调用 OneAPI 网关的 /v1/embeddings 生成稠密 / 图片向量
    2. 基于 rank_bm25.BM25Okapi 拟合语料，生成稀疏向量（dict[int, float]）
    3. 通过 Embedder 统一入口，对 DocumentChunk 同时填充 dense / sparse / image 三类向量

设计要点：
    - 统一用 httpx 直打 OneAPI 的 EMBEDDINGS_PATH（默认 /v1/embeddings）
      不依赖 openai SDK 的隐式路径拼接，便于自定义网关适配
    - OpenAIEmbedder：内部 LRU 缓存避免重复调用，批量 ≤ 100 条，单条失败不影响其他
    - BM25SparseEncoder：query-time encoder，编码文档时新词动态加入词表
    - ImageEmbedder：调用 OneAPI 多模态 embedding（OpenAI 兼容协议）；
      协议不通时降级为 None + 占位向量，便于后续接入
    - Embedder：懒加载子编码器；首次调用 fit BM25，后续支持增量 update_corpus
    - 所有网络/IO 调用 try/except 降级：dense=None 或 sparse={} 时不抛错
    - 中文分词优先用 jieba（未安装则按字符切分 + 英文正则切分）
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Union

import httpx
from rank_bm25 import BM25Okapi

from config import config
from db.schemas import (
    MODALITY_IMAGE,
    MODALITY_TABLE,
    MODALITY_TEXT,
    DocumentChunk,
)
from utils.logger import log
from utils.urls import build_embeddings_url
from utils.runtime_budget import model_post, cache_namespace

# 尝试导入 PIL（用于图片格式转换与 RGB 归一化）
try:  # pragma: no cover - 可选依赖
    from PIL import Image as _PILImage  # type: ignore
    _PIL_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PIL_AVAILABLE = False

# 尝试导入 jieba 做中文分词（未安装时降级为字符切分）
try:  # pragma: no cover - 可选依赖
    import jieba  # type: ignore
    _JIEBA_AVAILABLE = True
    jieba.setLogLevel("ERROR")  # 静默模式，避免分词过程刷屏
except Exception:  # noqa: BLE001
    _JIEBA_AVAILABLE = False


# ============================================================
# 共享：httpx 调用 OneAPI /v1/embeddings
# ============================================================

def _post_embeddings(payload: dict, timeout: float = 60.0) -> httpx.Response:
    """统一发送 POST 到 OneAPI 的 embeddings 端点。"""
    url = build_embeddings_url()
    headers = {
        "Authorization": f"Bearer {config.ONEAPI_API_KEY}",
        "Content-Type": "application/json",
    }
    return model_post(url, headers=headers, json=payload, timeout=timeout)


# ============================================================
# 1) OpenAI 文本稠密向量编码器
# ============================================================

class OpenAIEmbedder:
    """调用 OneAPI /v1/embeddings 生成 text-embedding-3-small 的 1536 维稠密向量。

    关键设计：
        - 直接用 httpx 调 {ONEAPI_BASE_URL}{EMBEDDINGS_PATH}，不依赖 SDK
        - 内部字典缓存避免对相同文本重复调用
        - 支持批量（≤ 100 条/批），单条失败不影响其他
        - 失败时返回 None / 全 None 占位
    """

    # OpenAI 单批上限（文档硬性限制）
    _BATCH_SIZE: int = 16  # 从 32 降到 16，减少单次请求对 OneAPI 后端的压力
    # 期望输出维度（与 config.DENSE_EMBED_DIM 对齐）
    _EXPECTED_DIM: int = config.DENSE_EMBED_DIM

    def __init__(self) -> None:
        if not config.ONEAPI_API_KEY:
            log.warning("ONEAPI_API_KEY 为空，稠密向量生成将返回 None")

        # 简易文本缓存（避免重复调用）；多线程下用锁保护
        self._cache: dict[str, list[float]] = {}
        self._cache_lock = threading.Lock()
        # 缓存容量上限，防止长跑内存膨胀
        self._cache_max: int = 4096

    # --------------- 公开 API ---------------

    def embed(self, text: str, max_retries: int = 5, timeout: float = 60.0) -> Optional[list[float]]:
        """编码单条文本；返回 None 表示失败。

        Args:
            max_retries: 最大重试次数（查询路径建议 2，摄入路径用默认 5）
            timeout: HTTP 超时秒数（查询路径建议 15，摄入路径用默认 60）
        """
        if not text:
            return None
        # 缓存命中
        with self._cache_lock:
            if cache_namespace() + text in self._cache:
                return list(self._cache[cache_namespace() + text])

        vec = self._call_remote([text], max_retries=max_retries, timeout=timeout)
        if vec and vec[0] is not None:
            self._store_cache(text, vec[0])
            return list(vec[0])
        return None

    def embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """批量编码；按 _BATCH_SIZE 切片，保留原顺序；失败位置为 None。"""
        results: list[Optional[list[float]]] = [None] * len(texts)
        if not texts:
            return results

        # 先用缓存填充可立即命中的位置
        todo_indices: list[int] = []
        todo_texts: list[str] = []
        with self._cache_lock:
            for i, t in enumerate(texts):
                if not t:
                    continue
                if cache_namespace() + t in self._cache:
                    results[i] = list(self._cache[cache_namespace() + t])
                else:
                    todo_indices.append(i)
                    todo_texts.append(t)

        if not todo_texts:
            return results

        # 分批调用，批次间加延迟避免打爆 OneAPI 后端
        total_batches = (len(todo_texts) + config.EMBED_BATCH_SIZE - 1) // config.EMBED_BATCH_SIZE
        log.info("[embed_batch] 开始向量化: %d 条文本，分 %d 个 batch (size=%d)",
                 len(todo_texts), total_batches, config.EMBED_BATCH_SIZE)
        for idx, start in enumerate(range(0, len(todo_texts), config.EMBED_BATCH_SIZE)):
            if idx > 0:
                time.sleep(1.0)  # 批次间等待 1s，让 OneAPI 后端喘息
            chunk = todo_texts[start:start + config.EMBED_BATCH_SIZE]
            chunk_indices = todo_indices[start:start + config.EMBED_BATCH_SIZE]
            log.info("[embed_batch] 调用 batch %d/%d (size=%d)",
                     idx + 1, total_batches, len(chunk))
            t0 = time.time()
            vecs = self._call_remote(chunk)
            elapsed = time.time() - t0
            ok = sum(1 for v in vecs if v is not None)
            log.info("[embed_batch] batch %d/%d 完成: 成功=%d/%d 耗时=%.1fs",
                     idx + 1, total_batches, ok, len(chunk), elapsed)
            for j, v in enumerate(vecs):
                if v is not None:
                    results[chunk_indices[j]] = list(v)
                    self._store_cache(chunk[j], v)
        return results

    # --------------- 内部实现 ---------------

    def _call_remote(
        self, texts: list[str], max_retries: int = 5, timeout: float = 60.0,
    ) -> list[Optional[list[float]]]:
        """远程调用一次 /v1/embeddings；任何异常都返回全 None。

        对 5xx/网络错误自动重试（指数退避），保证大批量摄入的稳定性。
        """
        if not config.ONEAPI_API_KEY or not texts:
            return [None] * len(texts)

        # 5xx/网络/限流/部分4xx错误自动重试（指数退避），保证大批量摄入稳定性
        # 退避序列：1s → 2s → 4s → 8s → 16s（查询路径可由调用方减小 max_retries 提前退出）
        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            log.info("[_call_remote] attempt %d/%d, model=%s, batch=%d, timeout=%.0fs",
                     attempt, max_retries, config.TEXT_EMBED_MODEL, len(texts), timeout)
            t0 = time.time()
            try:
                resp = _post_embeddings(
                    {"model": config.TEXT_EMBED_MODEL, "input": texts},
                    timeout=timeout,
                )
                log.info("[_call_remote] attempt %d 收到响应: HTTP %d 耗时=%.1fs",
                         attempt, resp.status_code, time.time() - t0)
                # 5xx 服务端错误：自动重试
                if resp.status_code in (500, 502, 503, 504):
                    log.warning(
                        f"embeddings 5xx（重试 {attempt}/{max_retries}）: "
                        f"HTTP {resp.status_code}, model={config.TEXT_EMBED_MODEL}, "
                        f"body={resp.text[:150]}"
                    )
                    if attempt < max_retries:
                        time.sleep(backoff ** (attempt + 1))
                        continue
                    return [None] * len(texts)
                # 429 限流：自动退避重试
                if resp.status_code == 429:
                    log.warning(
                        f"embeddings 429 限流（重试 {attempt}/{max_retries}）: "
                        f"model={config.TEXT_EMBED_MODEL}"
                    )
                    if attempt < max_retries:
                        time.sleep(backoff ** (attempt + 2))  # 限流多等一会
                        continue
                    return [None] * len(texts)
                # OneAPI 后端 bge-m3 在负载过高时实例会重启，返回 400 "Model not found"，
                # 需要更长的退避时间让后端实例完成模型加载
                if resp.status_code == 400 and "Model not found" in resp.text:
                    log.warning(
                        f"embeddings 400 Model not found（重试 {attempt}/{max_retries}）: "
                        f"model={config.TEXT_EMBED_MODEL}, body={resp.text[:200]}"
                    )
                    if attempt < max_retries:
                        time.sleep(backoff ** (attempt + 2))  # 比 5xx 多等一轮
                        continue
                    return [None] * len(texts)
                if resp.status_code != 200:
                    log.warning(
                        f"embeddings 调用失败: HTTP {resp.status_code}, "
                        f"model={config.TEXT_EMBED_MODEL}, "
                        f"url={build_embeddings_url()}, "
                        f"body={resp.text[:200]}"
                    )
                    return [None] * len(texts)
                data = resp.json()
                # 按输入顺序对齐结果
                out: list[Optional[list[float]]] = [None] * len(texts)
                items = data.get("data", [])
                for i in range(len(texts)):
                    try:
                        vec = list(items[i]["embedding"])
                        if len(vec) != self._EXPECTED_DIM:
                            log.warning(
                                f"Embedding 维度异常: 期望 {self._EXPECTED_DIM}, "
                                f"实际 {len(vec)}"
                            )
                            out[i] = None
                        else:
                            out[i] = vec
                    except Exception as e:  # noqa: BLE001
                        log.warning(f"单条 embedding 解析失败: {e}")
                        out[i] = None
                return out
            except Exception as e:  # noqa: BLE001
                log.warning(
                    f"embeddings 调用异常（重试 {attempt}/{max_retries}）: "
                    f"{type(e).__name__}: {e}"
                )
                if attempt < max_retries:
                    time.sleep(backoff ** (attempt + 1))
                    continue
                return [None] * len(texts)
        return [None] * len(texts)

    def _store_cache(self, text: str, vec: list[float]) -> None:
        """写入缓存；超出容量上限时淘汰最旧的一半（FIFO，按 dict 插入顺序）。"""
        with self._cache_lock:
            if len(self._cache) >= self._cache_max:
                # 淘汰最旧的一半条目（Python 3.7+ dict 保持插入顺序）
                evict_count = self._cache_max // 2
                keys_to_evict = list(self._cache.keys())[:evict_count]
                for k in keys_to_evict:
                    del self._cache[k]
            self._cache[cache_namespace() + text] = list(vec)


# ============================================================
# 2) 多模态图片向量编码器 (tongyi-embedding-vision-flash)
# ============================================================

class ImageEmbedder:
    """生成多模态图片向量（百炼原生协议优先，OneAPI OpenAI 兼容兜底）。

    协议选择：
        - native（百炼 DashScope 原生）：POST {DASHSCOPE_MULTIMODAL_URL}
          body = {
              "model": "tongyi-embedding-vision-plus-2026-03-06",
              "input": {"contents": [{"image": "data:image/jpeg;base64,..."}]},
              "parameters": {"dimension": 1024}
          }
          响应：{"output": {"embeddings": [{"embedding": [...]}]}}
        - openai（OneAPI 网关兼容）：
          POST {ONEAPI_BASE_URL}{EMBEDDINGS_PATH}
          body = {"model": ..., "input": [{"type": "image_url",
                  "image_url": {"url": "data:image/png;base64,..."}}]}

    输入支持：
        - 本地文件路径（str / Path）
        - 原始 bytes
        - PIL.Image 对象

    行为约定：
        - 单张失败返回 None，下游走零向量占位 + metadata 标记
        - 支持简单文本哈希缓存（避免重复传输同图）
        - 任何异常都降级，不抛错
    """

    # 期望输出维度（与 config.IMAGE_EMBED_DIM 对齐）
    _EXPECTED_DIM: int = config.IMAGE_EMBED_DIM

    # 单次最大像素边长（避免超大图传输超时）
    _MAX_EDGE: int = 1024

    # JPEG 压缩质量（PNG 也会转 JPEG 以减小体积）
    _JPEG_QUALITY: int = 85

    def __init__(self) -> None:
        # DashScope 原生协议密钥（DASHSCOPE_API_KEY > LLM_API_KEY > ONEAPI_API_KEY）
        self._api_key: str = config.DASHSCOPE_API_KEY
        if not self._api_key:
            log.warning("DashScope/OneAPI 密钥为空，图片向量将降级为零向量")

        # 用 (size, first-128-bits-of-bytes) 做缓存 key
        self._cache: dict[tuple[int, str], list[float]] = {}
        self._cache_lock = threading.Lock()
        self._cache_max: int = 1024
        # 文本查询缓存（文搜图）
        self._text_cache: dict[str, list[float]] = {}

        # 开关：DISABLE_IMAGE_EMBED=1 时直接跳过 OneAPI 调用，所有图片 embedding
        # 走 _fill_image_fallback 路径（text-embed 编码描述或全零向量）。
        self._disabled: bool = os.getenv("DISABLE_IMAGE_EMBED", "").lower() in ("1", "true", "yes")

        # 协议选择：
        # - native：百炼 DashScope 原生多模态协议（不支持 OpenAI 兼容），
        #   用于 dashscope.aliyuncs.com / *.maas.aliyuncs.com 域名，
        #   或显式配置 DASHSCOPE_MULTIMODAL_URL / DASHSCOPE_WORKSPACE_ID。
        # - openai：OneAPI 等网关的 OpenAI 兼容多模态接口。
        self._native: bool = False
        if not self._disabled:
            base = (config.ONEAPI_BASE_URL or "").lower()
            has_dashscope_base = (
                "dashscope.aliyuncs.com" in base or ".maas.aliyuncs.com" in base
            )
            explicit_native = bool(
                os.getenv("DASHSCOPE_MULTIMODAL_URL")
                or config.DASHSCOPE_WORKSPACE_ID
            )
            self._native = has_dashscope_base or explicit_native
            if self._native:
                log.info(
                    "ImageEmbedder 使用百炼原生多模态协议: model=%s, dim=%d, url=%s",
                    config.IMAGE_EMBED_MODEL, self._EXPECTED_DIM,
                    config.DASHSCOPE_MULTIMODAL_URL,
                )
        if self._disabled:
            log.warning(
                "ImageEmbedder 已禁用 (DISABLE_IMAGE_EMBED=1)："
                "不再调用图片向量接口，全部走 fallback 路径"
            )

    # --------------- 公开 API ---------------

    def embed(self, image: Union[str, Path, bytes, "Image"]) -> Optional[list[float]]:
        """编码单张图片，返回 1024 维向量；失败返回 None。"""
        key, raw = self._normalize(image)
        if raw is None:
            return None
        cache_key = (len(raw), _hash_bytes(raw[:128]))
        if cache_namespace():
            cache_key = (cache_namespace(), len(raw), _hash_bytes(raw))
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return list(cached)

        data_uri = self._bytes_to_data_uri(raw)
        if data_uri is None:
            return None

        if self._native:
            vec = self._call_native([{"image": data_uri}])
        else:
            vec = self._call_remote([data_uri])
        if vec and vec[0] is not None:
            with self._cache_lock:
                if len(self._cache) >= self._cache_max:
                    # 淘汰最旧的一半条目
                    evict_count = self._cache_max // 2
                    keys_to_evict = list(self._cache.keys())[:evict_count]
                    for k in keys_to_evict:
                        del self._cache[k]
                self._cache[cache_key] = list(vec[0])
            return list(vec[0])
        return None

    def embed_text(self, text: str) -> Optional[list[float]]:
        """用与图片相同的多模态模型编码文本查询（文搜图/图搜图查询侧）。

        仅原生协议支持文本输入（input={"contents": [{"text": ...}]}）；
        OpenAI 兼容网关无法把文本编码进图片向量空间，返回 None。
        """
        if not text or not text.strip():
            return None
        if self._disabled or not self._native:
            log.warning(
                "embed_text 需要百炼原生多模态协议（native=%s, disabled=%s）",
                self._native, self._disabled,
            )
            return None
        cached = self._text_cache.get(cache_namespace() + text)
        if cached is not None:
            return list(cached)
        vec = self._call_native([{"text": text}])
        if vec and vec[0] is not None:
            if len(self._text_cache) >= self._cache_max:
                evict_count = self._cache_max // 2
                for k in list(self._text_cache)[:evict_count]:
                    del self._text_cache[k]
            self._text_cache[cache_namespace() + text] = list(vec[0])
            return list(vec[0])
        return None

    # --------------- 内部实现 ---------------

    def _normalize(self, image: Union[str, Path, bytes, "Image"]) -> tuple[Optional[str], Optional[bytes]]:
        """把入参规整为 (cache_key_hint, png_or_jpeg_bytes)；失败返回 (None, None)。"""
        if isinstance(image, (str, Path)):
            p = Path(image)
            if not p.is_file():
                log.warning(f"图片文件不存在: {p}")
                return None, None
            try:
                raw = p.read_bytes()
            except Exception as e:  # noqa: BLE001
                log.warning(f"读取图片失败: {p}: {e}")
                return None, None
            return str(p), self._maybe_compress(raw)

        if isinstance(image, (bytes, bytearray)):
            return "bytes", self._maybe_compress(bytes(image))

        if _PIL_AVAILABLE and isinstance(image, _PILImage.Image):
            return "pil", self._pil_to_bytes(image)

        log.warning(f"不支持的图片输入类型: {type(image).__name__}")
        return None, None

    def _maybe_compress(self, raw: bytes) -> Optional[bytes]:
        """对超长边图片做等比缩放 + 转 JPEG，减小传输体积。"""
        if not _PIL_AVAILABLE:
            return raw
        try:
            img = _PILImage.open(io.BytesIO(raw))
            img.load()
        except Exception as e:  # noqa: BLE001
            log.warning(f"图片解码失败（PIL）: {e}")
            return None

        # 模式归一化（RGBA / P / L → RGB）
        if img.mode in ("RGBA", "LA", "P"):
            bg = _PILImage.new("RGB", img.size, (255, 255, 255))
            try:
                img = img.convert("RGBA")
                bg.paste(img, mask=img.split()[-1])
                img = bg
            except Exception:
                img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 等比缩放到 _MAX_EDGE 以内
        w, h = img.size
        if max(w, h) > self._MAX_EDGE:
            scale = self._MAX_EDGE / max(w, h)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = img.resize(new_size, _PILImage.LANCZOS)

        buf = io.BytesIO()
        try:
            img.save(buf, format="JPEG", quality=self._JPEG_QUALITY, optimize=True)
        except Exception as e:  # noqa: BLE001
            log.warning(f"图片压缩失败: {e}")
            return None
        return buf.getvalue()

    def _pil_to_bytes(self, image: "Image") -> Optional[bytes]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        if max(w, h) > self._MAX_EDGE:
            scale = self._MAX_EDGE / max(w, h)
            image = image.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                _PILImage.LANCZOS,
            )
        buf = io.BytesIO()
        try:
            image.save(buf, format="JPEG", quality=self._JPEG_QUALITY, optimize=True)
        except Exception as e:  # noqa: BLE001
            log.warning(f"PIL 编码失败: {e}")
            return None
        return buf.getvalue()

    def _bytes_to_data_uri(self, raw: bytes) -> Optional[str]:
        """将 JPEG/PNG bytes 编码为 data URI。"""
        if not raw:
            return None
        # JPEG 文件头 0xFFD8，PNG 0x89504E47
        if raw[:2] == b"\xff\xd8":
            mime = "image/jpeg"
        elif raw[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        else:
            mime = "image/jpeg"  # 上游已规范化，缺省按 jpeg
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _call_native(self, items: list[dict]) -> list[Optional[list[float]]]:
        """调用百炼多模态 Embedding 原生接口；5xx/429/网络异常自动重试。

        请求：
            POST {DASHSCOPE_MULTIMODAL_URL}
            {"model": ..., "input": {"contents": [{"image": ...} | {"text": ...}]},
             "parameters": {"dimension": dim}}
        响应：
            {"output": {"embeddings": [{"embedding": [...]}]}}
        """
        if self._disabled:
            return [None] * len(items)
        if not self._api_key or not items:
            return [None] * len(items)
        url = config.DASHSCOPE_MULTIMODAL_URL
        payload = {
            "model": config.IMAGE_EMBED_MODEL,
            "input": {"contents": items},
            "parameters": {"dimension": self._EXPECTED_DIM},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        max_retries = 3
        backoff = 2.0
        for attempt in range(1, max_retries + 1):
            try:
                resp = model_post(url, headers=headers, json=payload, timeout=120.0)
                if resp.status_code in (500, 502, 503, 504):
                    log.warning(
                        "多模态 embedding 5xx（重试 %d/%d）: HTTP %d, body=%s",
                        attempt, max_retries, resp.status_code, resp.text[:150],
                    )
                    if attempt < max_retries:
                        time.sleep(backoff ** attempt)
                        continue
                    return [None] * len(items)
                if resp.status_code == 429:
                    log.warning(
                        "多模态 embedding 429 限流（重试 %d/%d）: model=%s",
                        attempt, max_retries, config.IMAGE_EMBED_MODEL,
                    )
                    if attempt < max_retries:
                        time.sleep(backoff ** (attempt + 1))
                        continue
                    return [None] * len(items)
                if resp.status_code != 200:
                    log.warning(
                        "多模态 embedding 原生接口失败: HTTP %d, model=%s, url=%s, body=%s",
                        resp.status_code, config.IMAGE_EMBED_MODEL, url, resp.text[:300],
                    )
                    return [None] * len(items)
                data = resp.json()
                if not isinstance(data, dict):
                    return [None] * len(items)
                output = data.get("output", {}) or {}
                embeds = output.get("embeddings", []) if isinstance(output, dict) else []
                out: list[Optional[list[float]]] = [None] * len(items)
                for i in range(len(items)):
                    try:
                        item = embeds[i] if i < len(embeds) else None
                        if isinstance(item, dict):
                            vec = list(item["embedding"])
                        elif isinstance(item, list):
                            vec = list(item)
                        else:
                            out[i] = None
                            continue
                        if len(vec) != self._EXPECTED_DIM:
                            log.warning(
                                "多模态 embedding 维度异常: 期望 %d, 实际 %d",
                                self._EXPECTED_DIM, len(vec),
                            )
                            out[i] = None
                        else:
                            out[i] = vec
                    except Exception as e:  # noqa: BLE001
                        log.warning("单条多模态 embedding 解析失败: %s", e)
                        out[i] = None
                return out
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "多模态 embedding 调用异常（重试 %d/%d）: %s: %s",
                    attempt, max_retries, type(e).__name__, e,
                )
                if attempt < max_retries:
                    time.sleep(backoff ** attempt)
                    continue
                return [None] * len(items)
        return [None] * len(items)

    def _call_remote(self, data_uris: list[str]) -> list[Optional[list[float]]]:
        """调用 /v1/embeddings 多模态接口；5xx/429/网络异常自动重试。"""
        if not self._api_key or not data_uris:
            return [None] * len(data_uris)
        # 开关：完全跳过 OneAPI 调用，直接走 fallback 路径
        if self._disabled:
            return [None] * len(data_uris)
        # OpenAI 兼容的多模态 embedding 协议
        payload_input = [
            {"type": "image_url", "image_url": {"url": uri}}
            for uri in data_uris
        ]
        max_retries = 3
        backoff = 2.0
        for attempt in range(1, max_retries + 1):
            try:
                resp = _post_embeddings(
                    {"model": config.IMAGE_EMBED_MODEL, "input": payload_input},
                    timeout=120.0,
                )
                # 5xx 服务端错误：自动重试
                if resp.status_code in (500, 502, 503, 504):
                    log.warning(
                        "图片 embedding 5xx（重试 %d/%d）: HTTP %d, body=%s",
                        attempt, max_retries, resp.status_code, resp.text[:150],
                    )
                    if attempt < max_retries:
                        time.sleep(backoff ** attempt)
                        continue
                    return [None] * len(data_uris)
                # 429 限流：自动退避重试
                if resp.status_code == 429:
                    log.warning(
                        "图片 embedding 429 限流（重试 %d/%d）: model=%s",
                        attempt, max_retries, config.IMAGE_EMBED_MODEL,
                    )
                    if attempt < max_retries:
                        time.sleep(backoff ** (attempt + 1))
                        continue
                    return [None] * len(data_uris)
                if resp.status_code != 200:
                    log.warning(
                        "图片 embedding 调用失败: HTTP %d, model=%s, url=%s, body=%s",
                        resp.status_code, config.IMAGE_EMBED_MODEL,
                        build_embeddings_url(), resp.text[:300],
                    )
                    return [None] * len(data_uris)
                data = resp.json()
                items = data.get("data", [])
                out: list[Optional[list[float]]] = [None] * len(data_uris)
                for i in range(len(data_uris)):
                    try:
                        vec = list(items[i]["embedding"])
                        if len(vec) != self._EXPECTED_DIM:
                            log.warning(
                                "图片 embedding 维度异常: 期望 %d, 实际 %d",
                                self._EXPECTED_DIM, len(vec),
                            )
                            out[i] = None
                        else:
                            out[i] = vec
                    except Exception as e:  # noqa: BLE001
                        log.warning("单条图片 embedding 解析失败: %s", e)
                        out[i] = None
                return out
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "图片 embedding 调用异常（重试 %d/%d）: %s: %s",
                    attempt, max_retries, type(e).__name__, e,
                )
                if attempt < max_retries:
                    time.sleep(backoff ** attempt)
                    continue
                return [None] * len(data_uris)
        return [None] * len(data_uris)


def _hash_bytes(b: bytes) -> str:
    """稳定但不加密的字节摘要（用于缓存 key）。"""
    return hashlib.md5(b).hexdigest()


# ============================================================
# 3) BM25 稀疏向量编码器（query-time encoder）
# ============================================================

class BM25SparseEncoder:
    """基于 BM25Okapi 的稀疏向量编码器。

    关键约定：
        - encode_query / encode_documents 返回 dict[int, float]
        - 词项 id 由内部自增 vocab 分配
        - 编码查询时未登录词忽略；编码文档时新词动态加入词表
    """

    # BM25 超参（与 rank_bm25.BM25Okapi 默认保持一致）
    _K1: float = 1.5
    _B: float = 0.75

    def __init__(self) -> None:
        # 词表：token -> int id
        self._vocab: dict[str, int] = {}
        # 已 tokenize 的语料（用于 fit / 增量）
        self._corpus_tokens: list[list[str]] = []
        # 原始语料文本（便于 update_corpus 时追加）
        self._corpus_texts: list[str] = []
        # BM25 索引
        self._bm25: Optional[BM25Okapi] = None
        # 统计量（编码文档权重时需要）
        self._avgdl: float = 0.0
        self._doc_lens: list[int] = []
        self._df: dict[str, int] = {}
        self._N: int = 0

    # --------------- 词表 / 分词 ---------------

    def _tokenize(self, text: str) -> list[str]:
        """中英混合分词：有 jieba 用 jieba，否则中文按字 + 英文正则。"""
        if not text:
            return []
        text = text.lower()
        if _JIEBA_AVAILABLE:
            # jieba.cut 在中英混合场景下对英文会按空格/标点切分
            return [t for t in jieba.cut(text) if t.strip()]
        # fallback：英文/数字按词，中文按字
        en_tokens = re.findall(r"[a-z0-9]+", text)
        cn_tokens = re.findall(r"[\u4e00-\u9fff]", text)
        return en_tokens + cn_tokens

    def _get_or_assign_id(self, token: str) -> int:
        """词表查/增。"""
        tid = self._vocab.get(token)
        if tid is None:
            tid = len(self._vocab)
            self._vocab[token] = tid
        return tid

    # --------------- 语料拟合 ---------------

    def fit(self, corpus: list[str]) -> None:
        """用整段语料拟合 BM25 词表 + 统计量。重复调用会重置。"""
        self._vocab.clear()
        self._corpus_texts = list(corpus)
        self._corpus_tokens = [self._tokenize(t) for t in self._corpus_texts]
        # 先把语料里所有词都注册到词表
        for tokens in self._corpus_tokens:
            for tok in tokens:
                self._get_or_assign_id(tok)
        self._rebuild_index()

    def update_corpus(self, extra_texts: list[str]) -> None:
        """增量添加语料文本，仅分词新增文本并追加到词表和 token 列表。"""
        if not extra_texts:
            return
        self._corpus_texts.extend(extra_texts)
        # 仅对新增文本分词，追加到已有 token 列表（避免全量重复分词）
        new_tokens = [self._tokenize(t) for t in extra_texts]
        self._corpus_tokens.extend(new_tokens)
        for tokens in new_tokens:
            for tok in tokens:
                self._get_or_assign_id(tok)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """根据当前语料重建 BM25Okapi 与统计量。"""
        n = len(self._corpus_tokens)
        self._N = n
        if n == 0:
            self._bm25 = None
            self._avgdl = 0.0
            self._doc_lens = []
            self._df = {}
            return

        self._doc_lens = [len(toks) for toks in self._corpus_tokens]
        self._avgdl = sum(self._doc_lens) / n

        # 文档频次
        df: dict[str, int] = {}
        for toks in self._corpus_tokens:
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1
        self._df = df

        try:
            self._bm25 = BM25Okapi(self._corpus_tokens, k1=self._K1, b=self._B)
        except Exception as e:  # noqa: BLE001
            log.warning(f"BM25Okapi 初始化失败: {e}")
            self._bm25 = None

    # --------------- 编码接口 ---------------

    def encode_query(self, query: str) -> dict[int, float]:
        """编码查询：仅对词表内词项返回权重，未登录词忽略。

        权重使用 BM25 IDF 作为简化值（query 端不展开 TF 分量）。
        """
        if not query or self._N == 0:
            return {}
        tokens = self._tokenize(query)
        if not tokens:
            return {}
        out: dict[int, float] = {}
        for tok in tokens:
            tid = self._vocab.get(tok)
            if tid is None:
                continue
            out[tid] = out.get(tid, 0.0) + self._idf(tok)
        return out

    def encode_documents(self, texts: list[str]) -> list[dict[int, float]]:
        """编码文档列表：返回与 texts 等长的稀疏向量列表。
        文档里的新词会动态注册到词表（确保后续查询能命中）。
        未拟合语料时直接返回空字典列表。
        """
        if not texts:
            return []
        if self._N == 0:
            log.warning("BM25SparseEncoder 未 fit，文档编码返回空向量")
            return [{} for _ in texts]

        out: list[dict[int, float]] = []
        for text in texts:
            tokens = self._tokenize(text)
            if not tokens:
                out.append({})
                continue
            # 1) 统计当前文档的词频
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            # 2) 词表动态扩容 + 3) 计算每个词的 BM25 权重
            doc_len = len(tokens)
            vec: dict[int, float] = {}
            for tok, f in tf.items():
                tid = self._get_or_assign_id(tok)
                vec[tid] = self._tf_component(f, doc_len) * self._idf(tok)
            out.append(vec)
        return out

    # --------------- BM25 内部公式 ---------------

    def _idf(self, token: str) -> float:
        """BM25 IDF 公式（与 rank_bm25 一致）。"""
        df = self._df.get(token, 0)
        # 经典 BM25+1 形式
        return (self._N - df + 0.5) / (df + 0.5) + 1.0

    def _tf_component(self, tf: int, doc_len: int) -> float:
        """BM25 TF 归一化部分。"""
        if self._avgdl <= 0 or doc_len <= 0:
            return 0.0
        num = tf * (self._K1 + 1)
        den = tf + self._K1 * (1 - self._B + self._B * doc_len / self._avgdl)
        return num / den

    # --------------- 辅助查询 ---------------

    @property
    def vocab_size(self) -> int:
        """当前词表大小。"""
        return len(self._vocab)

    @property
    def is_fitted(self) -> bool:
        """是否已拟合。"""
        return self._N > 0


# ============================================================
# 3) 统一入口
# ============================================================

class Embedder:
    """向量化统一入口：懒加载三个子编码器，对 DocumentChunk 一站式填充向量。

    用法：
        embedder = Embedder()
        embedder.embed_chunks(chunks)            # 首次会 fit BM25
        embedder.update_corpus(extra_texts)      # 增量语料
        embedder.embed_chunks(more_chunks)        # 后续只编码
    """

    def __init__(self) -> None:
        # 懒加载：仅在首次 embed_chunks 时实例化，节省冷启动时间
        self._openai: Optional[OpenAIEmbedder] = None
        self._bm25: Optional[BM25SparseEncoder] = None
        self._image: Optional[ImageEmbedder] = None
        # 记录已纳入语料的 chunk id，避免重复 fit
        self._seen_chunk_ids: set[str] = set()
        self._lock = threading.Lock()

    # --------------- 子编码器懒加载 ---------------

    @property
    def openai(self) -> OpenAIEmbedder:
        if self._openai is None:
            self._openai = OpenAIEmbedder()
        return self._openai

    @property
    def bm25(self) -> BM25SparseEncoder:
        if self._bm25 is None:
            self._bm25 = BM25SparseEncoder()
        return self._bm25

    @property
    def image(self) -> ImageEmbedder:
        if self._image is None:
            self._image = ImageEmbedder()
        return self._image

    # --------------- 公开 API ---------------

    def update_corpus(self, extra_texts: Iterable[str]) -> None:
        """把额外语料加入 BM25（增量更新）。"""
        texts = [t for t in extra_texts if t]
        if not texts:
            return
        with self._lock:
            self.bm25.update_corpus(texts)

    def embed_query(self, query: str) -> tuple[Optional[list[float]], dict[int, float]]:
        """为单条查询生成 (dense_vector, sparse_vector)。

        行为：
            - dense 走 OpenAI 稠密编码（查询场景用 2 次重试 + 15s 超时，降低延迟）
            - sparse 走 BM25 稀疏编码（必须先 fit 过语料；否则返回空 dict）
        """
        if not query or not query.strip():
            return None, {}
        # 稠密向量：单条调用，查询场景降低重试和超时
        dense = self.openai.embed(query, max_retries=2, timeout=15.0)
        # 稀疏向量：BM25 已拟合时返回词表 id → IDF 权重
        sparse: dict[int, float] = {}
        if self.bm25.is_fitted:
            try:
                sparse = self.bm25.encode_query(query)
            except Exception as e:  # noqa: BLE001
                log.warning(f"BM25 查询编码失败: {e}")
                sparse = {}
        return dense, sparse

    def embed_image_query(self, query: str) -> Optional[list[float]]:
        """把文本查询编码到图片向量空间（文搜图/图搜图查询侧）。

        仅百炼原生多模态协议支持；不可用（禁用/无原生协议/无密钥）时返回 None，
        调用方应跳过图像向量检索，不影响主链路。
        """
        if not query or not query.strip():
            return None
        if self.image._disabled or not self.image._native:
            return None
        return self.image.embed_text(query)

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """为 chunks 填充 dense/sparse/image 三类向量（in-place + 返回）。
        行为约定：
            - text/table/mixed：dense + sparse
            - image：dense（来自 searchable_text）+ image_vector（暂留零向量 TODO）
            - 任何子步骤失败，该字段降级为空，不影响整体流程
        """
        if not chunks:
            return chunks

        with self._lock:
            # 1) 第一次或出现新 chunk 时，按需 fit / 增量更新 BM25
            new_texts = [
                c.searchable_text
                for c in chunks
                if c.id not in self._seen_chunk_ids and c.searchable_text
            ]
            if not self.bm25.is_fitted:
                if new_texts:
                    self.bm25.fit(new_texts)
            elif new_texts:
                # 增量添加：避免重复加入相同文本
                self.bm25.update_corpus(new_texts)

            for c in chunks:
                self._seen_chunk_ids.add(c.id)

            # 2) 批量编码稠密向量
            self._fill_dense_vectors(chunks)
            # 3) 批量编码稀疏向量
            self._fill_sparse_vectors(chunks)
            # 4) 填充 image_vector（仅 image 模态）
            self._fill_image_vectors(chunks)

        return chunks

    # --------------- 内部：按字段填充 ---------------

    def _fill_dense_vectors(self, chunks: list[DocumentChunk]) -> None:
        """对所有需要 dense 的 chunk 批量调用 /v1/embeddings。"""
        # 收集要编码的文本：所有模态的 searchable_text 都走 dense 编码
        targets: list[tuple[int, str]] = [
            (i, c.searchable_text) for i, c in enumerate(chunks)
            if c.searchable_text
        ]

        if not targets:
            return
        texts = [t for _, t in targets]
        vecs = self.openai.embed_batch(texts)
        for (idx, _), v in zip(targets, vecs):
            chunks[idx].dense_vector = list(v) if v else []

    def _fill_sparse_vectors(self, chunks: list[DocumentChunk]) -> None:
        """BM25 稀疏编码。

        稀疏向量仅缓存在 ``chunk.sparse_vector``（内存中），不写入 ChromaDB。
        """
        targets: list[tuple[int, str]] = []
        for i, c in enumerate(chunks):
            # image 模态不做 BM25（视觉内容与文本词项不匹配）
            if c.modality == MODALITY_IMAGE:
                continue
            targets.append((i, c.searchable_text))

        if not targets:
            for c in chunks:
                if c.modality == MODALITY_IMAGE:
                    c.sparse_vector = {}
            return

        texts = [t for _, t in targets]
        vecs = self.bm25.encode_documents(texts)
        for (idx, _), v in zip(targets, vecs):
            chunks[idx].sparse_vector = dict(v) if v else {}

        # 显式给 image 模态置空（防御性）
        for c in chunks:
            if c.modality == MODALITY_IMAGE:
                c.sparse_vector = {}

    def _fill_image_vectors(self, chunks: list[DocumentChunk]) -> None:
        """填充图片向量（并行批量处理）。

        收集所有 image 模态 chunk，通过 ThreadPoolExecutor 并行调用
        OneAPI 图片 embedding，显著减少多图文档的摄入耗时。
        失败时降级到 text-embedding 编码描述或零向量。
        """
        target_dim = self.image._EXPECTED_DIM  # 1024

        # 1) 收集需要真实 embedding 的图片 chunk（有 image_bytes 的 image 模态）
        image_indices: list[int] = []
        for i, c in enumerate(chunks):
            if c.modality != MODALITY_IMAGE:
                c.image_vector = []
                continue
            if c.metadata.get("image_bytes"):
                image_indices.append(i)

        # 2) 无图 chunk 直接降级
        if not image_indices:
            for c in chunks:
                if c.modality == MODALITY_IMAGE:
                    self._fill_image_fallback(c, target_dim)
            return

        # 3) 并行调用图片 embedding
        results: dict[int, Optional[list[float]]] = {}
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="img_emb") as pool:
            futures = {}
            for idx in image_indices:
                raw = chunks[idx].metadata.get("image_bytes")
                fut = pool.submit(self._embed_image_safe, raw)
                futures[fut] = idx

            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = None

        # 4) 应用结果：成功写入，失败降级
        for idx in image_indices:
            c = chunks[idx]
            vec = results.get(idx)
            if vec and len(vec) == target_dim:
                c.image_vector = list(vec)
                c.metadata["embedding_status"] = "ok"
                continue
            self._fill_image_fallback(c, target_dim)

    def _embed_image_safe(self, raw: bytes) -> Optional[list[float]]:
        """单张图片 embedding 的包装方法，供线程池调用。"""
        try:
            return self.image.embed(raw)
        except Exception as e:
            log.warning(f"image.embed 并行调用失败: {e}")
            return None

    def _fill_image_fallback(self, c: DocumentChunk, target_dim: int) -> None:
        """图片 embedding 降级路径：text-embedding 编码描述或零向量。

        DISABLE_IMAGE_EMBED=1 时，text-embedding 降级也跳过（避免在 bge-m3 模型
        不可用时触发大量重试），直接写零向量 + 状态标记。
        """
        raw = c.metadata.get("image_bytes")
        if not raw:
            c.image_vector = [0.0] * target_dim
            c.metadata["embedding_status"] = "missing_data"
            return

        # 图片 embedding 已全局禁用 → 不再调 text-embedding，直接零向量
        if self.image._disabled:
            c.image_vector = [0.0] * target_dim
            c.metadata["embedding_status"] = "disabled"
            return

        desc = (c.searchable_text or c.content or "").strip()
        if not desc:
            c.image_vector = [0.0] * target_dim
            c.metadata["embedding_status"] = "fallback_zero"
            return

        try:
            text_vec = self.openai.embed(desc)
        except Exception as e:
            log.warning(f"text-embedding 降级也失败: {e}")
            text_vec = None

        if text_vec and len(text_vec) >= target_dim:
            c.image_vector = list(text_vec[:target_dim])
            c.metadata["embedding_status"] = "fallback_text"
        elif text_vec:
            c.image_vector = list(text_vec) + [0.0] * (target_dim - len(text_vec))
            c.metadata["embedding_status"] = "fallback_text"
        else:
            c.image_vector = [0.0] * target_dim
            c.metadata["embedding_status"] = "fallback_zero"


__all__ = ["OpenAIEmbedder", "ImageEmbedder", "BM25SparseEncoder", "Embedder"]
