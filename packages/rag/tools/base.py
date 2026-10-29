"""工具层公共基类和工具函数"""

import json
from collections import Counter
from typing import Any

from utils.logger import log


def format_chunks_for_llm(chunks: list[dict]) -> str:
    """将 chunks 格式化为 LLM 可读的纯文本（按 page 升序排列）。

    Args:
        chunks: 文档 chunks 列表，每项含 content / page / file_name 等字段。

    Returns:
        格式化后的纯文本，每条前带页码标记。
    """
    if not chunks:
        return "（文档无内容）"
    sorted_chunks = sorted(chunks, key=lambda c: (c.get("page", 0) or 0))
    parts: list[str] = []
    seen: set[str] = set()
    for c in sorted_chunks:
        content = (c.get("content") or "").strip()
        if not content or content in seen:
            continue
        seen.add(content)
        page = c.get("page", "?")
        parts.append(f"[p.{page}] {content}")
    return "\n\n".join(parts)


def safe_parse_json(content: str) -> dict:
    """安全解析 JSON，处理 LLM 输出中的 markdown 包裹等各种格式问题。

    Args:
        content: 可能包含 JSON 的原始字符串。

    Returns:
        解析成功的 dict，失败时返回空 dict。
    """
    if not content:
        return {}
    text = content.strip()
    # 去掉 markdown 代码块包裹
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        log.warning("safe_parse_json 解析失败: %s | content_preview=%s",
                    e, text[:200])
        return {}
    return obj if isinstance(obj, dict) else {}


def extract_doc_ids(candidates: list[dict]) -> list[str]:
    """从候选列表中提取文档 ID 列表（按命中数降序）。

    Args:
        candidates: 检索返回的候选 chunk 列表。

    Returns:
        按命中 chunk 数降序排列的 doc_id 列表。
    """
    doc_counts: Counter[str] = Counter()
    for h in candidates:
        did = h.get("doc_id", "")
        if did:
            doc_counts[did] += 1
    return [did for did, _ in doc_counts.most_common()]


def collect_image_chunks(contexts: list[dict]) -> list[dict]:
    """从 contexts 中收集所有图片 chunk 信息（去重）。

    图片 chunk 可能出现在两个位置：
      1. contexts 顶层 — 直接检索到的 image 类型 chunk
      2. linked_chunks — 图文关联回查到的图片 chunk

    返回的每个条目包含 image_path（磁盘路径），供前端/报告读取图片文件。

    Args:
        contexts: 检索返回的上下文 chunk 列表。

    Returns:
        去重后的图片 chunk 摘要列表。
    """
    seen: set[str] = set()
    result: list[dict] = []

    def _append(chunk: dict) -> None:
        cid = str(chunk.get("id", ""))
        if not cid or cid in seen:
            return
        seen.add(cid)
        meta = chunk.get("metadata", {}) or {}
        image_path = meta.get("image_path", "")
        result.append({
            "id": cid,
            "page": chunk.get("page", 0),
            "file_name": chunk.get("file_name", ""),
            "description": (chunk.get("content") or "")[:200],
            "image_path": image_path,
        })

    for ctx in contexts:
        if ctx.get("modality") == "image":
            _append(ctx)
        for linked in ctx.get("linked_chunks") or []:
            if linked.get("modality") == "image":
                _append(linked)

    return result


def source_summary(hit: dict) -> dict[str, Any]:
    """生成单个检索结果的来源摘要（避免回传大字段）。

    Args:
        hit: 单个检索结果字典。

    Returns:
        精简后的来源摘要，包含 id / file_name / page / modality / content / score。
    """
    content = hit.get("content", "") or ""
    snippet = content[:240] + ("..." if len(content) > 240 else "")
    return {
        "id": hit.get("id", ""),
        "file_name": hit.get("file_name", ""),
        "page": hit.get("page", 0),
        "modality": hit.get("modality", "text"),
        "content": snippet,
        "score": hit.get("score", 0.0),
    }


def escape_filter_value(value: str) -> str:
    """转义 filter 表达式中的双引号，防止 ChromaDB where 条件被截断。"""
    return value.replace('"', '\\"')
