"""
本地 token 计数工具

职责：
    1. 估算一段文本的 token 数（用于切块粒度控制）
    2. 优先使用 tiktoken（OpenAI 官方 cl100k_base 编码）
    3. 降级为字符启发式（中文 1.5 字符/token，英文 0.25 字符/token）

设计要点：
    - 单次导入，无副作用
    - 函数式 API（不维护状态）
    - tiktoken 加载失败不抛错，仅 log 警告后降级
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from utils.logger import log

# 尝试加载 tiktoken（OpenAI 官方分词器）
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    log.warning("tiktoken 未安装，token 计数降级为字符启发式（精度较低）")


# tiktoken 编码单例（懒加载，线程安全）
_encoder: Optional["tiktoken.Encoding"] = None
_encoder_lock = __import__("threading").Lock()


def _get_encoder():
    """获取 tiktoken cl100k_base 编码器（懒加载单例，线程安全）。"""
    global _encoder
    if _TIKTOKEN_AVAILABLE and _encoder is None:
        with _encoder_lock:
            # 双重检查：锁内再次确认，避免竞态条件
            if _encoder is None:
                try:
                    _encoder = tiktoken.get_encoding("cl100k_base")
                except Exception as e:  # noqa: BLE001
                    log.warning("加载 tiktoken 编码器失败: %s", e)
                    return None
    return _encoder


@lru_cache(maxsize=2048)
def count_tokens(text: str) -> int:
    """估算一段文本的 token 数。

    Args:
        text: 待计数的文本（None 视作空字符串）。

    Returns:
        估算的 token 数（非负整数）。
    """
    if not text:
        return 0

    # 优先：tiktoken
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:  # noqa: BLE001
            pass

    # 降级：字符启发式
    return _heuristic_count(text)


def _heuristic_count(text: str) -> int:
    """字符启发式 token 计数（tiktoken 不可用时降级）。

    规则：
        - 中文字符（CJK）按 1.5 字符/token 估算
        - 英文/数字按 0.25 字符/token 估算（与 cl100k_base 接近）
        - 标点/空白按 0.5 字符/token
    """
    if not text:
        return 0

    cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", text))
    other_count = len(text) - cjk_count

    # 简化策略：CJK 1 char ≈ 1 token，ASCII 4 char ≈ 1 token
    cjk_tokens = cjk_count
    other_tokens = max(0, (other_count + 3) // 4)  # ceil 除法

    return cjk_tokens + other_tokens


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """按 token 限制切分文本。

    切分策略（从粗到细）：
        1) 优先按 \n\n 段落切分
        2) 单段仍超限 → 按 \n 行切分
        3) 单行仍超限 → 按 句子切分（中英标点 + 换行）
        4) 仍超限 → 按 字符数硬切（避免超长 token 阻塞）

    Args:
        text: 待切分文本。
        max_tokens: 单段最大 token 数（> 0）。

    Returns:
        切分后的文本片段列表。
    """
    if not text or not text.strip():
        return []
    if max_tokens <= 0:
        return [text]

    total = count_tokens(text)
    if total <= max_tokens:
        return [text]

    # 1) 段落级
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p and p.strip()]
    if len(paragraphs) > 1:
        result = _merge_fragments(paragraphs, max_tokens)
        if all(count_tokens(p) <= max_tokens for p in result):
            return result

    # 2) 行级
    lines = [line.strip() for line in text.split("\n") if line and line.strip()]
    if len(lines) > 1:
        result = _merge_fragments(lines, max_tokens)
        if all(count_tokens(p) <= max_tokens for p in result):
            return result

    # 3) 句子级（中英标点）
    sentence_pattern = r"(?<=[.!?。！？])\s+|(?<=[;；])\s*"
    sentences = [s.strip() for s in re.split(sentence_pattern, text) if s and s.strip()]
    if len(sentences) > 1:
        result = _merge_fragments(sentences, max_tokens)
        if all(count_tokens(p) <= max_tokens for p in result):
            return result

    # 4) 硬切（按实际 char/token 比例，避免中英文密度差异导致超限）
    # 中文 1 char ≈ 1 token，英文 4 char ≈ 1 token；用样本估算更准确
    sample = text[: min(len(text), 200)]
    sample_tokens = count_tokens(sample) if sample else 0
    if sample_tokens > 0:
        char_per_token = len(sample) / sample_tokens
    else:
        char_per_token = 4  # 英文兜底
    char_limit = max(1, int(max_tokens * char_per_token))
    return [text[i : i + char_limit] for i in range(0, len(text), char_limit)]


def _merge_fragments(fragments: list[str], max_tokens: int) -> list[str]:
    """把多个 fragment 累积合并为不超过 max_tokens 的块。

    Args:
        fragments: 待合并的片段（按顺序拼接）。
        max_tokens: 单块最大 token 数。

    Returns:
        合并后的文本块列表。
    """
    if not fragments:
        return []
    if max_tokens <= 0:
        return list(fragments)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0
    sep_tokens = count_tokens("\n\n")  # 段落间隔的 token 成本

    for frag in fragments:
        frag_tokens = count_tokens(frag)
        # 单个 fragment 自身超限 → 单独成块（不切，避免改动语义）
        if frag_tokens > max_tokens:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            chunks.append(frag)
            continue
        # 累积后是否超限
        extra = frag_tokens + (sep_tokens if current_parts else 0)
        if current_tokens + extra > max_tokens and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = [frag]
            current_tokens = frag_tokens
        else:
            current_parts.append(frag)
            current_tokens += extra

    if current_parts:
        chunks.append("\n\n".join(current_parts))
    return chunks


__all__ = ["count_tokens", "split_by_tokens"]
