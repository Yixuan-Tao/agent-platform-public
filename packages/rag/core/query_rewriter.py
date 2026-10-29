"""
多模态 RAG 系统 — Query 改写器 (Requery 模块)

职责：
    1. 调用 OneAPI 网关的 LLM（默认 qwen3.7-max）把用户的口语化/短查询
       改写为 N 个语义等价的检索 query 变体
    2. 输出严格 JSON 格式（rewrites 列表），含原 query 的标准化版本作为首元素
    3. 所有网络/解析失败统一降级为 [原 query]，不阻塞后续流程

设计要点：
    - 使用 OpenAI JSON 模式（response_format={"type": "json_object"}）保证输出稳定
    - 复用 @retry_on_rate_limit 装饰器处理 429/5xx 限流
    - 缓存由上层 QueryPipeline 统一管理（避免多层缓存失效）
    - 改写 prompt 严格约束"不新增/删减关键信息"，避免幻觉
"""
from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

from config import config
from utils.logger import log
from utils.retry import _is_retryable_error, retry_on_rate_limit
from utils.runtime_budget import model_chat


# 改写系统 prompt（中文为主，多语言兜底）
# 设计目标：让 LLM 输出标准 JSON 数组，第一项是原 query 的标准化版本
_REWRITE_SYSTEM_PROMPT = """你是硬件行业的技术文档检索助手，负责把用户的口语化或简短问题改写为更精准、更易被向量数据库命中的检索 query。

严格遵循以下规则：
1. 仅基于原 query 改写，不要新增/删减关键信息（如产品型号、技术参数、单位、章节名等）。
2. 输出严格的 JSON 对象，格式为：{"rewrites": ["query1", "query2", ...]}，不要包含任何额外文字或 markdown 标记。
3. 数组第一个元素必须是原 query 的标准化版本：
   - 补全省略的主语/宾语
   - 纠正错别字与口语化用词
   - 把别名/缩写映射为标准技术术语（例："钳位电压" → "Clamping Voltage VC"）
4. 后续元素为同义/近义改写，覆盖：
   - 技术参数的标准英文名与单位
   - 同一问题在不同粒度上的表述（细节 vs 总览）
5. 改写总条数 = 1 + n_variants（n_variants ≤ 2），不要包含原 query 的简单重复。
6. 保留原 query 的语言（中文/英文/德文）。"""


# 末尾按语种追加的指令（让 LLM 在用户消息中保持原语言）
_LANGUAGE_HINT = {
    "zh": "请保持与原 query 相同的语言（中文）。",
    "en": "Keep the same language as the original query (English).",
    "de": "Behalten Sie die gleiche Sprache wie die ursprüngliche Anfrage (Deutsch).",
}


class QueryRewriter:
    """Query 改写器：把单条 query 改写为 N 个语义等价的检索变体。"""

    def __init__(self, client: Optional[OpenAI] = None) -> None:
        # 默认从 config 构造 OneAPI 客户端；测试可注入
        self.client = client or OpenAI(
            base_url=config.CHAT_BASE_URL,
            api_key=config.CHAT_API_KEY,
            timeout=config.CHAT_TIMEOUT,
            max_retries=0,  # 主重试由装饰器管理
        )
        self.model = config.QWEN_CHAT_MODEL

    # ---------- 公开方法 ----------

    @retry_on_rate_limit(max_retries=3, base_delay=1.0, max_delay=30.0)
    def rewrite(
        self,
        query: str,
        language: str = "zh",
        n_variants: int = 2,
    ) -> list[str]:
        """改写 query，返回 [标准化原 query, 变体 1, 变体 2, ...]。

        Args:
            query: 原始用户查询。
            language: 语种代码（zh/en/de），影响 LLM 输出的语言一致性。
            n_variants: 额外生成的变体数（不含原 query）；推荐 1-2。

        Returns:
            改写 query 列表（长度 = 1 + n_variants）。
            任何失败场景统一降级为 [原 query]。
        """
        if not query or not query.strip():
            return []
        n_variants = max(0, min(int(n_variants), 2))  # 兜底：不超过 2

        try:
            messages = self._build_messages(query, language, n_variants)
            resp = model_chat(self.client,
                model=self.model,
                messages=messages,
                stream=False,
                temperature=0.1,  # 低温度：减少改写幻觉
                response_format={"type": "json_object"},
                extra_body=config.CHAT_EXTRA_BODY,
            )
            content = (resp.choices[0].message.content or "").strip()
            rewrites = self._parse_rewrites(content, n_variants)
            # 始终把原 query 的标准化版本放在最前（解析结果第一项就是）
            if not rewrites:
                return [query]
            return rewrites
        except Exception as e:  # noqa: BLE001
            # 可重试异常 → 让装饰器处理；不可重试 → 记录后降级
            if _is_retryable_error(e):
                raise
            log.warning("Query 改写失败（非重试异常），降级为原 query: %s", e)
            return [query]

    # ---------- 内部方法 ----------

    def _build_messages(
        self, query: str, language: str, n_variants: int,
    ) -> list[dict]:
        """组装 LLM 调用所需的 messages 列表。"""
        lang_hint = _LANGUAGE_HINT.get(language, _LANGUAGE_HINT["en"])
        # 在 system prompt 末尾追加变体数约束 + 语言约束
        sys_prompt = (
            f"{_REWRITE_SYSTEM_PROMPT}\n\n"
            f"本次任务：生成 1 个标准化版本 + {n_variants} 个变体 = "
            f"{1 + n_variants} 条 query。\n{lang_hint}"
        )
        user_prompt = f"原 query：\n{query}\n\n请输出改写结果。"

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _parse_rewrites(content: str, n_variants: int) -> list[str]:
        """解析 LLM 输出的 JSON，提取 rewrites 列表。

        失败/为空/格式异常时返回空列表（由调用方降级为 [原 query]）。
        """
        if not content:
            return []
        try:
            data = json.loads(content)
        except Exception:  # noqa: BLE001
            log.warning("Query 改写 JSON 解析失败: %.120s", content)
            return []
        rewrites = data.get("rewrites") if isinstance(data, dict) else None
        if not isinstance(rewrites, list):
            return []
        # 过滤：去掉空字符串/纯空白/重复项
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in rewrites:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
        # 截断到期望长度
        return cleaned[: 1 + n_variants]


# 口语化词正则编译缓存（性能：避免每次 query 都重新编译）
_COLLOQUIAL_RE_CACHE: Optional[re.Pattern] = None


def get_colloquial_pattern() -> re.Pattern:
    """获取口语化词正则的编译结果（懒加载）。"""
    global _COLLOQUIAL_RE_CACHE
    if _COLLOQUIAL_RE_CACHE is None:
        _COLLOQUIAL_RE_CACHE = re.compile(config.REQUERY_COLLOQUIAL_RE, re.IGNORECASE)
    return _COLLOQUIAL_RE_CACHE


__all__ = ["QueryRewriter", "get_colloquial_pattern"]
