"""
多模态 RAG 系统 — OpenAI Tool Calling 定义 (Task 7)

对外暴露的工具函数清单 + 分发器。
"""
from __future__ import annotations

from typing import Any

from pipeline.query import QueryPipeline


# ============================================================
# 工具定义
# ============================================================

TOOL_SEARCH: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_hardware_knowledge_base",
        "description": "在硬件行业知识库中检索与查询相关的文档片段（支持中/英/德文）",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户查询",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "返回 top_k 结果",
                },
            },
            "required": ["query"],
        },
    },
}

# 工具清单（按需扩展）
TOOLS: list[dict[str, Any]] = [TOOL_SEARCH]

# 工具名称常量（路由层按名分发）
TOOL_NAME_SEARCH: str = "search_hardware_knowledge_base"


# ============================================================
# 分发器
# ============================================================

def _handle_search(arguments: dict[str, Any], pipeline: QueryPipeline) -> dict[str, Any]:
    """执行知识库检索工具。"""
    query = (arguments.get("query") or "").strip()
    if not query:
        return {"error": "query 不能为空", "results": []}
    top_k = int(arguments.get("top_k") or 5)

    # 走完整的 query 流程，提取 sources 与 answer
    result = pipeline.query(query, top_k=top_k, use_rerank=True)
    return {
        "query": query,
        "answer": result.get("answer", ""),
        "results": result.get("sources", []),
        "language": result.get("language", "en"),
    }


# 工具名 → 处理函数的映射表
_HANDLERS: dict[str, Any] = {
    TOOL_NAME_SEARCH: _handle_search,
}


def handle_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    query_pipeline: QueryPipeline,
) -> dict[str, Any]:
    """根据工具名分发到对应处理函数。"""
    handler = _HANDLERS.get(tool_name)
    if handler is None:
        return {"error": f"未知工具: {tool_name}", "supported": list(_HANDLERS)}
    try:
        return handler(arguments or {}, query_pipeline)
    except Exception as e:  # noqa: BLE001
        return {"error": f"工具执行失败: {e}"}


__all__ = [
    "TOOL_SEARCH",
    "TOOLS",
    "TOOL_NAME_SEARCH",
    "handle_tool_call",
]
