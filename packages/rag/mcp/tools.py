"""
MCP 工具定义 — 注册所有可供 Agent 调用的工具
"""
from __future__ import annotations

from typing import Any

# 工具名称常量
TOOL_PLAN = "plan"
TOOL_SEARCH = "search_knowledge_base"
TOOL_COMPARE = "emc_compare_documents"

# 工具定义列表（MCP 标准格式）
MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": TOOL_PLAN,
        "description": "基于知识库检索生成结构化的工程实施计划文档。适用于硬件设计、测试方案、系统架构等场景。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "计划主题，如'TVS器件选型方案'、'EMC测试计划'",
                },
                "context": {
                    "type": "string",
                    "description": "额外上下文或约束条件（可选）",
                },
                "language": {
                    "type": "string",
                    "enum": ["zh", "en", "de"],
                    "default": "zh",
                    "description": "输出语言",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": TOOL_SEARCH,
        "description": "在硬件行业知识库中检索相关文档片段（支持中/英/德文）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "返回结果数",
                },
                "language": {
                    "type": "string",
                    "enum": ["zh", "en", "de"],
                    "default": "zh",
                    "description": "检索语言",
                },
            },
            "required": ["query"],
        },
    },
]

TOOL_EMC_COMPARE = {
    "type": "function",
    "function": {
        "name": "emc_compare_documents",
        "description": "对两份 EMC 标准文档进行结构化对比，支持章节路由和精细度控制",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "对比问题描述"},
                "doc_a": {"type": "string", "description": "文档 A 的 ID"},
                "doc_b": {"type": "string", "description": "文档 B 的 ID"},
                "language": {"type": "string", "description": "输出语言，zh 或 en", "default": "zh"}
            },
            "required": ["query", "doc_a", "doc_b"]
        }
    }
}

TOOL_COMPONENT_CHECK = {
    "type": "function",
    "function": {
        "name": "component_interchangeability_check",
        "description": "对两份元器件文档进行严格参数级对比，判断是否可互相替换",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_a": {"type": "string", "description": "文档 A（待替换件）的 ID"},
                "doc_b": {"type": "string", "description": "文档 B（替换件）的 ID"},
                "params_template": {"type": "object", "description": "参数表模板 JSON，如 {'electrical': {'VBR': {'unit': 'V', 'min': 26}}}"},
                "strict_mode": {"type": "boolean", "description": "严格模式", "default": True},
                "language": {"type": "string", "description": "输出语言", "default": "zh"}
            },
            "required": ["doc_a", "doc_b", "params_template"]
        }
    }
}

TOOL_VISUALIZE_EMC = {
    "type": "function",
    "function": {
        "name": "visualize_emc_table",
        "description": "将 EMC 标准中的限值表/参数表渲染为 ECharts 可视化图表",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "可视化描述"},
                "doc_id": {"type": "string", "description": "文档 ID（可选，自动匹配时留空）"},
                "doc_a": {"type": "string", "description": "叠加对比的文档 A ID（可选）"},
                "doc_b": {"type": "string", "description": "叠加对比的文档 B ID（可选）"},
                "top_k": {"type": "integer", "description": "候选表格 chunk 数", "default": 5}
            },
            "required": ["query"]
        }
    }
}

# OpenAI 格式的工具定义（用于 ALL_TOOLS）
TOOL_SEARCH_DEF = {
    "type": "function",
    "function": {
        "name": TOOL_SEARCH,
        "description": "在硬件行业知识库中检索相关文档片段（支持中/英/德文）",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询"},
                "top_k": {"type": "integer", "description": "返回结果数", "default": 5},
                "language": {"type": "string", "enum": ["zh", "en", "de"], "description": "检索语言", "default": "zh"},
            },
            "required": ["query"],
        },
    },
}

ALL_TOOLS = [TOOL_SEARCH_DEF, TOOL_EMC_COMPARE, TOOL_COMPONENT_CHECK, TOOL_VISUALIZE_EMC]

__all__ = ["TOOL_PLAN", "TOOL_SEARCH", "TOOL_COMPARE", "TOOL_EMC_COMPARE", "TOOL_COMPONENT_CHECK", "TOOL_VISUALIZE_EMC", "ALL_TOOLS", "MCP_TOOLS"]
