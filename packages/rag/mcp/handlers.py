"""
MCP 工具执行器 — 将 MCP 工具调用分发到现有 Pipeline
"""
from __future__ import annotations

import json
from typing import Any

from pipeline.query import QueryPipeline
from utils.logger import log


def _build_plan_prompt(topic: str, context: str, sources_text: str, language: str) -> str:
    """构建 plan 生成的 system prompt。"""
    lang_map = {"zh": "中文", "en": "English", "de": "Deutsch"}
    lang = lang_map.get(language, "中文")

    return f"""你是硬件工程领域的资深架构师，擅长基于技术文档制定实施计划。

请根据以下要求生成一份结构化的工程实施计划：

**计划主题**：{topic}
**额外约束**：{context or "无"}
**输出语言**：{lang}

**参考资料**（来自知识库检索）：
{sources_text if sources_text else "（无参考资料，请基于通用工程知识生成）"}

**输出格式要求**（严格按以下结构，使用 Markdown）：

## 1. 背景与动机
- 说明为什么需要这个计划，解决什么问题

## 2. 目标与范围
- 明确计划要达成的目标
- 界定范围（做什么 / 不做什么）

## 3. 技术方案
- 核心技术选型及理由
- 架构设计要点

## 4. 实施步骤
- 按优先级列出具体步骤（每步一句话描述）
- 标注关键依赖和前置条件

## 5. 风险与应对
- 识别 3-5 个主要风险
- 每个风险给出应对措施

## 6. 验证方案
- 如何验证计划达成目标
- 验收标准

## 7. 参考来源
- 列出参考资料中引用的文档和页码"""


def handle_plan(arguments: dict[str, Any], pipeline: QueryPipeline) -> dict[str, Any]:
    """执行 plan 工具：检索知识库 → 生成结构化计划。"""
    topic = (arguments.get("topic") or "").strip()
    if not topic:
        return {"error": "topic 不能为空"}

    context = (arguments.get("context") or "").strip()
    language = arguments.get("language", "zh")

    # 1) 检索知识库获取相关上下文
    search_result = pipeline.query(
        f"为以下计划主题检索相关技术资料：{topic} {context}",
        top_k=8,
        use_rerank=True,
        language=language,
    )

    sources = search_result.get("sources", [])
    sources_text = ""
    for i, src in enumerate(sources, 1):
        fn = src.get("file_name", "unknown")
        page = src.get("page", 0)
        content = src.get("content", "")[:1000]
        sources_text += f"\n[{i}] {fn} (p.{page})\n{content}\n"

    # 2) 构建 prompt 并调用 LLM 生成计划
    prompt = _build_plan_prompt(topic, context, sources_text, language)

    try:
        plan_content = pipeline.generator.generate(
            query=topic,
            contexts=[],
            language=language,
            system_prompt=prompt,
        )
    except Exception as e:
        log.exception("plan 生成失败: %s", e)
        return {"error": f"计划生成失败: {e}"}

    return {
        "plan_title": topic,
        "plan_content": plan_content,
        "sources": [
            {"file_name": s.get("file_name", ""), "page": s.get("page", 0)}
            for s in sources
        ],
        "language": language,
    }


def handle_search(arguments: dict[str, Any], pipeline: QueryPipeline) -> dict[str, Any]:
    """执行知识库检索。"""
    query = (arguments.get("query") or "").strip()
    if not query:
        return {"error": "query 不能为空"}
    top_k = int(arguments.get("top_k") or 5)
    language = arguments.get("language", "zh")

    result = pipeline.query(query, top_k=top_k, use_rerank=True, language=language)
    return {
        "query": query,
        "answer": result.get("answer", ""),
        "results": result.get("sources", []),
        "language": result.get("language", "en"),
    }


def handle_compare(arguments: dict[str, Any], pipeline: QueryPipeline) -> dict[str, Any]:
    """执行文档对比。"""
    query = (arguments.get("query") or "").strip()
    doc_a = (arguments.get("doc_a") or "").strip()
    doc_b = (arguments.get("doc_b") or "").strip()

    if not query or not doc_a or not doc_b:
        return {"error": "query, doc_a, doc_b 均不能为空"}

    result = pipeline.compare(query=query, doc_a=doc_a, doc_b=doc_b)
    return {
        "answer": result.get("answer", ""),
        "doc_a": doc_a,
        "doc_b": doc_b,
        "sources": result.get("sources", []),
    }


# ============================================================
# MCP 工具处理器（异步模式，与 server.py 的新分发机制配合）
# ============================================================


async def _handle_search(args: dict, request) -> dict:
    """处理知识库检索工具调用"""
    pipeline = _get_pipeline(request)
    return handle_search(args, pipeline)


async def _handle_emc_compare(args: dict, request) -> dict:
    """处理 EMC 标准对比工具调用"""
    from api.compare_routes import compare_emc as _compare_emc
    from api.schemas import EmcCompareRequest

    req = EmcCompareRequest(**args)
    result = await _compare_emc(req, request)
    return {"result": result.model_dump()}


async def _handle_component_check(args: dict, request) -> dict:
    """处理元器件替换判断工具调用"""
    from api.compare_routes import compare_component as _compare_component
    from api.schemas import ComponentCompareRequest

    req = ComponentCompareRequest(**args)
    result = await _compare_component(req, request)
    return {"result": result.model_dump()}


async def _handle_visualize_emc(args: dict, request) -> dict:
    """处理 EMC 表格可视化工具调用"""
    from api.compare_routes import visualize_emc as _visualize_emc
    from api.schemas import VisualizeEmcRequest

    req = VisualizeEmcRequest(**args)
    result = await _visualize_emc(req, request)
    return {"result": result.model_dump()}


# 工具名 → 异步处理函数的映射表
HANDLERS = {
    "search": _handle_search,
    "emc_compare_documents": _handle_emc_compare,
    "component_interchangeability_check": _handle_component_check,
    "visualize_emc_table": _handle_visualize_emc,
}


def _get_pipeline(request) -> QueryPipeline:
    """从 FastAPI Request 中获取 QueryPipeline。"""
    pipeline = getattr(request.app.state, "query_pipeline", None)
    if pipeline is None:
        pipeline = QueryPipeline()
        request.app.state.query_pipeline = pipeline
    return pipeline


__all__ = [
    "handle_plan", "handle_search", "handle_compare",
    "HANDLERS",
]
