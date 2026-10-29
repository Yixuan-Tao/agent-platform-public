"""EMC 领域工具（平台侧）：调用 RAG 服务的完整对比/替换/可视化能力。

历史实现在 ``packages/agents/.../emc_compare/tools.py`` 里做了一套简化逻辑
（只比章节标题、可视化返回占位 HTML），与 RAG 服务内成熟的实现不一致，
导致"从平台用"比"直接用 RAG"能力更弱。这里改为服务适配层：

    - 参数与 RAG 的 /compare/* 契约一致
    - 返回结构化字段（comparison_matrix / citations / datasets / answer_status）
    - RAG 不可用时返回明确说明，而不是静默给出弱结论
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from loguru import logger

from agent_platform.core.config import Settings, get_settings
from agent_platform.tools.base import BaseTool
from agent_platform.tools.result import StructuredToolText
from agent_platform.tools.rag_contract import post as rag_post


class EmcServiceError(RuntimeError):
    """调用 RAG 对比服务失败。"""


class EmcCompareService:
    """RAG `/compare/*` 端点的轻量客户端。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _headers(self) -> dict[str, str]:
        token = getattr(self.settings, "rag_service_token", "")
        if not token:
            return {}
        return {"X-Local-Token": token, "Authorization": f"Bearer {token}"}

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.rag_service_url.rstrip('/')}{path}"
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                return await rag_post(client, self.settings, path, payload)
        except httpx.HTTPError as exc:
            raise EmcServiceError(f"RAG 对比服务不可达: {exc}") from exc
        except ValueError as exc:
            raise EmcServiceError("RAG 对比服务返回非 JSON 内容") from exc

    async def list_documents(self) -> list[dict[str, Any]]:
        """读取 RAG 文档列表（含 doc_id）。"""
        url = f"{self.settings.rag_service_url.rstrip('/')}/documents"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            raise EmcServiceError(f"RAG 文档列表不可达: {exc}") from exc
        if resp.status_code >= 400:
            raise EmcServiceError(f"RAG 文档列表返回 {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise EmcServiceError("RAG 文档列表返回非 JSON 内容") from exc
        return list(data.get("documents") or [])

    async def resolve_doc_id(self, value: str) -> str:
        """把「doc_id 或文件名关键词」解析为 doc_id。

        ``/compare/*`` 只接受 doc_id，而 LLM 只可能看到文件名，因此需要这层映射；
        否则工具在真实使用中必然传错参数、返回"未在指定文档中找到内容"。
        """
        candidate = (value or "").strip()
        if not candidate:
            raise EmcServiceError("文档标识为空")
        if re.fullmatch(r"[0-9a-fA-F]{8,32}", candidate):
            return candidate
        docs = await self.list_documents()
        lowered = candidate.lower()
        exact = [d for d in docs if str(d.get("file_name", "")).lower() == lowered]
        partial = [d for d in docs if lowered in str(d.get("file_name", "")).lower()]
        if len(exact or partial) > 1:
            raise EmcServiceError("文档关键词命中多个版本，请明确选择 doc_id：" + json.dumps(exact or partial, ensure_ascii=False))
        for match in exact or partial:
            doc_id = str(match.get("doc_id") or "")
            if doc_id:
                logger.info("EMC 文档解析: {} -> {} ({})", candidate, match.get("file_name"), doc_id)
                return doc_id
        raise EmcServiceError(
            f"未找到匹配的文档: {candidate}（可用文件名示例："
            + ", ".join(str(d.get("file_name", "")) for d in docs[:3])
            + "）"
        )

    async def post_with_docs(
        self, path: str, payload: dict[str, Any], *, doc_keys: tuple[str, ...]
    ) -> dict[str, Any]:
        """先解析 payload 中的文档标识，再发起请求。"""
        resolved = dict(payload)
        for key in doc_keys:
            value = resolved.get(key)
            if value:
                resolved[key] = await self.resolve_doc_id(str(value))
        return await self.post(path, resolved)


def _format_matrix(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = ["| 字段 | A | B | 结论 |", "| --- | --- | --- | --- |"]
    for row in rows:
        status = {"same": "一致", "different": "存在差异", "missing": "证据不足"}.get(
            str(row.get("status", "")), str(row.get("status", ""))
        )
        a = ", ".join(map(str, row.get("a") or [])) or "-"
        b = ", ".join(map(str, row.get("b") or [])) or "-"
        lines.append(f"| {row.get('field', '')} | {a} | {b} | {status} |")
    return "\n".join(lines)


def _format_citations(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return ""
    seen: list[str] = []
    for c in citations:
        label = " / ".join(
            str(c.get(k, "") or "")
            for k in ("file_name", "section", "page")
        ).strip(" /")
        if label and label not in seen:
            seen.append(label)
    if not seen:
        return ""
    return "证据来源：\n" + "\n".join(f"- {item}" for item in seen[:20])


class EmcCompareTool(BaseTool):
    name = "emc_compare"
    description = (
        "对比两份 EMC 标准/版本：基于人工确认证据或完整语料，输出逐项对比矩阵、"
        "结论与引用来源。用于标准对比、限值差异、测试项目差异等问题。"
    )
    args_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "对比需求，如“对比辐射发射限值”"},
            "doc_a": {"type": "string", "description": "文档 A：文件名关键词（或已知 doc_id）"},
            "doc_b": {"type": "string", "description": "文档 B：文件名关键词（或已知 doc_id）"},
            "language": {"type": "string", "description": "输出语言，默认 zh"},
            "comparison_mode": {
                "type": "string",
                "description": "standard（整份标准对比）或 test_item（单项测试对比）",
            },
            "test_item": {"type": "string", "description": "单项对比时的测试项目，如 RE/CE/BCI"},
        },
        "required": ["query", "doc_a", "doc_b"],
    }

    def __init__(self, service: EmcCompareService | None = None) -> None:
        self.service = service or EmcCompareService()

    async def execute(
        self,
        query: str,
        doc_a: str,
        doc_b: str,
        language: str = "zh",
        comparison_mode: str = "standard",
        test_item: str = "",
        **kwargs: Any,
    ) -> str:
        payload = {
            "query": query,
            "doc_a": doc_a,
            "doc_b": doc_b,
            "language": language or "zh",
            "comparison_mode": comparison_mode or "standard",
            "test_item": test_item or "",
        }
        try:
            data = await self.service.post_with_docs(
                "/compare/emc", payload, doc_keys=("doc_a", "doc_b")
            )
        except EmcServiceError as exc:
            logger.warning("EMC 对比失败: {}", exc)
            return StructuredToolText(f"EMC 对比失败：{exc}。请确认 RAG 服务已启动且两份文档均已入库。", status="failed", error_code="RAG_ERROR")

        parts: list[str] = [str(data.get("answer", "")).strip()]
        matrix = _format_matrix(data.get("comparison_matrix") or data.get("comparison_items") or [])
        if matrix:
            parts.append("逐项对比：\n" + matrix)
        citations = _format_citations(data.get("citations") or [])
        if citations:
            parts.append(citations)
        status = data.get("answer_status") or "ok"
        if status != "ok":
            parts.append(f"（答案状态：{status}）")
        return StructuredToolText("\n\n".join(p for p in parts if p), data=data,
            status="success" if status == "ok" else "partial", evidence_refs=data.get("citations") or [])


class EmcComponentCheckTool(BaseTool):
    name = "emc_component_check"
    description = (
        "判断候选元件能否替代原元件：按参数模板逐字段比对，输出匹配情况、"
        "容差判定、未覆盖参数与引用来源。"
    )
    args_schema = {
        "type": "object",
        "properties": {
            "doc_a": {"type": "string", "description": "原元件文档：文件名关键词（或已知 doc_id）"},
            "doc_b": {"type": "string", "description": "候选元件文档：文件名关键词（或已知 doc_id）"},
            "params_template": {
                "type": "string",
                "description": "参数模板 JSON 字符串，如 {\"VBR\": {\"min\": 24, \"max\": 30}}",
            },
            "strict_mode": {"type": "boolean", "description": "严格模式，默认 true"},
            "language": {"type": "string", "description": "输出语言，默认 zh"},
        },
        "required": ["doc_a", "doc_b", "params_template"],
    }

    def __init__(self, service: EmcCompareService | None = None) -> None:
        self.service = service or EmcCompareService()

    async def execute(
        self,
        doc_a: str,
        doc_b: str,
        params_template: str,
        strict_mode: bool = True,
        language: str = "zh",
        **kwargs: Any,
    ) -> str:
        try:
            template = json.loads(params_template)
        except (TypeError, ValueError) as exc:
            return f"参数模板不是合法 JSON: {exc}"
        if not isinstance(template, dict):
            return "参数模板必须是 JSON 对象（字段 -> 要求）。"
        payload = {
            "doc_a": doc_a,
            "doc_b": doc_b,
            "params_template": template,
            "strict_mode": bool(strict_mode),
            "language": language or "zh",
        }
        try:
            data = await self.service.post_with_docs(
                "/compare/component", payload, doc_keys=("doc_a", "doc_b")
            )
        except EmcServiceError as exc:
            logger.warning("元件替换判定失败: {}", exc)
            return StructuredToolText(f"元件替换判定失败：{exc}。请确认 RAG 服务已启动且两份文档均已入库。", status="failed", error_code="RAG_ERROR")

        lines = [
            f"结论：{data.get('conclusion', '')}",
            str(data.get("conclusion_detail", "")).strip(),
        ]
        for item in data.get("params_comparison") or []:
            mark = "通过" if item.get("match") else "不通过"
            lines.append(
                f"[{mark}] {item.get('param', '')}"
                f"：A={item.get('doc_a_value')} / B={item.get('doc_b_value')}"
                f"{'（' + str(item.get('note')) + '）' if item.get('note') else ''}"
            )
        failed = data.get("failed_params") or []
        if failed:
            lines.append("未在文档中找到的参数：" + ", ".join(map(str, failed)))
        citations = _format_citations(data.get("citations") or [])
        if citations:
            lines.append(citations)
        if (data.get("answer_status") or "ok") != "ok":
            lines.append(f"（答案状态：{data['answer_status']}）")
        return StructuredToolText("\n".join(line for line in lines if line), data=data,
            status="success" if (data.get("answer_status") or "ok") == "ok" else "partial", evidence_refs=data.get("citations") or [])


class EmcVisualizeTool(BaseTool):
    name = "emc_visualize"
    description = "把 EMC 限值表/参数表渲染为可嵌入的 ECharts 可视化 HTML 片段与数据集。"
    args_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "要可视化的内容，如“辐射发射限值表”"},
            "doc_id": {"type": "string", "description": "单文档 doc_id"},
            "doc_a": {"type": "string", "description": "对比场景下的文档 A"},
            "doc_b": {"type": "string", "description": "对比场景下的文档 B"},
            "table_type": {"type": "string", "description": "表格类型提示（可选）"},
        },
        "required": ["query"],
    }

    def __init__(self, service: EmcCompareService | None = None) -> None:
        self.service = service or EmcCompareService()

    async def execute(
        self,
        query: str,
        doc_id: str | None = None,
        doc_a: str | None = None,
        doc_b: str | None = None,
        table_type: str | None = None,
        **kwargs: Any,
    ) -> str:
        payload = {
            "query": query,
            "doc_id": doc_id,
            "doc_a": doc_a,
            "doc_b": doc_b,
            "table_type": table_type,
        }
        try:
            data = await self.service.post_with_docs(
                "/compare/visualize", payload, doc_keys=("doc_id", "doc_a", "doc_b")
            )
        except EmcServiceError as exc:
            logger.warning("EMC 可视化失败: {}", exc)
            return StructuredToolText(f"EMC 可视化失败：{exc}", status="failed", error_code="RAG_ERROR")
        datasets = data.get("datasets") or []
        return json.dumps(
            {
                "table_type": data.get("table_type", ""),
                "datasets": datasets,
                "html_bundle": data.get("html_bundle", ""),
                "sources": data.get("sources") or [],
            },
            ensure_ascii=False,
        )


__all__ = [
    "EmcCompareService",
    "EmcCompareTool",
    "EmcComponentCheckTool",
    "EmcServiceError",
    "EmcVisualizeTool",
]
