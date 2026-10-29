"""RAG 知识检索工具：调用独立 RAG 服务的纯检索端点 /tools/retrieve。

使用纯检索端点（不生成答案）的原因：Agent 自己会基于片段生成回答，
如果检索端点同时生成，会多花一次模型调用的延迟与费用，而且那份答案会被丢弃。
"""

from __future__ import annotations

from typing import Any

import httpx

from agent_platform.core.config import get_settings
from agent_platform.tools.base import BaseTool
from agent_platform.tools.result import StructuredToolText
from agent_platform.tools.rag_contract import post as rag_post, RagContractError
from agent_platform.core.execution import BudgetExceeded


class RagSearchTool(BaseTool):
    name = "rag_search"
    description = (
        "从企业知识库检索资料（多模态 RAG：PDF/DOCX/图片/表格），"
        "用于回答需要文档依据的问题。返回检索片段，引用来源。"
    )
    args_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索问题或关键词"},
            "top_k": {"type": "integer", "description": "返回片段数量，默认 5"},
        },
        "required": ["query"],
    }

    def __init__(self, base_url: str | None = None) -> None:
        self.settings = get_settings().model_copy(update={"rag_service_url": base_url}) if base_url else get_settings()
        self.base_url = self.settings.rag_service_url.rstrip("/")

    async def execute(self, query: str, top_k: int = 5, **kwargs: Any) -> str:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                data = await rag_post(client, self.settings, "/tools/retrieve", {"query": query, "top_k": int(top_k)})
        except (BudgetExceeded, RagContractError):
            raise
        except Exception as exc:  # noqa: BLE001
            return StructuredToolText(f"知识库检索失败（{self.base_url}）: {exc}", status="failed", error_code="RAG_UNAVAILABLE")

        results = data if isinstance(data, list) else data.get("results", data.get("documents", []))
        if not results:
            return StructuredToolText("知识库没有检索到相关内容。", status="partial", data=data, error_code="NO_EVIDENCE")
        lines = [f"检索到 {len(results)} 条相关内容："]
        for i, item in enumerate(results, 1):
            if isinstance(item, dict):
                source = item.get("source") or item.get("document_name") or item.get("file_name") or ""
                text = item.get("text") or item.get("content") or item.get("chunk") or str(item)
                score = item.get("score")
                score_txt = f" (score={score:.3f})" if isinstance(score, (int, float)) else ""
                lines.append(f"[{i}]{score_txt} 来源: {source}\n{text}")
            else:
                lines.append(f"[{i}] {item}")
        return StructuredToolText("\n\n".join(lines), data=data, evidence_refs=[
            {k: item[k] for k in ("doc_id", "chunk_id", "source", "document_name", "file_name", "page", "section", "version") if k in item}
            for item in results if isinstance(item, dict)])
