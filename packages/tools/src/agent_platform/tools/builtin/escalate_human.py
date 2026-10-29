"""转人工工具。"""

from __future__ import annotations

from typing import Any

from agent_platform.tools.base import BaseTool


class EscalateHumanTool(BaseTool):
    name = "escalate_human"
    description = "把问题转交给人工支持（生成转交摘要），用于 Agent 无法处理或用户要求的场景。"
    args_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "给人工支持的问题摘要"},
            "reason": {"type": "string", "description": "转人工原因"},
        },
        "required": ["summary"],
    }

    async def execute(self, summary: str, reason: str = "", **kwargs: Any) -> str:
        reason_txt = f"（原因：{reason}）" if reason else ""
        return f"已生成转交摘要{reason_txt}（尚未提交人工支持渠道）：{summary}"
