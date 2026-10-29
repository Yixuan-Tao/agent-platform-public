"""Jira 工单创建工具。"""

from __future__ import annotations

from typing import Any

import httpx

from agent_platform.core.config import get_settings
from agent_platform.tools.base import BaseTool
from agent_platform.tools.result import StructuredToolText, ToolResult


class JiraCreateTicketTool(BaseTool):
    required_role = "user"
    read_only = False
    idempotent = False
    required_scopes = ("jira:create", "jira:read")
    name = "jira_create_ticket"
    description = (
        "在 Jira 创建 IT 工单。用于故障报修、服务请求等需要跟踪的场景。"
        "入参 title、description、priority、category 等。"
    )
    args_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": 255, "description": "工单标题"},
            "category": {"type": "string", "description": "问题分类，如 网络问题/账号权限/软件安装"},
            "description": {"type": "string", "description": "问题详细描述"},
            "priority": {"type": "string", "enum": ["高", "中", "低", "High", "Medium", "Low"], "description": "优先级: 高/中/低"},
            "device_type": {"type": "string", "description": "设备类型"},
            "os": {"type": "string", "description": "操作系统"},
            "error_message": {"type": "string", "description": "错误信息"},
            "urgency": {"type": "string", "description": "紧急程度"},
        },
        "required": ["title", "description"],
    }

    def __init__(self) -> None:
        self.settings = get_settings()

    async def execute(self, title: str, description: str, **kwargs: Any) -> str:
        if not self.settings.jira_url:
            return StructuredToolText(
                "Jira 未配置（JIRA_URL 为空）。请配置后重试，"
                "或将工单信息整理给人工处理。", status="failed", error_code="NOT_CONFIGURED"
            )
        details = (f"分类: {kwargs.get('category', '')}\n"
                   f"设备: {kwargs.get('device_type', '')} / {kwargs.get('os', '')}\n"
                   f"错误信息: {kwargs.get('error_message', '')}\n"
                   f"紧急程度: {kwargs.get('urgency', '')}\n\n{description}")
        payload = {
            "fields": {
                "project": {"key": "IT"},
                "summary": title,
                "description": {"type": "doc", "version": 1, "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": line}]} if line else {"type": "paragraph", "content": []}
                    for line in details.splitlines()]},
                "issuetype": {"name": "Task"},
                "priority": {"name": {"高": "High", "中": "Medium", "低": "Low"}.get(kwargs.get("priority"), kwargs.get("priority", "Medium"))},
            }
        }
        from agent_platform.core.execution import execution_context, fingerprint
        ctx = execution_context.get()
        if ctx and ctx.run_id:
            # Searchable action marker supports reconciliation; Jira still has no native exactly-once guarantee.
            parameters = {"title": title, "description": description, **kwargs}
            payload["fields"]["labels"] = ["agent-platform-" + fingerprint({"run": ctx.run_id, "tool": self.name, "args": parameters})[:32]]
        auth = (self.settings.jira_user, self.settings.jira_token)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.settings.jira_url.rstrip('/')}/rest/api/3/issue",
                    json=payload,
                    auth=auth,
                )
                resp.raise_for_status()
                issue = resp.json()
        except Exception as exc:  # noqa: BLE001
            rejected = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {400, 401, 403, 404, 422}
            return StructuredToolText(f"创建 Jira 工单失败: {exc}", status="failed" if rejected else "unknown", error_code="JIRA_ERROR")
        if not isinstance(issue, dict) or not issue.get("key"):
            return StructuredToolText("外部响应未提供可核验工单 ID，需对账", status="unknown", data=issue, error_code="INVALID_EXTERNAL_RESULT")
        key = issue.get("key", "?")
        url = f"{self.settings.jira_url.rstrip('/')}/browse/{key}"
        verification = await JiraGetTicketTool().execute(key)
        fields = verification.data.get("fields", {}) if verification.status == "success" else {}
        verified = verification.status == "success" and verification.data.get("key") == key and fields.get("summary") == title
        if ctx and ctx.run_id:
            verified = verified and payload["fields"]["labels"][0] in fields.get("labels", [])
        return StructuredToolText(f"工单已创建: {key} ({url})" if verified else f"工单创建返回 {key}，查询核验尚未完成", status="success" if verified else "partial", data={**issue, "title": title},
            external_refs=[{"system": "jira", "id": key, "url": url, "verified": verified}])


class JiraGetTicketTool(BaseTool):
    name = "jira_get_ticket"
    description = "查询 Jira 工单，核验真实工单 ID 与当前状态。"
    required_scopes = ("jira:read",)
    args_schema = {"type": "object", "properties": {"key": {"type": "string", "pattern": "^[A-Z][A-Z0-9_]*-[0-9]+$"}}, "required": ["key"], "additionalProperties": False}
    async def execute(self, key):
        settings = get_settings()
        if not settings.jira_url:
            return ToolResult(status="failed", error_code="NOT_CONFIGURED")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(f"{settings.jira_url.rstrip('/')}/rest/api/3/issue/{key}", auth=(settings.jira_user, settings.jira_token))
                response.raise_for_status()
                data = response.json()
            return ToolResult(status="success", data=data, external_refs=[{"system": "jira", "id": key, "verified": data.get("key") == key}])
        except httpx.HTTPError:
            return ToolResult(status="failed", error_code="JIRA_QUERY_FAILED", retryable=True)


async def reconcile_jira(run_id, arguments, key=None):
    """Read-only reconciliation requires a matching action marker and original title."""
    from agent_platform.core.execution import fingerprint
    label = "agent-platform-" + fingerprint({"run": run_id, "tool": "jira_create_ticket", "args": arguments})[:32]
    settings = get_settings()
    if not settings.jira_url:
        return ToolResult(status="failed", error_code="NOT_CONFIGURED")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            auth = (settings.jira_user, settings.jira_token)
            if key:
                import re
                if not re.fullmatch(r"[A-Z][A-Z0-9_]*-[0-9]+", key):
                    return ToolResult(status="failed", error_code="INVALID_KEY")
                response = await client.get(f"{settings.jira_url.rstrip('/')}/rest/api/3/issue/{key}", auth=auth)
                response.raise_for_status()
                issues = [response.json()]
            else:
                response = await client.get(f"{settings.jira_url.rstrip('/')}/rest/api/3/search/jql",
                    params={"jql": f'labels = "{label}"', "maxResults": 2, "fields": "summary,labels"}, auth=auth)
                response.raise_for_status()
                issues = response.json().get("issues", [])
        matching = [issue for issue in issues if label in issue.get("fields", {}).get("labels", [])
                    and issue.get("fields", {}).get("summary") == arguments.get("title")]
        if len(matching) != 1:
            return ToolResult(status="unknown", error_code="RECONCILIATION_INCONCLUSIVE")
        issue = matching[0]
        return ToolResult(status="success", data=issue, external_refs=[{"system": "jira", "id": issue["key"], "verified": True,
                          "url": settings.jira_url.rstrip("/") + "/browse/" + issue["key"]}])
    except httpx.HTTPError:
        return ToolResult(status="unknown", error_code="JIRA_QUERY_FAILED")
