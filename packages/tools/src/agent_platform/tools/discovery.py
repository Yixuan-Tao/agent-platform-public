"""Local discovery cannot discover permissions; independent execution is explicit."""
from agent_platform.skills.registry import tokens
from agent_platform.tools.base import BaseTool
from agent_platform.tools.result import ToolResult
from agent_platform.core.execution import execution_context

def discover_tools(registry, query, *, scopes, permitted=None, limit=5):
    query_tokens = set(tokens(query))
    eligible = [tool for tool in registry.list() if set(tool.required_scopes).issubset(scopes)
                and (permitted is None or tool.name in permitted)]
    return sorted([tool.manifest() for tool in eligible if query_tokens & set(tokens(tool.name + " " + tool.description))],
                  key=lambda tool: len(query_tokens & set(tokens(tool["name"] + " " + tool["description"]))), reverse=True)[:limit]

async def independent_calls(gateway, calls, *, independent=False, role="user"):
    if not independent or any(not gateway.get(call["name"]).read_only for call in calls):
        return [await gateway.execute(call["name"], call.get("arguments"), role) for call in calls]
    return await gateway.execute_many(calls, role)


class DiscoverToolsTool(BaseTool):
    name = "discover_authorized_tools"
    description = "检索本任务已有权限及技能范围内的工具，返回完整参数定义。"
    args_schema = {"type": "object", "properties": {"query": {"type": "string", "minLength": 1}}, "required": ["query"], "additionalProperties": False}
    def __init__(self, registry):
        self.registry = registry
    async def execute(self, query):
        ctx = execution_context.get()
        from agent_platform.tools.gateway import ToolGateway, ToolAuthorizationError
        candidates = discover_tools(self.registry, query, scopes=ctx.tool_scopes, permitted=ctx.permitted_tools)
        allowed = []
        gateway = ToolGateway(tools=self.registry.list())
        for item in candidates:
            try:
                gateway.authorize(item["name"], ctx.role)
                allowed.append(item)
            except ToolAuthorizationError:
                continue
        return ToolResult(status="success", data=allowed)


class InvokeDiscoveredTool(BaseTool):
    name = "invoke_authorized_tool"
    description = "使用已发现工具的完整 JSON Schema 调用；权限、审批与写入账本仍由工具网关检查。"
    orchestration = True
    timeout_seconds = 90
    args_schema = {"type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["name", "arguments"], "additionalProperties": False}
    def __init__(self, tools):
        from agent_platform.tools.gateway import ToolGateway
        self.gateway = ToolGateway(tools=tools)
    async def execute(self, name, arguments):
        ctx = execution_context.get()
        return await self.gateway.execute(name, arguments, ctx.role)
