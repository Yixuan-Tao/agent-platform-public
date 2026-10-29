"""Official MCP SDK adapter with explicit local trust policy and pagination."""
from contextlib import AsyncExitStack
from typing import Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from agent_platform.tools.base import BaseTool
from agent_platform.tools.result import ToolResult


class SDKClient:
    def __init__(self, config):
        self.config, self.stack, self.session = config, AsyncExitStack(), None
        self.capabilities = None

    async def start(self):
        config = self.config
        if config.get("transport", "stdio") == "stdio":
            command = config["command"]
            transport = await self.stack.enter_async_context(stdio_client(StdioServerParameters(
                command=command[0], args=command[1:], env=config.get("env"))))
        else:
            transport = await self.stack.enter_async_context(streamable_http_client(config["url"]))
        self.session = await self.stack.enter_async_context(ClientSession(transport[0], transport[1]))
        self.capabilities = (await self.session.initialize()).capabilities

    async def tools(self):
        cursor, items, seen = None, [], set()
        while True:
            page = await self.session.list_tools(cursor=cursor)
            items.extend(page.tools)
            cursor = page.nextCursor
            if not cursor:
                return items
            if cursor in seen:
                raise RuntimeError("MCP pagination cursor repeated")
            seen.add(cursor)

    async def close(self):
        await self.stack.aclose()


class SDKTool(BaseTool):
    def __init__(self, client, remote, policy):
        self.client, self.remote = client, remote
        self.name = f"mcp__{client.config['name']}__{remote.name}"
        self.description = remote.description or remote.name
        self.args_schema = remote.inputSchema
        self.output_schema = None  # MCP outputSchema validates structuredContent, not the enclosing envelope.
        # Remote annotations/descriptions never grant local permission or idempotency.
        self.required_role = policy.get("required_role", "user")
        self.required_scopes = tuple(policy.get("scopes", [f"mcp:{client.config['name']}:{remote.name}"]))
        self.read_only = policy.get("read_only", False)
        self.idempotent = policy.get("idempotent", False)

    async def execute(self, **arguments):
        result = await self.client.session.call_tool(self.remote.name, arguments=arguments)
        if self.remote.outputSchema and not result.isError:
            from jsonschema import Draft202012Validator
            Draft202012Validator(self.remote.outputSchema).validate(result.structuredContent)
        return ToolResult(status="failed" if result.isError else "success",
                          data={"structuredContent": result.structuredContent,
                                "content": [block.model_dump(mode="json") for block in result.content]},
                          error_code="MCP_IS_ERROR" if result.isError else "")


async def register_sdk_servers(registry, configs):
    clients = []
    try:
        for config in configs:
            if not config.get("trusted") or not isinstance(config.get("tool_permissions"), dict):
                raise ValueError("MCP 服务必须登记 trusted 与 tool_permissions")
            client = SDKClient(config)
            clients.append(client)
            await client.start()
            for remote in await client.tools():
                if remote.name in config["tool_permissions"]:
                    registry.register(SDKTool(client, remote, config["tool_permissions"][remote.name]))
        return clients
    except BaseException:
        for client in reversed(clients):
            await client.close()
        raise
