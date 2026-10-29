"""用户插件工具：把 SandboxedPlugin（子进程 + JSON IPC 隔离）适配为 BaseTool，接入调用链。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from agent_platform.tools.base import BaseTool
from agent_platform.tools.gateway import ToolError
from agent_platform.tools.sandbox import SandboxedPlugin


class PluginTool(BaseTool):
    read_only = False
    idempotent = False
    """沙箱插件包装：execute 走子进程 JSON IPC，隔离崩溃与超时。"""

    def __init__(
        self,
        name: str,
        description: str,
        command: list[str],
        args_schema: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.required_scopes = (f"plugin:{name}",)
        self.description = description
        self.args_schema = args_schema or {"type": "object", "properties": {}, "required": []}
        self._plugin = SandboxedPlugin(command, timeout=timeout)

    async def execute(self, **kwargs: Any) -> str:
        result = await self._plugin.run({"action": "query", "args": kwargs})
        if result.get("ok") is False:
            raise ToolError(
                "PLUGIN_ERROR",
                str(result.get("error", "插件执行失败")),
                retryable=False,
            )
        return json.dumps(result, ensure_ascii=False)


def scan_plugins(plugin_dir: str) -> list[PluginTool]:
    """扫描插件目录，每个含 plugin.json 的子目录注册为一个 PluginTool。

    plugin.json 格式：
      {"name", "description", "command": ["python", "plugin.py"], "args_schema"}
    命令中的相对路径以插件目录为基准解析为绝对路径。
    """
    if not plugin_dir:
        return []
    root = Path(plugin_dir)
    if not root.is_dir():
        logger.warning("插件目录不存在: {}", plugin_dir)
        return []
    plugins: list[PluginTool] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        spec_file = entry / "plugin.json"
        if not spec_file.is_file():
            continue
        try:
            spec = json.loads(spec_file.read_text(encoding="utf-8"))
            command = [str(c) for c in (spec.get("command") or ["python", "plugin.py"])]
            for i, arg in enumerate(command):
                candidate = entry / arg
                if candidate.exists():
                    command[i] = str(candidate.resolve())
            plugins.append(
                PluginTool(
                    name=spec.get("name", entry.name),
                    description=spec.get("description", ""),
                    command=command,
                    args_schema=spec.get("args_schema"),
                )
            )
            logger.info("发现插件: {}", spec.get("name", entry.name))
        except Exception as exc:  # noqa: BLE001
            logger.warning("插件 {} 加载失败: {}", entry.name, exc)
    return plugins
