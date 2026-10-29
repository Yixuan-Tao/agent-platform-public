"""用户插件沙箱：子进程 + JSON IPC，隔离崩溃。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger


class SandboxError(RuntimeError):
    pass


class SandboxedPlugin:
    """在子进程中运行用户插件。

    用法：
        plugin = SandboxedPlugin(["python", "plugin.py"])
        result = await plugin.run({"action": "query", "args": ...})
    """

    def __init__(self, command: list[str], timeout: float = 30.0) -> None:
        self.command = command
        self.timeout = timeout

    async def run(self, input_data: dict[str, Any] | str) -> dict[str, Any]:
        """向子进程 stdin 写入 JSON，解析 stdout 的 JSON 返回。"""
        payload = input_data if isinstance(input_data, str) else json.dumps(input_data, ensure_ascii=False)
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise SandboxError(f"插件进程启动失败（命令不存在）: {self.command}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(payload.encode("utf-8")), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise SandboxError(f"插件执行超时（>{self.timeout}s），已终止")
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")[:2000]
            logger.warning("插件子进程退出码 {}: {}", proc.returncode, err)
            return {"ok": False, "error": err, "returncode": proc.returncode}
        try:
            return json.loads(stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"插件输出不是合法 JSON: {exc}", "raw": stdout.decode("utf-8", errors="replace")[:2000]}
