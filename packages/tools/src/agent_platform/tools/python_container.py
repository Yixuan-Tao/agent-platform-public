"""Fail-closed Docker execution; never substitutes an ordinary Python subprocess."""
from __future__ import annotations
import asyncio
import shutil
import uuid
from pathlib import Path
from agent_platform.tools.base import BaseTool
from agent_platform.tools.result import ToolResult
from agent_platform.core.execution import execution_context


async def accept_environment(image="agent-platform-python:1"):
    """Opt-in startup acceptance; pin the accepted local image ID, never pull."""
    if shutil.which("docker") is None:
        raise RuntimeError("受限 Python 需要可用 Docker daemon 与预构建镜像")
    async def inspect(*arguments):
        process = await asyncio.create_subprocess_exec("docker", *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            output, _ = await asyncio.wait_for(process.communicate(), 10)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError("Docker 环境验收超时") from None
        if process.returncode:
            raise RuntimeError("Docker daemon 或受限 Python 镜像不可用")
        return output.decode().strip()
    await inspect("info", "--format", "{{.OSType}}")
    image_id = await inspect("image", "inspect", "--format", "{{.Id}}", image)
    if not image_id.startswith("sha256:"):
        raise RuntimeError("Docker 镜像 ID 无效")
    return image_id


class ContainerPythonTool(BaseTool):
    name = "restricted_python"
    description = "在离线受限容器处理本任务数据，支持 numpy、pandas、matplotlib；不允许外部动作。"
    args_schema = {"type": "object", "properties": {"code": {"type": "string", "maxLength": 64000}, "inputs": {"type": "object", "additionalProperties": {"type": "string", "maxLength": 1000000}}}, "required": ["code"], "additionalProperties": False}
    required_scopes = ("python:execute",)
    timeout_seconds = 75  # Includes Docker startup and cleanup around the 60-second execution bound.
    def __init__(self, root, image="agent-platform-python:1"):
        self.root, self.image = Path(root).resolve(), image

    def command(self, name, incoming, outgoing):
        for path in (incoming, outgoing):
            if not path.resolve().is_relative_to(self.root) or path.is_symlink():
                raise ValueError("任务目录越界")
        return ["docker", "run", "--name", name, "--rm", "--pull=never", "--network=none", "--read-only",
                "--user=65532:65532", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=32",
                "--cpus=1", "--memory=512m", "--memory-swap=512m", "--tmpfs=/tmp:rw,noexec,nosuid,size=32m",
                "--mount", f"type=bind,src={incoming},dst=/input,readonly",
                "--mount", f"type=bind,src={outgoing},dst=/output", "--workdir=/output", self.image,
                "timeout", "--signal=KILL", "60s", "python", "-I", "/input/main.py"]

    async def execute(self, code, inputs=None):
        if shutil.which("docker") is None:
            return ToolResult(status="failed", error_code="DOCKER_UNAVAILABLE")
        ctx = execution_context.get()
        if not ctx or not ctx.run_id:
            return ToolResult(status="failed", error_code="RUN_REQUIRED")
        if sum(len(value.encode("utf-8")) for value in (inputs or {}).values()) + len(code.encode("utf-8")) > 10 * 1024 * 1024:
            return ToolResult(status="failed", error_code="INPUT_LIMIT")
        task = self.root / ctx.run_id / uuid.uuid4().hex
        incoming, outgoing = task / "input", task / "output"
        incoming.mkdir(parents=True)
        outgoing.mkdir()
        outgoing.chmod(0o777)  # private per-run mount must be writable by container uid 65532
        for name, content in (inputs or {}).items():
            if Path(name).name != name or name in {"main.py", ".", ".."}:
                raise ValueError("输入只允许普通文件名")
            (incoming / name).write_text(content, encoding="utf-8")
        (incoming / "main.py").write_text(code, encoding="utf-8")
        name = "ap-python-" + uuid.uuid4().hex
        process = None
        monitor = None
        def check_outputs():
            total = 0
            for path in outgoing.rglob("*"):
                if path.is_symlink() or not path.resolve().is_relative_to(outgoing.resolve()):
                    raise ValueError("输出包含非法链接")
                if path.is_file():
                    total += path.stat().st_size
            if total > 10 * 1024 * 1024:
                raise ValueError("产物总量超过10MiB")
        async def watch_outputs():
            while True:
                await asyncio.sleep(.1)
                await asyncio.to_thread(check_outputs)
        async def bounded(pipe):
            output = bytearray()
            while block := await pipe.read(8192):
                output.extend(block)
                if len(output) > 65536:
                    raise ValueError("文本输出超过64KiB")
            return bytes(output)
        try:
            process = await asyncio.create_subprocess_exec(*self.command(name, incoming, outgoing),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            monitor = asyncio.create_task(watch_outputs())
            reader = asyncio.create_task(bounded(process.stdout))
            done, _ = await asyncio.wait({monitor, reader}, timeout=60, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
                raise asyncio.TimeoutError("容器执行超过60秒")
            if monitor in done:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
                await monitor
            text = await reader
            await asyncio.wait_for(process.wait(), 5)
            if process.returncode:
                return ToolResult(status="failed", data={"output": text.decode(errors="replace")}, error_code="CONTAINER_FAILED")
            files, total = [], 0
            for path in outgoing.rglob("*"):
                if path.is_symlink() or not path.resolve().is_relative_to(outgoing.resolve()):
                    raise ValueError("输出包含非法链接")
                if path.is_file():
                    total += path.stat().st_size
                    if total > 10 * 1024 * 1024:
                        raise ValueError("产物总量超过10MiB")
                    files.append(path)
            artifacts = []
            for path in files:
                artifacts.append(await ctx.resources["store"].artifact(ctx.run_id, ctx.resources["fence"], path.name, path.read_bytes(), "application/octet-stream"))
            return ToolResult(status="success", data={"output": text.decode(errors="replace"), "artifacts": artifacts})
        except (OSError, asyncio.TimeoutError, ValueError) as exc:
            return ToolResult(status="failed", error_code=type(exc).__name__.upper(), data=str(exc))
        finally:
            if monitor:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
            try:
                cleanup = await asyncio.create_subprocess_exec("docker", "rm", "-f", name,
                             stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                try:
                    await asyncio.wait_for(cleanup.wait(), 10)
                except asyncio.TimeoutError:
                    cleanup.kill()
                    await cleanup.wait()
            except OSError:
                pass
            if process and process.returncode is None:
                process.kill()
                await process.wait()
