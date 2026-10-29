"""工具调用运行时网关：调用前校验 + ACL 授权 + 超时/重试/熔断 + 结构化错误。"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Iterable

from loguru import logger
from pydantic import Field, ValidationError, create_model

from agent_platform.tools.base import BaseTool, ToolRegistry
from agent_platform.tools.result import ToolResult
from agent_platform.core.execution import BudgetExceeded, execution_context, is_graph_interrupt, reserve
from jsonschema import Draft202012Validator, ValidationError as SchemaError

_ROLE_LEVEL = {"service": 2, "viewer": 1, "user": 2, "admin": 3}


class ToolError(Exception):
    """工具调用错误（带结构化错误码）。"""

    def __init__(self, error_code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.retryable = retryable


class ToolValidationError(ToolError):
    def __init__(self, message: str) -> None:
        super().__init__("INVALID_ARGS", message, retryable=False)


class ToolAuthorizationError(ToolError):
    def __init__(self, message: str) -> None:
        super().__init__("FORBIDDEN", message, retryable=False)


class CircuitBreaker:
    """工具层本地熔断器（独立实现，避免 tools→core 依赖）。"""

    def __init__(self, threshold: int = 5, reset_seconds: int = 60) -> None:
        self.threshold = threshold
        self.reset_seconds = reset_seconds
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    async def check(self) -> bool:
        async with self._lock:
            if self._opened_at is None:
                return True
            if time.time() - self._opened_at >= self.reset_seconds:
                self._opened_at = None
                self._failures = 0
                return True
            return False

    async def record_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._opened_at = None

    async def record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            if self._failures >= self.threshold:
                self._opened_at = time.time()


def _map_type(t: str) -> Any:
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        return list
    return str


def _schema_to_model(name: str, schema: dict[str, Any]) -> Any:
    """把 JSON Schema 参数定义转成 pydantic 模型（调用前校验用）。"""
    props = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: dict[str, Any] = {}
    for pname, pinfo in props.items():
        ptype = _map_type(pinfo.get("type", "string"))
        if pname in required:
            fields[pname] = (ptype, Field(description=pinfo.get("description", "")))
        else:
            fields[pname] = (ptype | None, Field(default=None, description=pinfo.get("description", "")))
    return create_model(name, **fields)


class ToolGateway:
    """统一工具执行入口。

    用法：优先用 execute() 返回结构化结果
      {"ok": true, "result": ...}
    或 {"ok": false, "error_code", "message", "retryable"}。
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        tools: Iterable[BaseTool] | None = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        breaker_threshold: int = 5,
        breaker_reset_seconds: int = 60,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        if registry is not None:
            for tool in registry.list():
                self._tools[tool.name] = tool
        for tool in tools or []:
            self._tools[tool.name] = tool
        self._models: dict[str, Any] = {}
        self.timeout = timeout
        self.max_retries = max_retries
        self._breaker = CircuitBreaker(breaker_threshold, breaker_reset_seconds)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise ToolError("TOOL_NOT_FOUND", f"工具不存在: {name}")
        return self._tools[name]

    def _model_for(self, name: str) -> Any:
        if name not in self._models:
            self._models[name] = _schema_to_model(name, self.get(name).args_schema)
        return self._models[name]

    def validate(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """调用前 schema 校验；失败抛 ToolValidationError。"""
        try:
            schema = dict(self.get(name).args_schema)
            schema.setdefault("additionalProperties", False)
            Draft202012Validator(schema).validate(args)
            return dict(args)
        except SchemaError as exc:
            raise ToolValidationError(f"参数校验失败: {exc.message}") from exc

    def authorize(self, name: str, principal_role: str = "user") -> None:
        """ACL 授权：required_role 为空则放行，否则按角色层级校验。"""
        required = self.get(name).required_role or ""
        ctx = execution_context.get()
        if ctx:
            if not self.get(name).read_only and not ctx.run_id:
                raise ToolAuthorizationError("写动作需要可持久化的任务审批；请通过任务工作台提交")
            if ctx.permitted_tools is not None and name not in ctx.permitted_tools:
                raise ToolAuthorizationError(f"Skill 未授权工具: {name}")
            if not set(self.get(name).required_scopes).issubset(ctx.tool_scopes):
                raise ToolAuthorizationError(f"缺少工具权限: {name}")
        if not required:
            return
        level = _ROLE_LEVEL.get(principal_role or "user", 0)
        need = _ROLE_LEVEL.get(required, 1)
        if level < need:
            raise ToolAuthorizationError(
                f"角色 {principal_role or 'user'} 无权限调用 {name}（需要角色 {required}）"
            )

    async def _execute_once(self, name: str, args: dict[str, Any]) -> str:
        ctx = execution_context.get()
        semaphore = ctx.resources.get("tool_semaphore") if ctx and ctx.resources else None
        if self.get(name).orchestration:
            semaphore = None  # Delegated calls acquire the real resource slot themselves.
        lock = ctx.resources.get("write_lock") if ctx and ctx.resources and not self.get(name).read_only else None
        async def invoke():
            if ctx and ctx.tool_start:
                await ctx.tool_start(name, args, self.get(name))
            return await self.get(name).execute(**args)
        if semaphore:
            async with semaphore:
                if lock:
                    async with lock:
                        return await invoke()
                return await invoke()
        return await invoke()

    async def execute(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        principal_role: str = "user",
    ) -> dict[str, Any]:
        """执行工具，返回结构化结果；任何失败都返回 dict 而非抛异常。"""
        args = args or {}
        try:
            validated = self.validate(name, args)
            self.authorize(name, principal_role)
        except ToolError as exc:
            return {
                "ok": False, "name": name,
                "error_code": exc.error_code,
                "message": exc.message,
                "retryable": exc.retryable,
                "status": "failed", "data": None, "evidence_refs": [], "external_refs": [],
            }

        if not await self._breaker.check():
            return {
                "ok": False, "name": name,
                "error_code": "CIRCUIT_OPEN",
                "message": "熔断器开启，暂缓工具调用",
                "retryable": True,
                "status": "failed", "data": None, "evidence_refs": [], "external_refs": [],
            }

        ctx = execution_context.get()
        cached = await ctx.tool_before(name, validated, self.get(name)) if ctx and ctx.tool_before else None
        if cached is not None:
            return cached
        last: BaseException | None = None
        tool = self.get(name)
        retries = self.max_retries if tool.read_only or tool.idempotent else 0
        for attempt in range(retries + 1):
            try:
                await reserve("tool_attempts")
                result = await asyncio.wait_for(
                    self._execute_once(name, validated), timeout=tool.timeout_seconds or self.timeout
                )
                await self._breaker.record_success()
                if hasattr(result, "tool_result"):
                    normalized = result.tool_result.model_dump()
                elif isinstance(result, ToolResult):
                    normalized = result.model_dump()
                elif isinstance(result, dict) and "status" in result:
                    normalized = ToolResult.model_validate(result).model_dump()
                else:
                    normalized = ToolResult(status="success", data=result).model_dump()
                if tool.output_schema and normalized["status"] == "success":
                    Draft202012Validator(tool.output_schema).validate(normalized["data"])
                output = {"ok": normalized["status"] == "success", "name": name,
                          "result": result.text() if isinstance(result, ToolResult) else result,
                          **normalized}
                if ctx and ctx.tool_after:
                    await ctx.tool_after(name, validated, output)
                return output
            except asyncio.TimeoutError:
                last = asyncio.TimeoutError()
                if attempt < retries:
                    continue
            except ToolError as exc:
                last = exc
                if exc.retryable and attempt < retries:
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                if is_graph_interrupt(exc) or isinstance(exc, BudgetExceeded):
                    raise
                last = exc
                break

        await self._breaker.record_failure()
        if isinstance(last, asyncio.TimeoutError):
            output = {
                "ok": False, "name": name,
                "status": "failed" if tool.read_only else "unknown",
                "data": None, "evidence_refs": [], "external_refs": [],
                "error_code": "TIMEOUT",
                "message": f"工具调用超时（>{self.timeout}s）",
                "retryable": bool(tool.read_only or tool.idempotent),
            }
        elif isinstance(last, ToolError):
            output = {
                "ok": False, "name": name,
                "error_code": last.error_code,
                "message": last.message,
                "retryable": last.retryable,
            }
        else:
            logger.warning("工具 {} 执行失败: {}", name, last)
            output = {
            "ok": False, "name": name,
            "error_code": "INTERNAL",
            "message": str(last),
            "retryable": False,
        }
        output.setdefault("status", "failed" if tool.read_only else "unknown")
        output.setdefault("data", None)
        output.setdefault("evidence_refs", [])
        output.setdefault("external_refs", [])
        if ctx and ctx.tool_after:
            await ctx.tool_after(name, validated, output)
        return output

    async def execute_many(
        self,
        calls: list[dict[str, Any]],
        principal_role: str = "user",
    ) -> list[dict[str, Any]]:
        """并行执行多个独立调用，逐调用隔离错误。"""

        async def _one(call: dict[str, Any]) -> dict[str, Any]:
            return await self.execute(
                call.get("name", ""), call.get("arguments"), principal_role
            )

        semaphore = asyncio.Semaphore(3)
        async def limited(call):
            async with semaphore:
                return await _one(call)
        return await asyncio.gather(*(limited(c) for c in calls or []))
