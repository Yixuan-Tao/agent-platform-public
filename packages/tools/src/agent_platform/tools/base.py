"""工具契约与注册表。"""

from __future__ import annotations

import inspect
import json
import threading
from importlib import import_module
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine

from loguru import logger


class BaseTool(ABC):
    """所有 Agent 工具的基类。

    - name: 工具名（唯一，小写下划线）
    - description: 给 LLM 看的自然语言说明
    - args_schema: JSON Schema 形式的参数说明（供 MCP/LLM 使用）
    """

    name: str = ""
    description: str = ""
    args_schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    # 调用所需的最低角色；空字符串 = 任何调用方可用（默认）
    required_role: str = ""
    read_only: bool = True
    idempotent: bool = True
    required_scopes: tuple[str, ...] = ()
    output_schema: dict[str, Any] | None = None
    orchestration: bool = False
    timeout_seconds: float | None = None

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """执行工具，返回给 LLM 的文本结果。"""
        raise NotImplementedError

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema,
            "read_only": self.read_only,
            "idempotent": self.idempotent,
            "required_scopes": list(self.required_scopes),
            "required_role": self.required_role,
            "output_schema": self.output_schema,
            "timeout_seconds": self.timeout_seconds,
        }


class ToolRegistry:
    """线程安全的工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._lock = threading.Lock()

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError("工具 name 不能为空")
        with self._lock:
            if tool.name in self._tools:
                raise ValueError(f"工具已注册: {tool.name}")
            self._tools[tool.name] = tool
            logger.debug("注册工具: {}", tool.name)

    def get(self, name: str) -> BaseTool:
        with self._lock:
            if name not in self._tools:
                raise KeyError(f"工具不存在: {name}")
            return self._tools[name]

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._tools)

    def list(self) -> list[BaseTool]:
        with self._lock:
            return list(self._tools.values())

    def manifests(self) -> list[dict[str, Any]]:
        return [t.manifest() for t in self.list()]

    def register_from_config(self, specs: list[dict[str, Any]]) -> int:
        """按声明式配置动态注册工具。

        每个 spec：{"name", "description", "module", "function"}，
        function 为 async 函数，签名类型注解会转成 args_schema。
        返回成功注册数量；单个失败仅告警，不中断其余注册。
        """
        registered = 0
        for spec in specs or []:
            name = spec.get("name", "")
            module = spec.get("module", "")
            func_name = spec.get("function", "")
            if not name or not module or not func_name:
                logger.warning("工具配置缺失字段，跳过: {}", spec)
                continue
            try:
                func = getattr(import_module(module), func_name)
                tool = create_tool(
                        name=name,
                        description=spec.get("description", ""),
                        func=func,
                    )
                tool.read_only = bool(spec.get("read_only", False))
                tool.idempotent = bool(spec.get("idempotent", False))
                tool.required_scopes = tuple(spec.get("required_scopes", ["tool:" + name]))
                tool.required_role = str(spec.get("required_role", "user"))
                self.register(tool)
                registered += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("动态注册工具失败 {}: {}", name, exc)
        return registered


def create_tool(name: str, description: str, func: Callable[..., Coroutine]) -> BaseTool:
    """把 async 函数包装为 BaseTool。

    函数签名里的类型注解会转成 args_schema 的 properties（简化为 type/description）。
    """

    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "kwargs"):
            continue
        ann = param.annotation
        ptype = "string"
        if ann is int:
            ptype = "integer"
        elif ann is float:
            ptype = "number"
        elif ann is bool:
            ptype = "boolean"
        properties[pname] = {"type": ptype, "description": f"参数 {pname}"}
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    class _Wrapped(BaseTool):
        async def execute(self, **kwargs: Any) -> str:
            result = await func(**kwargs)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)

    _Wrapped.name = name
    _Wrapped.description = description
    _Wrapped.args_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    return _Wrapped()
