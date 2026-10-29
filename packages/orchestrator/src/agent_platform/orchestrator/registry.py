"""AgentRegistry：注册 Manifest + 图构建器，生成路由表。"""

from __future__ import annotations

import threading
from typing import Any, Callable

from loguru import logger

from agent_platform.core.models import AgentManifest

AgentBuilder = Callable[..., Any]


class AgentRegistry:
    def __init__(self) -> None:
        self._manifests: dict[str, AgentManifest] = {}
        self._builders: dict[str, AgentBuilder] = {}
        self._graphs: dict[str, Any] = {}
        self._lock = threading.Lock()

    def register(self, manifest: AgentManifest, builder: AgentBuilder) -> None:
        with self._lock:
            if manifest.name in self._manifests:
                raise ValueError(f"Agent 已注册: {manifest.name}")
            self._manifests[manifest.name] = manifest
            self._builders[manifest.name] = builder
            logger.info("注册 Agent: {} ({})", manifest.name, manifest.description[:40])

    def get_manifest(self, name: str) -> AgentManifest:
        with self._lock:
            if name not in self._manifests:
                raise KeyError(f"Agent 不存在: {name}")
            return self._manifests[name]

    def get_builder(self, name: str) -> AgentBuilder:
        with self._lock:
            if name not in self._builders:
                raise KeyError(f"Agent 不存在: {name}")
            return self._builders[name]

    def list_manifests(self) -> list[AgentManifest]:
        with self._lock:
            return [m for m in self._manifests.values() if m.enabled]

    def names(self) -> list[str]:
        return [m.name for m in self.list_manifests()]

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._manifests

    def build(self, name: str, llm_router: Any, tool_registry: Any) -> Any:
        """构建（并缓存）Agent 图。"""
        from agent_platform.core.execution import execution_context, fingerprint
        ctx = execution_context.get()
        settings = getattr(llm_router, "settings", None)
        from pathlib import Path
        import inspect
        builder = self.get_builder(name)
        source = inspect.getsourcefile(builder)
        prompts = {}
        if source:
            for path in Path(source).parent.rglob("*.py"):
                prompts[path.name] = fingerprint(path.read_text(encoding="utf-8"))
        config = settings.model_dump(exclude={"llm_api_key", "api_key", "admin_api_key", "rag_service_token", "jira_token", "initial_admin_password", "langfuse_secret_key"}) if hasattr(settings, "model_dump") else {"model": llm_router.resolve_model(name) if hasattr(llm_router, "resolve_model") else "unconfigured"}
        key = fingerprint({"agent": self.get_manifest(name).model_dump(), "prompts": prompts, "config": config, "tools": tool_registry.manifests() if tool_registry else [], "profile": ctx.profile if ctx else {}, "run": ctx.run_id if ctx and ctx.checkpointer else ""})
        with self._lock:
            if key in self._graphs:
                return self._graphs[key]
        builder = self.get_builder(name)
        graph = builder(llm_router, tool_registry)
        with self._lock:
            if len(self._graphs) >= 128:
                self._graphs.clear()
            self._graphs[key] = graph
        return graph


_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry
