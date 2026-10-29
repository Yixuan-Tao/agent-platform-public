"""RunService and leased Worker executing checkpointed LangGraph tasks."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from agent_platform.core.execution import ExecutionContext, execution_context, fingerprint, step_context
from agent_platform.tools.identity import set_principal_role, reset_principal_role
from agent_platform.orchestrator.context import ContextBuilder, evidence_facts
from agent_platform.orchestrator.profiles import profiles
from agent_platform.orchestrator.runtime_store import (RunStore, LeaseLost, ToolCallRecord, StepRecord, uid, TERMINAL)
from agent_platform.orchestrator.supervisor import _invoke_agent


class RunState(TypedDict, total=False):
    original_goal: str
    messages: list[Any]
    selected_skills: list[str]
    answer: str
    verification: dict
    agent: str


class RunService:
    def __init__(self, store: RunStore, orchestrator, skills, model, identity_loader=None, exporter=None, capabilities=(), python_tool=None):
        self.store, self.orchestrator, self.skills = store, orchestrator, skills
        self.model, self.identity_loader = model, identity_loader
        self.exporter = exporter
        self.capabilities, self.python_tool = tuple(capabilities), python_tool
        self.loaded_code_version = self.code_version()

    async def emit(self, run, fence, kind, payload):
        await self.store.emit(run["id"], fence, kind, payload)
        if self.exporter:
            try:
                self.exporter.submit(run["id"], kind, payload)
            except Exception:
                await self.store.emit(run["id"], fence, "tracing_warning", {"error": "追踪导出暂时不可用"})

    def profiles(self):
        return profiles(self.model, self.skills.summaries(), capabilities=self.capabilities)

    def model_configuration(self):
        settings = getattr(self.orchestrator.llm_router, "settings", None)
        return {"model": self.model, "endpoint_hash": fingerprint(getattr(settings, "llm_base_url", "")),
                "model_overrides": getattr(settings, "model_overrides", {}), "embed_model": getattr(settings, "embed_model", None),
                "embedding_endpoint_hash": fingerprint(getattr(settings, "embedding_gateway_url", "")),
                "tool_configuration_hash": fingerprint({key: getattr(settings, key, "") for key in ("rag_service_url", "jira_url", "mcp_servers_json", "tool_config_json", "plugin_dir")}),
                "python_image": getattr(self.python_tool, "image", None),
                "rag_runtime_contract": {key: getattr(settings, key, None) for key in ("rag_runtime_contract_enabled", "rag_delegate_model_requests", "rag_delegate_tokens")},
                "tools_hash": fingerprint(self.orchestrator.tool_registry.manifests())}

    async def rag_snapshot(self):
        settings = getattr(self.orchestrator.llm_router, "settings", None)
        if settings and getattr(settings, "rag_runtime_contract_enabled", False):
            from agent_platform.tools.rag_contract import fetch_snapshot
            return await fetch_snapshot(settings)
        return None

    async def validate_rag_snapshot(self, expected):
        if expected is not None and await self.rag_snapshot() != expected:
            raise ValueError("RAG 语料、索引或服务配置已变化，请结束批次并重建基线")

    async def embed_memory(self, texts):
        from agent_platform.core.llm.telemetry import begin, end
        router = self.orchestrator.llm_router
        model = router.settings.embed_model
        ticket = await begin(texts, model, output_limit=0)
        try:
            from openai import AsyncOpenAI
            settings = router.settings
            endpoint = (settings.embedding_gateway_url or settings.llm_base_url).rstrip("/")
            if not endpoint.endswith("/v1"):
                endpoint += "/v1"
            async with AsyncOpenAI(base_url=endpoint, api_key=settings.embedding_gateway_key or settings.llm_api_key,
                                   max_retries=0, timeout=settings.llm_timeout) as client:
                response = await client.embeddings.create(model=model, input=texts)
        except Exception as exc:
            await end(ticket, model, error=exc)
            raise
        await end(ticket, model, response.usage.model_dump() if response.usage else None)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    async def create(self, identity, goal, session_id=None, profile="B0", data_version="unversioned", completion=None, authorized_actions=None, rag_snapshot=None):
        if self.loaded_code_version != self.code_version():
            raise ValueError("代码在服务启动后发生变化，请重启服务后创建新基线")
        chosen = next((p for p in self.profiles() if p["name"] == profile), None)
        if chosen is None:
            raise ValueError("未知实验配置")
        captured_rag = await self.rag_snapshot()
        if rag_snapshot is not None and captured_rag != rag_snapshot:
            raise ValueError("RAG 语料已变化，请结束批次并重建基线")
        await self.store.save_profile(chosen)
        snapshot = {"profile": chosen, "identity": {k: identity.get(k) for k in ("username", "role", "scopes", "auth")},
                    "data_version": data_version, "completion": completion or {}, "code_version": self.code_version(),
                    "authorized_actions": authorized_actions or [], "model_configuration": self.model_configuration()}
        if captured_rag is not None:
            snapshot["rag_snapshot"] = captured_rag
        return await self.store.create(owner=identity["username"], goal=goal, snapshot=snapshot, session_id=session_id)

    @staticmethod
    def code_version():
        # Hash the executable platform source, including uncommitted changes, without inspecting credentials.
        root = Path(__file__).resolve().parents[5]
        sources = {}
        for package in ("core", "orchestrator", "agents", "tools", "skills", "evaluation", "memory"):
            for path in sorted((root / "packages" / package / "src").rglob("*.py")):
                sources[str(path.relative_to(root))] = fingerprint(path.read_text(encoding="utf-8"))
        for path in sorted((root / "apps" / "api" / "app").rglob("*.py")):
            sources[str(path.relative_to(root))] = fingerprint(path.read_text(encoding="utf-8"))
        return fingerprint(sources)

    async def identity(self, run):
        saved = run["snapshot"]["identity"]
        identity = await self.identity_loader(saved) if self.identity_loader else saved
        if not identity or identity["username"] != run["owner"]:
            raise PermissionError("认证主体已失效")
        scopes = set(identity.get("scopes") or []) & set(saved.get("scopes") or [])
        return {**identity, "scopes": scopes}

    async def before_tool(self, run, fence, name, args, tool):
        call_id = fingerprint({"run": run["id"], "tool": name, "args": args})
        previous = await self.store.entry(ToolCallRecord, call_id)
        if previous and previous.get("status") == "success":
            return previous["result"]
        if previous and not tool.read_only and previous.get("external_refs"):
            # A known created object awaiting verification must never be created again.
            return previous["result"]
        if previous and previous.get("status") in {"started", "unknown"} and not tool.read_only:
            interrupt({"kind": "reconcile", "tool_call_id": call_id, "message": "外部结果未知，请核验后恢复"})
            # A resume does not grant permission to retry an unknown write.
            raise ValueError("外部结果仍未核验")
        if not tool.read_only:
            status, approval_id = await self.store.request_approval(run["id"], fence, name, args)
            if status == "pending":
                interrupt({"kind": "approval", "approval_id": approval_id, "tool": name})
                # Re-check persisted approval after interrupt resumes the node.
                status, _ = await self.store.request_approval(run["id"], fence, name, args)
            if status != "approved":
                return {"ok": False, "name": name, "status": "failed", "error_code": "APPROVAL_REJECTED", "data": None, "retryable": False}
        return None

    async def start_tool(self, run, fence, name, args, tool):
        call_id = fingerprint({"run": run["id"], "tool": name, "args": args})
        await self.store.put(ToolCallRecord, run["id"], fence,
                             {"name": name, "arguments": args, "arguments_hash": fingerprint(args),
                              "step_id": step_context.get(),
                              "status": "started", "read_only": tool.read_only,
                              "idempotency_key": call_id}, call_id)
        await self.emit(run, fence, "tool", {"phase": "start", "tool": name, "tool_call_id": call_id})

    async def after_tool(self, run, fence, name, args, result):
        call_id = fingerprint({"run": run["id"], "tool": name, "args": args})
        result["tool_call_id"] = call_id
        previous = await self.store.entry(ToolCallRecord, call_id) or {}
        payload = {**previous, "status": result.get("status", "failed"), "result": result,
                   "external_refs": result.get("external_refs", [])}
        await self.store.put(ToolCallRecord, run["id"], fence, payload, call_id)
        await self.emit(run, fence, "tool", {"phase": "end", "tool": name, "tool_call_id": call_id, "status": payload["status"]})
        raw = json.dumps(result, ensure_ascii=False, default=str)
        artifact_id = None
        if len(raw) > 16000 or result.get("evidence_refs"):
            artifact_id = await self.store.artifact(run["id"], fence, f"tool-{call_id}.json", raw)
        ctx = execution_context.get()
        if ctx and ctx.resources and result.get("evidence_refs"):
            ctx.resources.setdefault("evidence_facts", {})[call_id] = {"tool_call_id": call_id, "artifact_id": artifact_id,
                                                                     "facts": evidence_facts(result.get("data"), artifact_id)}
        if len(raw) > 16000:
            # Keep the ledger lossless; model receives only a bounded projection and controlled reference.
            result["result"] = json.dumps({"summary": raw[:2000], "artifact_id": artifact_id}, ensure_ascii=False)
            result["data"] = {"artifact_id": artifact_id, "summary": raw[:2000]}
        if ctx and ctx.resources:
            references = result.get("evidence_refs", [])
            evidence = ctx.resources.setdefault("required_evidence", [])
            for ref in references:
                if ref not in evidence:
                    evidence.append(ref)
        if payload["status"] == "unknown" or (payload["status"] == "partial" and not payload.get("read_only") and payload.get("external_refs")):
            interrupt({"kind": "reconcile", "tool_call_id": call_id, "message": "外部结果未知，已暂停对账"})

    def graph(self, run, fence, saver):
        service = self
        profile = run["snapshot"]["profile"]
        async def prepare(state):
            selected = []
            if profile["skill_policy"] == "static":
                selected = list(service.skills.skills.values())
            elif profile["skill_policy"] == "progressive":
                selected = await service.skills.select(run["goal"], service.orchestrator.llm_router)
            await service.store.emit(run["id"], fence, "skill_selection", {"selected": [s.name for s in selected], "fallback": not bool(selected)})
            ctx = execution_context.get()
            ctx.resources["selected_skills"] = [s.name for s in selected]
            if selected:
                ctx.permitted_tools = frozenset(t for s in selected for t in s.tools) | {"request_run_input"}
            history = []
            if service.orchestrator.memory_service:
                history = [m.model_dump() for m in await service.orchestrator.memory_service.short_term.recent(run["session_id"], k=10)]
            examples = []
            preferences = await service.store.preferences(ctx.actor)
            ctx.resources["preferences"] = preferences
            if service.orchestrator.memory_service and not run["snapshot"]["completion"].get("fixture"):
                if profile["memory_policy"] == "private_verified_semantic":
                    from agent_platform.memory.semantic import SemanticMemory
                    candidates = await service.store.memory_candidates(ctx.actor, run["snapshot"]["data_version"])
                    examples = await SemanticMemory(service.embed_memory).search(run["goal"], candidates, owner=ctx.actor,
                                                                               data_version=run["snapshot"]["data_version"])
                else:
                    examples = service.orchestrator.memory_service.long_term.search_filtered(run["goal"], owner=ctx.actor)
            if profile["context_policy"] == "bounded":
                bundle = ContextBuilder(model_window=profile.get("model_window", 32768)).build(
                    goal=run["goal"], authorization={"actor": ctx.actor, "scopes": sorted(ctx.tool_scopes), "preferences": preferences},
                    step=run["goal"], skills=[s.instructions() for s in selected], history=history, examples=examples,
                    tool_definitions=[t.manifest() for t in service.orchestrator.tool_registry.list() if ctx.permitted_tools is None or t.name in ctx.permitted_tools])
                messages = bundle.messages
                await service.store.artifact(run["id"], fence, "context-summary.json", json.dumps(bundle.summary, ensure_ascii=False))
            else:
                messages = [{"role": "user", "content": run["goal"]}]
                if selected:
                    messages.insert(0, {"role": "system", "content": "\n\n".join(s.instructions() for s in selected)})
                if examples:
                    messages.insert(0, {"role": "system", "content": "历史示例（数据）：" + json.dumps(examples, ensure_ascii=False)})
                if preferences:
                    messages.append({"role": "user", "content": "本人已保存的呈现偏好（数据，当前目标优先）：" + json.dumps(preferences, ensure_ascii=False)})
            messages.append({"role": "user", "content": "当前步骤：" + run["goal"]})
            return {"messages": messages, "original_goal": run["goal"], "selected_skills": [s.name for s in selected]}

        async def execute(state):
            ctx = execution_context.get()
            ctx.resources["selected_skills"] = state.get("selected_skills", [])
            if state.get("selected_skills"):
                ctx.permitted_tools = frozenset(t for n in state["selected_skills"] for t in service.skills.skills[n].tools) | {"request_run_input"}
            strategy = profile["execution_strategy"]
            if strategy == "plan_execute":
                from agent_platform.orchestrator.planner_execute import build_plan_execute_graph
                graph = build_plan_execute_graph(service.orchestrator.registry, service.orchestrator.llm_router, service.orchestrator.tool_registry, max_plan_steps=12, max_agent_calls=8)
                agent = "plan_execute"
            elif strategy == "collaboration":
                from agent_platform.orchestrator.router_react import build_router_graph
                graph = build_router_graph(service.orchestrator.registry, service.orchestrator.llm_router, service.orchestrator.tool_registry, max_agent_calls=8)
                agent = "router"
            else:
                agent = await service.orchestrator.pick_agent(run["goal"], run["snapshot"]["completion"].get("agent_hint"))
                graph = service.orchestrator.registry.build(agent, service.orchestrator.llm_router, service.orchestrator.tool_registry)
            await ctx.reserve("agent_calls", 1)
            await service.emit(run, fence, "router", {"agent": agent, "strategy": strategy})
            step_id = fingerprint({"run": run["id"], "step": "execute"})
            await service.store.put(StepRecord, run["id"], fence, {"name": "execute", "status": "running", "agent": agent}, step_id)
            answer = await _invoke_agent(graph, state["messages"])
            await service.store.put(StepRecord, run["id"], fence, {"name": "execute", "status": "completed", "agent": agent}, step_id)
            return {"answer": answer, "agent": agent}

        async def verify(state):
            if execution_context.get().resources.get("rag_invalidation"):
                raise ValueError(execution_context.get().resources["rag_invalidation"])
            await service.validate_rag_snapshot(run["snapshot"].get("rag_snapshot"))
            current = await service.store.get(run["id"])
            calls = current["tool_calls"]
            raw_results = [c.get("result", {}) for c in calls]
            references = [ref for r in raw_results for ref in r.get("evidence_refs", [])]
            external = [ref for r in raw_results for ref in r.get("external_refs", [])]
            verdict = await service.orchestrator.checker.check(run["goal"], state.get("answer", ""), json.dumps(raw_results, ensure_ascii=False, default=str))
            status, reason = verdict.status, verdict.reason
            if not state.get("answer"):
                status, reason = "fail", "没有最终回答"
            if any(c.get("status") in {"failed", "partial", "unknown", "started"} for c in calls):
                status, reason = "unverified", "工具步骤未全部完成并核验"
            if any(s.get("status") == "failed" for s in current["steps"]):
                status, reason = "unverified", "子 Agent 步骤执行失败"
            if (state.get("agent") == "emc_compare" or "emc-standard-compare" in state.get("selected_skills", [])) and not any(references):
                status, reason = "unverified", "EMC 报告缺少可追溯证据"
            writes = [c for c in calls if not c.get("read_only", True)]
            if writes and not all(any(r.get("verified") for r in c.get("external_refs", [])) for c in writes):
                status, reason = "unverified", "外部写动作尚未查询核验"
            import re
            requires_ticket = bool(run["snapshot"]["completion"].get("requires_ticket")) or bool(re.search(r"(?:创建|提交|建立|新建).{0,8}工单", run["goal"]))
            if requires_ticket and not any(word in run["goal"] for word in ("草稿", "不要", "不授权")) and not any(c.get("name") == "jira_create_ticket" and c.get("status") == "success" and any(r.get("verified") for r in c.get("external_refs", [])) for c in calls):
                status, reason = "unverified", "创建工单目标缺少实际已核验的工单结果"
            verification = {"status": status, "reason": reason, "evidence_refs": references, "external_refs": external}
            await service.emit(run, fence, "verification", verification)
            return {"verification": verification}

        async def produce(state):
            await service.validate_rag_snapshot(run["snapshot"].get("rag_snapshot"))
            from agent_platform.core.db import ChatMessageRecord
            from sqlalchemy import select
            async with service.store.transaction() as db:
                await service.store._locked(db, run["id"], fence)
                if not await db.scalar(select(ChatMessageRecord.id).where(ChatMessageRecord.message_id == run["id"] + "-answer")):
                    db.add(ChatMessageRecord(message_id=run["id"] + "-answer", session_id=run["session_id"],
                        user_id=run["owner"], role="assistant", content=state["answer"], agent=state["agent"], run_id=run["id"]))
            current = await service.store.get(run["id"])
            results = [c.get("result", {}) for c in current["tool_calls"]]
            await service.store.artifact(run["id"], fence, "report.md", state["answer"], "text/markdown")
            matrices = [r.get("data") for r in results if isinstance(r.get("data"), dict) and any(k in r["data"] for k in ("matrix", "comparison_matrix", "comparisons"))]
            await service.store.artifact(run["id"], fence, "comparison-matrix.json", json.dumps({"matrices": matrices, "tool_results": results}, ensure_ascii=False, default=str))
            await service.store.artifact(run["id"], fence, "sources.json", json.dumps(state["verification"]["evidence_refs"], ensure_ascii=False))
            await service.store.artifact(run["id"], fence, "run-record.json", json.dumps({"run_id": run["id"], "snapshot": run["snapshot"], "budget": current["budget"], "verification": state["verification"]}, ensure_ascii=False, default=str))
            for offset in range(0, len(state["answer"]), 80):
                await service.store.emit(run["id"], fence, "answer_delta", {"content": state["answer"][offset:offset+80]}, dedup_key=f"final-answer-{offset}")
            await service.store.emit(run["id"], fence, "answer", {"answer": state["answer"], "agent": state["agent"]}, dedup_key="final-answer")
            return {}
        graph = StateGraph(RunState)
        for name, node in (("prepare", prepare), ("execute", execute), ("verify", verify), ("produce", produce)):
            graph.add_node(name, node)
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "execute")
        graph.add_edge("execute", "verify")
        graph.add_edge("verify", "produce")
        graph.add_edge("produce", END)
        return graph.compile(checkpointer=saver)


class Worker:
    def __init__(self, service, saver, concurrency=2, heartbeat=10):
        self.service, self.saver = service, saver
        self.id, self.concurrency, self.heartbeat_interval = uid(), min(concurrency, 2), heartbeat
        self.tasks = {}
        self.stopping = False

    async def execute(self, run):
        fence = (self.id, run["epoch"])
        store = self.service.store
        try:
            identity = await self.service.identity(run)
        except Exception as exc:
            await store.finish(run["id"], fence, "failed", {"error": str(exc)})
            return
        profile = {**run["snapshot"]["profile"], "original_goal": run["goal"]}
        current = asyncio.current_task()
        started, accounted = time.monotonic(), [0.0]
        async def heartbeat():
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                try:
                    elapsed = time.monotonic() - started
                    await store.heartbeat(run["id"], fence, elapsed-accounted[0])
                    accounted[0] = elapsed
                except Exception:
                    current.cancel()
                    return
        beat = asyncio.create_task(heartbeat())
        ctx = ExecutionContext(actor=run["owner"], role=identity["role"], run_id=run["id"],
                               profile=profile, checkpointer=self.saver,
                               graph_config={"configurable": {"thread_id": run["id"]}, "recursion_limit": 80},
                               tool_scopes=frozenset(identity["scopes"]),
                               resources={"store": store, "skills": self.service.skills, "selected_skills": [], "fence": fence,
                                          "python_tool": self.service.python_tool,
                                          "fixture": run["snapshot"]["completion"].get("fixture"),
                                          "rag_snapshot": run["snapshot"].get("rag_snapshot"),
                                          "exporter": self.service.exporter,
                                          "tool_semaphore": asyncio.Semaphore(3), "write_lock": asyncio.Lock(),
                                          "agent_semaphore": asyncio.Semaphore(2)},
                               reserve=lambda kind, amount=1: store.reserve(run["id"], fence, kind, amount),
                               settle=lambda kind, amount: store.settle(run["id"], fence, kind, amount),
                               emit=lambda kind, payload: self.service.emit(run, fence, kind, payload),
                               tool_before=lambda name, args, tool: self.service.before_tool(run, fence, name, args, tool),
                               tool_start=lambda name, args, tool: self.service.start_tool(run, fence, name, args, tool),
                               tool_after=lambda name, args, result: self.service.after_tool(run, fence, name, args, result))
        token = execution_context.set(ctx)
        step_token = step_context.set(fingerprint({"run": run["id"], "step": "execute"}))
        role_token = set_principal_role(identity["role"])
        try:
            # Never resume a different skill source under an old configuration hash.
            if run["snapshot"]["profile"]["skills"] != self.service.skills.summaries():
                raise ValueError("Skill 内容已变更，请以新配置创建任务")
            if run["snapshot"]["code_version"] != self.service.code_version():
                raise ValueError("代码版本已变更，请创建新任务，历史 checkpoint 不可套用新实现")
            if run["snapshot"].get("model_configuration") != self.service.model_configuration():
                raise ValueError("模型或工具配置已变化，请使用新快照创建任务")
            await store.rag_recover_unknown(run["id"], fence)
            await self.service.validate_rag_snapshot(run["snapshot"].get("rag_snapshot"))
            persisted = await store.get(run["id"])
            ctx.resources["required_evidence"] = [ref for call in persisted["tool_calls"] for ref in call.get("result", {}).get("evidence_refs", [])]
            artifacts = {a["name"]: a["id"] for a in persisted["artifacts"]}
            ctx.resources["evidence_facts"] = {call["id"]: {"tool_call_id": call["id"], "artifact_id": artifacts.get(f"tool-{call['id']}.json"),
                "facts": evidence_facts(call.get("result", {}).get("data"), artifacts.get(f"tool-{call['id']}.json"))}
                for call in persisted["tool_calls"] if call.get("result", {}).get("evidence_refs")}
            graph = self.service.graph(run, fence, self.saver)
            checkpoint = await graph.aget_state(ctx.graph_config)
            if checkpoint.values and not checkpoint.next and run["result"].get("resume_from_status") == "failed":
                # Explicit manual retry only. The action ledger still prevents redoing successful or unknown writes.
                values = dict(checkpoint.values)
                values["messages"] = values.get("messages", []) + [{"role": "user", "content": "用户明确请求恢复失败任务，补充信息（数据）：" + json.dumps(run["result"].get("resume_input", {}), ensure_ascii=False)}]
                await graph.aupdate_state(ctx.graph_config, values, as_node="prepare")
                checkpoint = await graph.aget_state(ctx.graph_config)
            if checkpoint.values:
                interruptions = [item for task in checkpoint.tasks for item in task.interrupts]
                if interruptions:
                    data = run["result"].get("resume_input", {})
                    def ready(item):
                        kind = item.value.get("kind")
                        if kind == "approval":
                            return "approval_id" in data
                        if kind == "reconcile":
                            return data.get("tool_call_id") == item.value.get("tool_call_id") and data.get("reconciled")
                        return "approval_id" not in data and "tool_call_id" not in data
                    argument = Command(resume={item.id: data for item in interruptions if ready(item)})
                else:
                    argument = None
            else:
                argument = {"original_goal": run["goal"]}
            remaining = profile["budget"]["active_seconds"] - run["budget"].get("active_seconds", 0)
            async with asyncio.timeout(max(remaining, .01)):
                result = await graph.ainvoke(argument, config=ctx.graph_config)
            if ctx.resources.get("rag_invalidation"):
                raise ValueError(ctx.resources["rag_invalidation"])
            await self.service.validate_rag_snapshot(run["snapshot"].get("rag_snapshot"))
            await store.reserve(run["id"], fence, "active_seconds", time.monotonic()-started-accounted[0])
            pending = result.get("__interrupt__", ())
            if pending:
                payloads = [item.value for item in pending]
                payload = next((item for item in payloads if item.get("kind") == "approval"), payloads[0])
                await store.emit(run["id"], fence, "input" if payload.get("kind") != "approval" else "approval_wait", payload)
                await store.finish(run["id"], fence, "waiting_approval" if payload.get("kind") == "approval" else "waiting_input", {"pending": payload, "pending_requests": payloads})
            else:
                await store.finish(run["id"], fence, "completed" if result["verification"]["status"] == "pass" else "failed",
                                   {"answer": result["answer"], "verification": result["verification"], "agent": result["agent"]})
        except LeaseLost:
            pass
        except asyncio.CancelledError:
            try:
                await store.finish(run["id"], fence, "failed", {"error": "执行超时或 Worker 停止；可从 checkpoint 恢复"})
            except LeaseLost:
                pass
            raise
        except Exception as exc:
            try:
                await store.finish(run["id"], fence, "failed", {"error": f"{type(exc).__name__}: {exc}"})
            except LeaseLost:
                pass
        finally:
            beat.cancel()
            await asyncio.gather(beat, return_exceptions=True)
            reset_principal_role(role_token)
            step_context.reset(step_token)
            execution_context.reset(token)

    async def serve(self):
        while not self.stopping:
            done = [key for key, task in self.tasks.items() if task.done()]
            for key in done:
                await asyncio.gather(self.tasks.pop(key), return_exceptions=True)
            for run in await self.service.store.claim(self.id, limit=self.concurrency-len(self.tasks)) if len(self.tasks) < self.concurrency else []:
                self.tasks[run["id"]] = asyncio.create_task(self.execute(run))
            await asyncio.sleep(0.5)

    async def close(self):
        self.stopping = True
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
