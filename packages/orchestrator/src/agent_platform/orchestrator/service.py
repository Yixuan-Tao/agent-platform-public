"""OrchestratorService：路由 → Agent 执行 → 反思 → 事件流。"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from loguru import logger

from agent_platform.orchestrator.agent_tool import configure_agent_budget, reset_agent_budget
from agent_platform.orchestrator.checker import Checker, ReflectionLoop
from agent_platform.orchestrator.registry import AgentRegistry
from agent_platform.orchestrator.router import SemanticRouter
from agent_platform.orchestrator.supervisor import _invoke_agent
from agent_platform.core.execution import execution_context, is_graph_interrupt


class OrchestratorService:
    def __init__(
        self,
        registry: AgentRegistry,
        llm_router: Any,
        tool_registry: Any,
        memory_service: Any = None,
        max_attempts: int = 2,
        chunk_size: int = 60,
        use_router_react: bool = False,
        max_agent_calls: int = 4,
        plan_mode: bool = True,
        max_plan_steps: int = 6,
        answer_streaming: bool = False,
        agent_call_budget: int = 8,
    ) -> None:
        self.registry = registry
        self.llm_router = llm_router
        self.tool_registry = tool_registry
        self.memory_service = memory_service
        self.use_router_react = use_router_react
        self.max_agent_calls = max_agent_calls
        self.plan_mode = plan_mode
        self.max_plan_steps = max_plan_steps
        self.answer_streaming = answer_streaming
        self.agent_call_budget = agent_call_budget
        self.router = SemanticRouter()
        self.checker = Checker(llm_router)
        self.reflection = ReflectionLoop(self.checker, max_iterations=max_attempts)
        self.chunk_size = chunk_size
        from agent_platform.orchestrator.planner_execute import ComplexityJudge

        self.complexity = ComplexityJudge(
            llm_router, use_llm=os.getenv("PLANNER_LLM_JUDGE", "0") == "1"
        )
        self._router_graph_cache: Any = None
        self._plan_graph_cache: Any = None

    def _compose_messages(
        self, messages: list[Any], few_shot: str = "", feedback: str = ""
    ) -> list[Any]:
        """组装送模型的上下文：历史 + few-shot + 质检反馈。"""
        msgs = list(messages)
        parts: list[str] = []
        if few_shot:
            parts.append(few_shot)
        if feedback:
            parts.append(f"质检反馈，请据此改进回答：{feedback}")
        if parts:
            msgs = msgs + [{"role": "system", "content": "\n".join(parts)}]
        return msgs

    async def _stream_answer(self, graph: Any, msgs: list[Any]) -> AsyncIterator[tuple[str, str]]:
        """逐 token 产出模型增量；图不支持流式时自动回退为非流式。

        产出 ``("delta", 增量)``，最后产出一次 ``("final", 完整答案)``；
        失败且尚未产出任何增量时回退 ``_invoke_agent``，已产出增量则报错终止，
        避免同一段回答被发送两次。
        """
        if not hasattr(graph, "astream"):
            yield "final", await _invoke_agent(graph, msgs)
            return
        emitted: list[str] = []
        final_state = None
        try:
            # Real LangGraph: consume values as well, never re-invoke after stream completion.
            real_graph = hasattr(graph, "get_state")
            ctx = execution_context.get()
            kwargs = {"config": ctx.graph_config} if ctx else {}
            async for item in graph.astream(
                {"messages": msgs, "original_goal": ctx.profile.get("original_goal", "") if ctx else ""},
                stream_mode=["messages", "values"] if real_graph else "messages", **kwargs
            ):
                if real_graph:
                    mode, payload = item
                    if mode == "values":
                        final_state = payload
                        continue
                    chunk, _meta = payload
                    # Only an explicit answer node may publish live tokens. Expert/plan tokens stay internal.
                    if _meta.get("langgraph_node") != "final_answer":
                        continue
                else:
                    chunk, _meta = item
                    if _meta.get("langgraph_node") in {"plan", "tools", "execute", "supervisor"}:
                        continue
                # 工具返回（ToolMessage）不是回答正文，过滤掉避免污染输出
                if type(chunk).__name__ == "ToolMessage":
                    continue
                text = chunk if isinstance(chunk, str) else getattr(chunk, "content", "")
                if not text:
                    continue
                emitted.append(str(text))
                yield "delta", str(text)
            if emitted:
                yield "final", "".join(emitted)
                return
            if final_state:
                answer = final_state.get("final_answer", "")
                if not answer and final_state.get("messages"):
                    last = final_state["messages"][-1]
                    answer = getattr(last, "content", "") if not isinstance(last, dict) else last.get("content", "")
                if answer:
                    yield "final", str(answer)
                    return
            yield "error", "执行结束但没有最终回答"
            return
        except Exception as exc:  # noqa: BLE001
            if is_graph_interrupt(exc):
                raise
            if emitted:
                logger.warning("流式生成中断（已产出 {} 段）: {}", len(emitted), exc)
                yield "error", f"流式生成中断: {exc}"
                return
            yield "error", f"流式生成失败: {exc}"

    async def pick_agent(self, user_message: str, agent_hint: str | None = None) -> str:
        if agent_hint and self.registry.has(agent_hint):
            return agent_hint
        agent, confidence = self.router.route(user_message)
        if agent and self.registry.has(agent):
            return agent
        agents = self.registry.names()
        descriptions = {m.name: m.description for m in self.registry.list_manifests()}
        chosen = await self.router.ensure_llm_route(
            user_message, self.llm_router, agents, descriptions
        )
        return chosen if chosen else (agents[0] if agents else "unknown")

    def _build_few_shot(self, user_message: str) -> str:
        """长期记忆 few-shot（可选增强）；检索为空/失败/超限一律降级为空。"""
        few_shot = ""
        if self.memory_service is not None and getattr(
            self.memory_service, "few_shot_enabled", True
        ):
            try:
                hits = self.memory_service.long_term.search_filtered(
                    user_message,
                    owner=execution_context.get().actor if execution_context.get() else "",
                    k=getattr(self.memory_service, "few_shot_k", 3),
                    min_score=getattr(self.memory_service, "few_shot_min_score", 0.5),
                    max_tokens=getattr(self.memory_service, "few_shot_max_tokens", 500),
                )
                if hits:
                    few_shot = "参考历史优质问答（可借鉴其表述与结构）：\n" + "\n".join(
                        f"- Q: {h['question']}\n  A: {h['answer']}" for h in hits
                    )
            except Exception:  # noqa: BLE001
                few_shot = ""
        return few_shot

    async def route_and_run(
        self,
        session_id: str,
        user_message: str,
        agent_hint: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        # 请求级预算：计数保存在 contextvar 中，按 asyncio 任务隔离
        reset_agent_budget()
        configure_agent_budget(self.agent_call_budget)
        if not self.registry.names():
            yield {"type": "error", "message": "没有可用 Agent"}
            return

        agent = await self.pick_agent(user_message, agent_hint)
        if not self.registry.has(agent):
            yield {"type": "error", "message": f"Agent 不存在: {agent}"}
            return
        yield {"type": "agent_switch", "agent": agent}
        logger.info("[{}] 路由到 Agent: {}", session_id, agent)

        # 短期记忆（可选）：拼接历史
        messages: list[Any] = [{"role": "user", "content": user_message}]
        if self.memory_service is not None:
            try:
                history = await self.memory_service.short_term.recent(session_id, k=10)
                if len(history) > 1:
                    messages = [m.model_dump() for m in history]
            except Exception:  # noqa: BLE001
                pass

        # 长期记忆 few-shot（可选增强）
        few_shot = self._build_few_shot(user_message)

        graph = self.registry.build(agent, self.llm_router, self.tool_registry)

        async def generate(feedback: str) -> str:
            msgs = self._compose_messages(messages, few_shot, feedback)
            try:
                answer = await _invoke_agent(graph, msgs)
            except Exception as exc:  # noqa: BLE001
                if is_graph_interrupt(exc):
                    raise
                logger.exception("Agent 执行失败")
                return f"Agent 执行失败: {exc}"
            return answer

        if self.answer_streaming:
            answer = ""
            attempts = 1
            async for kind, payload in self._stream_answer(
                graph, self._compose_messages(messages, few_shot)
            ):
                if kind == "delta":
                    yield {"type": "token", "content": payload}
                    continue
                if kind == "final":
                    answer = payload
                    continue
                yield {"type": "error", "message": payload}
                return
            # 内容已下发给用户，质检仅记录，不做重试
            passed, reason = await self.checker.check(user_message, answer)
            if not passed:
                logger.info("[{}] 流式回答质检未通过（仅记录）: {}", session_id, reason[:120])
        else:
            answer, attempts = await self.reflection.run(user_message, generate)

        if not answer:
            yield {"type": "error", "message": "Agent 未产出回答"}
            return

        if not self.answer_streaming:
            for i in range(0, len(answer), self.chunk_size):
                yield {"type": "token", "content": answer[i : i + self.chunk_size]}
        yield {
            "type": "done",
            "answer": answer,
            "agent": agent,
            "attempts": attempts,
        }

    def _router_graph(self) -> Any:
        """懒构建并缓存 Router ReAct 图（实例级，避免跨注册表污染）。"""
        if self._router_graph_cache is None:
            from agent_platform.orchestrator.router_react import build_router_graph

            self._router_graph_cache = build_router_graph(
                self.registry, self.llm_router, self.tool_registry, self.max_agent_calls
            )
        return self._router_graph_cache

    def _plan_graph(self) -> Any:
        """懒构建并缓存 Plan-and-Execute 图。"""
        if self._plan_graph_cache is None:
            from agent_platform.orchestrator.planner_execute import build_plan_execute_graph

            self._plan_graph_cache = build_plan_execute_graph(
                self.registry,
                self.llm_router,
                self.tool_registry,
                self.max_plan_steps,
                self.agent_call_budget,
            )
        return self._plan_graph_cache

    async def route_and_run_react(
        self,
        session_id: str,
        user_message: str,
        agent_hint: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Router-as-ReAct 双模式入口：快速路径 → 复杂度判定 → ReAct / Plan-and-Execute。"""
        reset_agent_budget()
        configure_agent_budget(self.agent_call_budget)
        if not self.registry.names():
            yield {"type": "error", "message": "没有可用 Agent"}
            return

        # 快速路径：显式 agent_hint 或语义路由高置信度（>=0.9）走原单 Agent 逻辑
        if agent_hint and self.registry.has(agent_hint):
            async for event in self.route_and_run(session_id, user_message, agent_hint):
                yield event
            return
        agent, confidence = self.router.route(user_message)
        if agent and self.registry.has(agent) and confidence >= 0.9:
            async for event in self.route_and_run(session_id, user_message, agent):
                yield event
            return

        complexity = await self.complexity.judge(user_message)

        # 上下文组装：短期记忆 + 长期记忆 few-shot
        messages: list[Any] = [{"role": "user", "content": user_message}]
        if self.memory_service is not None:
            try:
                history = await self.memory_service.short_term.recent(session_id, k=10)
                if len(history) > 1:
                    messages = [m.model_dump() for m in history]
            except Exception:  # noqa: BLE001
                pass
        few_shot = self._build_few_shot(user_message)
        if few_shot:
            messages = messages + [{"role": "system", "content": few_shot}]

        if complexity == "complex" and self.plan_mode:
            yield {
                "type": "plan_start",
                "complexity": complexity,
                "message": "检测到复杂任务，进入 Plan-and-Execute 模式",
            }
            graph = self._plan_graph()
            agent_name = "plan_execute"
        else:
            yield {
                "type": "router_start",
                "complexity": complexity,
                "message": "进入 Router ReAct 模式",
            }
            graph = self._router_graph()
            agent_name = "router"

        try:
            if self.answer_streaming:
                answer = ""
                async for kind, payload in self._stream_answer(graph, messages):
                    if kind == "delta":
                        yield {"type": "token", "content": payload}
                        continue
                    if kind == "final":
                        answer = payload
                        continue
                    yield {"type": "error", "message": payload}
                    return
            else:
                answer = await _invoke_agent(graph, messages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Router/Plan 图执行失败")
            yield {"type": "error", "message": f"Agent 执行失败: {exc}"}
            return
        if not answer:
            yield {"type": "error", "message": "Agent 未产出回答"}
            return
        if not self.answer_streaming:
            for i in range(0, len(answer), self.chunk_size):
                yield {"type": "token", "content": answer[i : i + self.chunk_size]}
        yield {
            "type": "done",
            "answer": answer,
            "agent": agent_name,
            "attempts": 1,
        }
