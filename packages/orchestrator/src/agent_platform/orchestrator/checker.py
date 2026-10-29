"""反思校验：Checker + ReflectionLoop（生成→校验→重做）。"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from loguru import logger

from agent_platform.core.models import ChatMessage
from pydantic import BaseModel
from typing import Literal


class CheckResult(BaseModel):
    status: Literal["pass", "fail", "unverified"]
    reason: str = ""

    def __iter__(self):
        yield self.status == "pass"
        yield self.reason


class Checker:
    def __init__(self, llm_router: Any) -> None:
        self.llm_router = llm_router

    async def check(self, task: str, answer: str, evidence: str = "") -> CheckResult:
        """返回 (是否通过, 原因/反馈)。"""
        prompt = (
            "你是质检员。判断以下回答是否：\n"
            "1) 完整回答了用户问题；2) 无明显幻觉或编造；3) 条理清晰。\n"
            "若通过输出：PASS\n"
            "若不通过，输出：FAIL: <具体原因与修改建议>\n\n"
            f"用户问题：{task}\n\n回答：{answer}\n\n可核验证据：{evidence or '未提供；不能确认事实正确性'}"
        )
        try:
            verdict = await self.llm_router.chat(
                [ChatMessage(role="user", content=prompt)], temperature=0.0
            )
        except Exception as exc:  # noqa: BLE001
            from agent_platform.core.execution import BudgetExceeded, is_graph_interrupt
            if isinstance(exc, BudgetExceeded) or is_graph_interrupt(exc):
                raise
            logger.warning("Checker 不可用: {}", type(exc).__name__)
            return CheckResult(status="unverified", reason="质检服务不可用")
        verdict = (verdict or "").strip()
        if verdict.startswith("PASS"):
            return CheckResult(status="pass")
        reason = verdict.removeprefix("FAIL:").strip() if "FAIL" in verdict else verdict
        return CheckResult(status="fail", reason=reason)


class ReflectionLoop:
    """生成 → 校验 → 反馈重做，最多 max_iterations 轮。"""

    def __init__(self, checker: Checker, max_iterations: int = 2) -> None:
        self.checker = checker
        self.max_iterations = max_iterations

    async def run(
        self,
        task: str,
        generator: Callable[[str], Awaitable[str]],
    ) -> tuple[str, int]:
        """generator 接收可选改进反馈，返回新答案。返回 (答案, 尝试次数)。"""
        feedback = ""
        answer = ""
        attempts = 0
        for attempt in range(1, self.max_iterations + 1):
            attempts = attempt
            answer = await generator(feedback)
            if not answer:
                break
            result = await self.checker.check(task, answer)
            passed, reason = result
            if getattr(result, "status", "") == "unverified":
                return answer, attempts
            if passed:
                return answer, attempts
            feedback = reason or "回答不够好，请改进。"
            logger.info("反思第 {} 轮未通过: {}", attempt, feedback[:80])
        return answer, attempts
