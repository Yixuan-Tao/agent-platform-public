"""反馈闭环：up 延迟入库（可撤销），down/retry 记录原因。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from agent_platform.core.models import Feedback
from agent_platform.memory.long_term import LongTermMemory


class FeedbackService:
    def __init__(
        self,
        long_term: LongTermMemory,
        delay_seconds: float = 5.0,
    ) -> None:
        self.long_term = long_term
        self.delay_seconds = delay_seconds
        self._pending: dict[str, asyncio.Task] = {}

    async def record(self, feedback: Feedback, answer: str = "") -> None:
        """up：延迟入库；down/retry：仅记日志（持久化由网关处理）。"""
        question = feedback.reason or feedback.session_id
        await self._record(feedback, question=question, answer=answer)

    async def record_with_qa(
        self, feedback: Feedback, *, question: str, answer: str, owner: str = ""
    ) -> None:
        """带原始问答的反馈入口：由调用方提供溯源到的问题与答案。

        与 :meth:`record` 的差异是这里不再用 ``session_id`` 或 ``reason`` 充当
        问题，避免长期记忆里出现无意义的 few-shot 样本。
        """
        await self._record(feedback, question=question, answer=answer, owner=owner)

    async def _record(self, feedback: Feedback, *, question: str, answer: str, owner: str = "") -> None:
        if feedback.feedback_type == "up":
            if not question or not answer:
                logger.warning(
                    "好评缺少可溯源问答，跳过长期记忆入库: session={} message={}",
                    feedback.session_id,
                    feedback.message_id,
                )
                return
            key = f"{feedback.session_id}:{feedback.message_id}"
            # 同一消息重复点赞时，先撤销上一次待入库任务，避免重复样本。
            previous = self._pending.pop(key, None)
            if previous is not None:
                previous.cancel()
            task = asyncio.create_task(
                self._delayed_add(
                    session_id=feedback.session_id,
                    message_id=feedback.message_id,
                    question=question,
                    answer=answer,
                    owner=owner,
                )
            )
            self._pending[key] = task
            logger.info("收到好评，{}s 后入库（可撤销）", self.delay_seconds)
        elif feedback.feedback_type == "down":
            logger.info("收到差评: session={} reason={}", feedback.session_id, feedback.reason)
        elif feedback.feedback_type == "retry":
            logger.info(
                "收到修正: session={} correction={}", feedback.session_id, feedback.correction
            )

    async def _delayed_add(self, session_id: str, message_id: str, question: str, answer: str, owner: str = "") -> None:
        try:
            await asyncio.sleep(self.delay_seconds)
            if answer:
                self.long_term.add_qa(question=question, answer=answer, weight=1.0, owner=owner, state="candidate", meta={"session_id": session_id, "message_id": message_id})
                logger.info("问答已入长期记忆: {} -> {}", question[:30], answer[:30])
        except asyncio.CancelledError:
            logger.info("入库已撤销（用户 undo）")
            raise

    async def undo(self, session_id: str, message_id: str) -> bool:
        """在延迟窗口内撤销入库。"""
        key = f"{session_id}:{message_id}"
        task = self._pending.pop(key, None)
        if task is None:
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True
