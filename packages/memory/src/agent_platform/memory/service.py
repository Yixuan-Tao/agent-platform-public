"""MemoryService：组合短/长期记忆与反馈服务。"""

from __future__ import annotations

from agent_platform.core.config import Settings, get_settings
from agent_platform.memory.feedback import FeedbackService
from agent_platform.memory.long_term import LongTermMemory
from agent_platform.memory.short_term import ShortTermMemory


class MemoryService:
    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        feedback: FeedbackService,
        few_shot_enabled: bool = True,
        few_shot_k: int = 3,
        few_shot_min_score: float = 0.5,
        few_shot_max_tokens: int = 500,
    ) -> None:
        self.short_term = short_term
        self.long_term = long_term
        self.feedback = feedback
        self.few_shot_enabled = few_shot_enabled
        self.few_shot_k = few_shot_k
        self.few_shot_min_score = few_shot_min_score
        self.few_shot_max_tokens = few_shot_max_tokens


_memory_service: MemoryService | None = None


def build_memory_service(settings: Settings | None = None) -> MemoryService:
    global _memory_service
    if _memory_service is not None:
        return _memory_service
    settings = settings or get_settings()
    short_term = ShortTermMemory(
        redis_url=settings.redis_url,
        use_fakeredis=settings.use_fakeredis,
        ttl_seconds=settings.short_memory_ttl_seconds,
    )
    long_term = LongTermMemory(chroma_dir=settings.chroma_dir)
    feedback = FeedbackService(long_term=long_term)
    _memory_service = MemoryService(
        short_term=short_term,
        long_term=long_term,
        feedback=feedback,
        few_shot_enabled=settings.memory_few_shot_enabled,
        few_shot_k=settings.memory_few_shot_k,
        few_shot_min_score=settings.memory_few_shot_min_score,
        few_shot_max_tokens=settings.memory_few_shot_max_tokens,
    )
    return _memory_service
