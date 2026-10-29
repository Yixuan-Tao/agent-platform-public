# Test values are synthetic fixtures for software behavior.
"""memory 包测试。"""

from __future__ import annotations

import asyncio
import tempfile

from agent_platform.core.models import ChatMessage, Feedback
from agent_platform.memory.feedback import FeedbackService
from agent_platform.memory.long_term import LongTermMemory
from agent_platform.memory.short_term import ShortTermMemory


async def test_short_term_roundtrip():
    mem = ShortTermMemory(use_fakeredis=True)
    await mem.append_message("s1", ChatMessage(role="user", content="你好"))
    await mem.append_message("s1", ChatMessage(role="assistant", content="世界"))
    msgs = await mem.get_messages("s1")
    assert len(msgs) == 2
    assert msgs[0].content == "你好"
    await mem.clear("s1")
    assert await mem.get_messages("s1") == []


async def test_short_term_recent_k():
    mem = ShortTermMemory(use_fakeredis=True)
    for i in range(5):
        await mem.append_message("s2", ChatMessage(role="user", content=str(i)))
    recent = await mem.recent("s2", k=2)
    assert [m.content for m in recent] == ["3", "4"]


async def test_long_term_add_search():
    lt = LongTermMemory(chroma_dir=tempfile.mkdtemp(prefix="lt_"))
    lt.add_qa("VPN 无法连接怎么办", "重启 VPN 客户端并重新登录", weight=1.0, scope="shared", state="published")
    lt.add_qa("打印机卡纸", "打开后盖取出卡纸", weight=1.0, scope="shared", state="published")
    results = lt.search("VPN 连不上", k=1)
    assert results and "VPN" in results[0]["question"]
    assert lt.count() >= 2


def test_long_term_memory_uses_chromadb_and_persists():
    """回归：自定义嵌入函数缺少 name() 会导致 ChromaDB 初始化失败并静默降级为内存。"""
    chroma_dir = tempfile.mkdtemp(prefix="lt_persist_")
    lt = LongTermMemory(chroma_dir=chroma_dir)
    assert lt._collection is not None, (
        "ChromaDB 初始化失败会退化为纯内存，重启即丢失长期记忆"
    )
    lt.add_qa("VPN 无法连接怎么办", "重启客户端并重新登录", weight=1.0, scope="shared", state="published")
    assert lt.count() == 1

    # 重新打开同一目录：数据仍在，证明确实持久化
    reopened = LongTermMemory(chroma_dir=chroma_dir)
    assert reopened._collection is not None
    assert reopened.count() == 1
    hits = reopened.search("VPN 连不上", k=1)
    assert hits and "VPN" in hits[0]["question"]


def test_local_hash_embedding_function_satisfies_chromadb_contract():
    from agent_platform.memory.long_term import _LocalHashEmbeddingFunction

    ef = _LocalHashEmbeddingFunction()
    assert ef.name() == "agent_platform_local_hash"
    assert ef.is_legacy() is False
    assert ef.default_space() == "cosine"
    assert "cosine" in ef.supported_spaces()
    rebuilt = _LocalHashEmbeddingFunction.build_from_config(ef.get_config())
    assert rebuilt.get_config() == {"dim": 256}
    vectors = rebuilt(["abc", "abd"])
    assert len(vectors) == 2
    assert abs(sum(v * v for v in vectors[0]) - 1.0) < 1e-6, "向量应为归一化结果"


async def test_long_term_search_filtered_min_score():
    """相关性阈值：低分结果被丢弃。"""
    lt = LongTermMemory(chroma_dir=tempfile.mkdtemp(prefix="ltf_"))
    lt._collection = None  # 强制走纯内存确定性路径
    lt._fallback = [
        {"question": "A" * 10, "answer": "B" * 10, "weight": 1.0, "scope": "shared", "state": "published"},
        {"question": "C" * 10, "answer": "D" * 10, "weight": 1.0, "scope": "shared", "state": "published"},
    ]
    results = lt.search_filtered("A" * 10, k=3, min_score=0.5, max_tokens=500)
    assert len(results) == 1
    assert results[0]["question"] == "A" * 10


async def test_long_term_search_filtered_token_cap():
    """token 累计截断：放得下第一条、放不下第二条时只返回第一条。"""
    lt = LongTermMemory(chroma_dir=tempfile.mkdtemp(prefix="ltf_"))
    lt._collection = None
    lt._fallback = [
        {"question": "A" * 10, "answer": "B" * 200, "weight": 1.0, "scope": "shared", "state": "published"},
        {"question": "C" * 10, "answer": "D" * 400, "weight": 1.0, "scope": "shared", "state": "published"},
    ]
    results = lt.search_filtered("A" * 10, k=3, min_score=0.0, max_tokens=100)
    assert len(results) == 1
    assert results[0]["question"] == "A" * 10


async def test_long_term_search_filtered_k_cap():
    """条数上限：k 生效。"""
    lt = LongTermMemory(chroma_dir=tempfile.mkdtemp(prefix="ltf_"))
    lt._collection = None
    lt._fallback = [
        {"question": "A" * 10, "answer": "B" * 10, "weight": 1.0},
        {"question": "A" * 10 + "x", "answer": "B" * 10, "weight": 1.0},
        {"question": "A" * 10 + "y", "answer": "B" * 10, "weight": 1.0},
    ]
    results = lt.search_filtered("A" * 10, k=2, min_score=0.0, max_tokens=500)
    assert len(results) <= 2


async def test_feedback_delayed_add_and_undo():
    lt = LongTermMemory(chroma_dir=tempfile.mkdtemp(prefix="fb_"))
    svc = FeedbackService(long_term=lt, delay_seconds=0.05)
    fb = Feedback(session_id="s", message_id="m", feedback_type="up")
    fb2 = Feedback(
        session_id="s", message_id="m", feedback_type="up", reason="q", correction="a"
    )
    # undo 在延迟窗口内生效
    await svc.record(fb, answer="a1")
    assert await svc.undo("s", "m") is True
    await asyncio.sleep(0.15)
    assert lt.count() == 0
    # 不撤销则入库
    await svc.record(fb2, answer="a2")
    await asyncio.sleep(0.15)
    assert lt.count() == 1

def test_private_candidates_and_legacy_memory_never_cross_owner(tmp_path):
    memory = LongTermMemory(str(tmp_path / "memory"))
    memory.add_qa("VPN legacy", "unknown owner")
    memory.add_qa("VPN candidate", "not verified", owner="alice")
    memory.add_qa("VPN alice", "private answer", owner="alice", state="verified")
    memory.add_qa("VPN shared", "public answer", owner="admin", scope="shared", state="published")
    assert {r["answer"] for r in memory.search("VPN", owner="bob")} == {"public answer"}
    assert {r["answer"] for r in memory.search("VPN", owner="alice")} == {"public answer", "private answer"}
    assert memory.count() == 4

def test_memory_revoke_persisted_source_not_only_pending(tmp_path):
    memory = LongTermMemory(str(tmp_path / "memory"))
    memory.add_qa("VPN", "answer", owner="alice", state="verified", meta={"session_id": "s", "message_id": "m"})
    memory.revoke_source("bob", "s", "m")
    assert memory.search("VPN", owner="alice")
    memory.revoke_source("alice", "s", "m")
    assert not memory.search("VPN", owner="alice")
