"""长期记忆：ChromaDB 持久化的 feedback_qa_pairs 集合，做 few-shot。

为避免下载 embedding 模型，文本直接存 documents，检索用字符级相似度。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from loguru import logger

try:  # chromadb 为可选依赖：缺失时退化为普通对象，保持模块可导入
    from chromadb.api.types import EmbeddingFunction as _ChromaEmbeddingFunction
except Exception:  # noqa: BLE001  pragma: no cover
    _ChromaEmbeddingFunction = object  # type: ignore[assignment,misc]


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    text = text.lower()
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def _similarity(a: str, b: str) -> float:
    """基于 2-gram 的 Jaccard 相似度，简单且无需模型。"""
    sa, sb = _char_ngrams(a), _char_ngrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _stable_hash(text: str) -> int:
    """确定性字符串哈希（md5），保证跨进程一致。"""
    return int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:8], "big")


def _estimate_tokens(text: str) -> int:
    """轻量 token 近似（无外部依赖）：中文按字符数，英文按 4 字符 1 token，取保守上界。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + other // 4 + 1


class _LocalHashEmbeddingFunction(_ChromaEmbeddingFunction):  # type: ignore[misc,valid-type]
    """本地确定性嵌入函数：字符 2-gram 哈希向量（256 维、余弦归一化）。

    用于替代 ChromaDB 默认的 MiniLM 嵌入函数，避免首次使用时报错并触发
    模型下载（需写 ~/.cache/chroma），保持长期记忆完全离线可用。

    ChromaDB 1.x 要求自定义嵌入函数实现 ``name`` / ``get_config`` /
    ``build_from_config``，否则集合创建会抛
    ``'_LocalHashEmbeddingFunction' object has no attribute 'name'``，
    长期记忆会静默降级为纯内存（重启即丢失）。
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @staticmethod
    def name() -> str:
        return "agent_platform_local_hash"

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict[str, Any]:
        return {"dim": self._dim}

    @staticmethod
    def build_from_config(config: dict[str, Any] | None) -> "_LocalHashEmbeddingFunction":
        return _LocalHashEmbeddingFunction(dim=int((config or {}).get("dim", 256)))

    def __call__(self, input: Any) -> list[list[float]]:
        texts = [input] if isinstance(input, str) else list(input)
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for gram in _char_ngrams(text or "", 2):
                vec[_stable_hash(gram) % self._dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class LongTermMemory:
    """把"被点赞的问答对"持久化，供后续会话做 few-shot。"""

    def __init__(self, chroma_dir: str = "chroma_data") -> None:
        self.chroma_dir = chroma_dir
        self._client = None
        self._collection = None
        self._fallback: list[dict[str, Any]] = []
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=chroma_dir)
            self._collection = self._client.get_or_create_collection(
                name="feedback_qa_pairs",
                metadata={"hnsw:space": "cosine"},
                embedding_function=_LocalHashEmbeddingFunction(),
            )
            logger.debug("LongTermMemory 使用 ChromaDB: {}", chroma_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChromaDB 初始化失败，长期记忆降级为纯内存: {}", exc)
            self._client = None

    def _id(self, question: str, answer: str) -> str:
        return hashlib.md5(f"{question}|{answer}".encode("utf-8")).hexdigest()[:16]

    def add_qa(self, question: str, answer: str, weight: float = 1.0, meta: dict[str, Any] | None = None, *, owner: str = "", scope: str = "private", state: str = "candidate") -> None:
        if not question or not answer:
            return
        doc_id = self._id(owner + ":" + question, answer)
        metadata = {"question": question, "answer": answer, "weight": float(weight)}
        if meta:
            metadata.update(meta)
        metadata.update(owner=owner, scope=scope, state=state)
        if self._collection is not None:
            self._collection.upsert(
                ids=[doc_id],
                documents=[f"Q: {question}\nA: {answer}"],
                metadatas=[metadata],
            )
        else:
            self._fallback = [item for item in self._fallback if not (
                item.get("owner") == owner and item.get("question") == question and item.get("answer") == answer)]
            self._fallback.append(metadata)

    def search(self, query: str, k: int = 5, *, owner: str = "") -> list[dict[str, Any]]:
        allowed = {"$and": [{"scope": "shared"}, {"state": "published"}]}
        if owner:
            allowed = {"$or": [allowed, {"$and": [{"owner": owner}, {"scope": "private"}, {"state": "verified"}]}]}
        if self._collection is not None:
            try:
                result = self._collection.query(
                    query_texts=[query], n_results=k, where=allowed, include=["metadatas", "distances"]
                )
                metas = (result.get("metadatas") or [[]])[0]
                dists = (result.get("distances") or [[]])[0]
                out = []
                for meta, dist in zip(metas, dists):
                    if not meta:
                        continue
                    score = float(meta.get("weight", 1.0)) * (1.0 - float(dist))
                    out.append(
                        {
                            **meta,
                            "question": meta.get("question", ""),
                            "answer": meta.get("answer", ""),
                            "score": score,
                        }
                    )
                return out[:k]
            except Exception as exc:  # noqa: BLE001
                logger.warning("ChromaDB 查询失败，回退内存检索: {}", exc)
        scored = sorted(
            ((_similarity(query, item["question"]), item) for item in self._fallback
             if (item.get("scope") == "shared" and item.get("state") == "published")
             or (owner and item.get("owner") == owner and item.get("scope") == "private" and item.get("state") == "verified")),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            {**item, "score": round(score * item.get("weight", 1.0), 4)}
            for score, item in scored[:k]
            if score > 0
        ]

    def search_filtered(
        self,
        query: str,
        *,
        k: int = 3,
        min_score: float = 0.5,
        max_tokens: int = 500,
        owner: str = "",
    ) -> list[dict[str, Any]]:
        """检索相似 QA，串行应用：相关性阈值过滤 → 条数上限 → token 累计截断。

        返回结果必然满足：条数 <= k、得分 >= min_score、累计 token <= max_tokens。
        用于把长期记忆作为 few-shot 安全注入上下文，防止膨胀与污染。
        """
        results = self.search(query, k=k, owner=owner)
        passed = [r for r in results if r.get("score", 0.0) >= min_score]
        out: list[dict[str, Any]] = []
        used = 0
        for r in passed:
            approx = _estimate_tokens(str(r.get("question", "")) + str(r.get("answer", "")))
            if used + approx > max_tokens:
                break
            out.append(r)
            used += approx
        return out

    def count(self) -> int:
        if self._collection is not None:
            try:
                return self._collection.count()
            except Exception:  # noqa: BLE001
                pass
        return len(self._fallback)

    def revoke_source(self, owner: str, session_id: str, message_id: str) -> None:
        if self._collection is not None:
            records = self._collection.get(where={"$and": [{"owner": owner}, {"session_id": session_id}, {"message_id": message_id}]}, include=["metadatas"])
            if records["ids"]:
                self._collection.update(ids=records["ids"], metadatas=[{**meta, "state": "revoked"} for meta in records["metadatas"]])
        for item in self._fallback:
            if item.get("owner") == owner and item.get("session_id") == session_id and item.get("message_id") == message_id:
                item["state"] = "revoked"
