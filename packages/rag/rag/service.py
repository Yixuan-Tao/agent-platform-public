"""
RAG 服务接口层 — 封装 chroma / embedder / retriever 为统一入口。

职责：
    为上层（API 路由 / CLI / 测试）提供简洁的调用接口，
    不做业务逻辑，仅做方法委托与参数适配。
"""

from __future__ import annotations

from typing import Any, Optional

from core.retriever import Retriever
from db.chroma_client import ChromaClientSingleton
from core.embedder import Embedder


class RagService:
    """RAG 服务统一入口。

    封装底层检索、向量化、存储组件，提供高层调用接口。
    """

    def __init__(
        self,
        chroma: ChromaClientSingleton,
        embedder: Embedder,
        retriever: Retriever,
    ) -> None:
        self._chroma = chroma
        self._embedder = embedder
        self._retriever = retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filter_expr: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """检索相关 chunks（混合检索 + 重排）。

        Args:
            query: 用户查询文本。
            top_k: 返回的 chunk 数量上限。
            filter_expr: ChromaDB 过滤表达式，如 ``doc_id == "abc"``。

        Returns:
            标准化 chunk 列表（已含 linked_chunks 图文关联信息）。
        """
        return self._retriever.retrieve(query, top_k=top_k, filter_expr=filter_expr)

    def get_document_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        """获取指定文档的所有 chunks。

        Args:
            doc_id: 目标文档 ID。

        Returns:
            按 page 升序排列的 chunk 列表。
        """
        return self._chroma.get_all_by_doc_id(doc_id)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """获取单个 chunk 的详细信息。

        Args:
            chunk_id: chunk 的唯一标识。

        Returns:
            chunk 的 entity 字典，未找到时返回 None。
        """
        return self._chroma.get_by_id(chunk_id)

    def list_documents(self) -> list[dict[str, Any]]:
        """列出所有已索引的文档。

        Returns:
            每个元素为 ``{"file_name": str, "chunk_count": int}`` 的列表。
        """
        return self._chroma.list_documents()

    def retrieve_all_by_doc(
        self,
        doc_id: str,
        modalities: Optional[tuple[str, ...]] = None,
        section_id_prefix: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """全量拉取文档 chunks（委托 retriever.retrieve_all_by_doc）。

        对比/可视化工具使用此方法获取完整章节数据，不走语义检索。
        支持按 section_id 前缀过滤，确保 100% 覆盖指定章节。

        Args:
            doc_id: 目标文档 ID。
            modalities: 模态过滤（"text" / "table" / "image"），None=全量。
            section_id_prefix: 章节编号前缀，如 ``"5.3"`` 仅返回该章节下的 chunks。

        Returns:
            标准化 chunk 列表（page 升序）。
        """
        return self._retriever.retrieve_all_by_doc(
            doc_id, modalities=modalities, section_id_prefix=section_id_prefix,
        )

    def retrieve_by_section(
        self,
        doc_id: str,
        section_id: str,
        section_topic: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """按章节拉取 chunks（委托 retriever.retrieve_by_section）。

        章节路由使用：按 section_topic 过滤，确保 100% 覆盖该主题。
        跨文档对比时，topic 是跨文档通用的语义标识，section_id 仅用于文档内定位。

        Args:
            doc_id: 目标文档 ID。
            section_id: 章节编号前缀（支持子章节匹配），或空字符串仅按 topic 过滤。
            section_topic: 可选 topic 过滤（如 ``"immunity.bci"``）。

        Returns:
            匹配章节的标准化 chunk 列表。
        """
        return self._retriever.retrieve_by_section(
            doc_id, section_id, section_topic=section_topic,
        )

    def search_by_keyword(
        self,
        keyword: str,
        doc_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """按关键词搜索 chunks（基于 BM25 + 稠密向量的混合检索）。

        底层复用 Retriever.retrieve() 的混合检索能力，
        将 keyword 作为 query 进行语义 + 关键词联合检索。

        Args:
            keyword: 搜索关键词或短语。
            doc_id: 可选，限定搜索范围到指定文档。

        Returns:
            匹配的标准化 chunk 列表。
        """
        filter_expr = None
        if doc_id:
            filter_expr = f'doc_id == "{doc_id}"'
        return self._retriever.retrieve(keyword, top_k=20, filter_expr=filter_expr)
