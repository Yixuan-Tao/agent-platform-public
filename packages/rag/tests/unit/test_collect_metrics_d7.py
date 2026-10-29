# Test values are synthetic fixtures for software behavior.
"""
collect_metrics D7 图片数口径单元测试。

口径：image_count 统计 document_chunks 中 modality=image 的条数；
image_vector_count 统计 image_chunks 集合条数（真实图向量）。
"""
from __future__ import annotations

from unittest.mock import MagicMock


def _fake_chroma(image_ids: list[str], vector_count: int) -> MagicMock:
    chroma = MagicMock()
    doc_coll = MagicMock()
    doc_coll.get.return_value = {
        "ids": image_ids,
        "metadatas": [{} for _ in image_ids],
    }
    img_coll = MagicMock()
    img_coll.count.return_value = vector_count
    chroma._doc_collection = doc_coll
    chroma._image_collection = img_coll
    chroma.list_documents.return_value = [{"file_name": "a.pdf", "chunk_count": 5}]
    chroma.get_collection_stats.return_value = {"row_count": 5, "names": ["document_chunks", "image_chunks"]}
    return chroma


def test_image_count_from_doc_collection(monkeypatch):
    from scripts.collect_metrics import collect_rag_counts
    from db.chroma_client import ChromaClientSingleton

    chroma = _fake_chroma(["i1", "i2", "i3"], vector_count=3)
    monkeypatch.setattr(ChromaClientSingleton, "get_instance", classmethod(lambda cls: chroma))
    result = collect_rag_counts()
    assert result["image_count"] == 3
    assert result["image_vector_count"] == 3


def test_image_vector_count_can_differ(monkeypatch):
    from scripts.collect_metrics import collect_rag_counts
    from db.chroma_client import ChromaClientSingleton

    # 图片记录 3 条，但真实图向量只有 1 条（回填未完成）
    chroma = _fake_chroma(["i1", "i2", "i3"], vector_count=1)
    monkeypatch.setattr(ChromaClientSingleton, "get_instance", classmethod(lambda cls: chroma))
    result = collect_rag_counts()
    assert result["image_count"] == 3
    assert result["image_vector_count"] == 1
