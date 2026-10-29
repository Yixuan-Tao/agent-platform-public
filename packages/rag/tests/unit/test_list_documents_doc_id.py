# Test values are synthetic fixtures for software behavior.
"""文档列表需带 doc_id：/compare/* 按 doc_id 定位文档，调用方只有文件名。"""

import json
import threading

from db.chroma_client import ChromaClientSingleton


class _Collection:
    def __init__(self, metadatas):
        self.name = "fake"
        self._metadatas = metadatas

    def get(self, include=None):
        return {"ids": [f"id{i}" for i in range(len(self._metadatas))],
                "metadatas": self._metadatas}


def _client(doc_metas, image_metas=None):
    """构造一个跳过 __init__ 的实例，避免连接真实 ChromaDB。"""
    inst = ChromaClientSingleton.__new__(ChromaClientSingleton)
    inst._client = object()
    inst._doc_collection = _Collection(doc_metas)
    inst._image_collection = _Collection(image_metas or [])
    inst._list_cache = None
    inst._list_cache_ts = 0.0
    inst._list_cache_lock = threading.Lock()
    inst._LIST_CACHE_TTL = 30.0
    return inst


def test_list_documents_returns_doc_id_per_file():
    inst = _client(
        [
            {"file_name": "CISPR25_2021.pdf", "doc_id": "aaaa1111bbbb"},
            {"file_name": "CISPR25_2021.pdf", "doc_id": "aaaa1111bbbb"},
            {"file_name": "ISO_11452-4.pdf", "doc_id": "cccc2222dddd"},
        ]
    )
    docs = {d["file_name"]: d for d in inst.list_documents()}

    assert docs["CISPR25_2021.pdf"]["doc_id"] == "aaaa1111bbbb"
    assert docs["CISPR25_2021.pdf"]["chunk_count"] == 2
    assert docs["ISO_11452-4.pdf"]["doc_id"] == "cccc2222dddd"


def test_list_documents_merges_image_collection_and_keeps_doc_id():
    inst = _client(
        [{"file_name": "a.pdf", "doc_id": "d1"}],
        [{"file_name": "a.pdf", "doc_id": "d1"}],
    )
    docs = inst.list_documents()
    assert len(docs) == 1
    assert docs[0]["chunk_count"] == 2
    assert docs[0]["doc_id"] == "d1"


def test_list_documents_tolerates_missing_doc_id():
    inst = _client([{"file_name": "legacy.pdf"}])
    docs = inst.list_documents()
    assert docs[0]["doc_id"] == "", "旧数据没有 doc_id 时应返回空串而不是报错"


def test_list_documents_result_is_cached():
    inst = _client([{"file_name": "a.pdf", "doc_id": "d1"}])
    first = inst.list_documents()
    inst._doc_collection = _Collection([])  # 换掉底层数据
    cached = inst.list_documents()
    assert first == cached
    inst.invalidate_list_cache()
    assert inst.list_documents() == []


def test_document_list_response_schema_accepts_doc_id():
    from api.schemas import DocumentListItem

    item = DocumentListItem(file_name="a.pdf", chunk_count=3, doc_id="d1")
    payload = json.loads(item.model_dump_json())
    assert payload["doc_id"] == "d1"

    legacy = DocumentListItem(file_name="a.pdf", chunk_count=1)
    assert legacy.doc_id == ""
