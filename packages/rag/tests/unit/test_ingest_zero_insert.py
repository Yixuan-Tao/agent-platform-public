# Test values are synthetic fixtures for software behavior.
"""回归：向量化失败导致 0 条写入时，入库必须报错而不是假成功。"""

from core.parser import DocumentChunk as ParserChunk
from pipeline.ingest import IngestPipeline


class _Parser:
    def parse(self, _path, file_name=None):
        return [
            ParserChunk(
                chunk_id="chunk", doc_id="doc-1", file_name=file_name or "spec.txt",
                modality="text", content="content", searchable_text="content", page=1,
            )
        ]


class _Embedder:
    """模拟 embedding 网关不可用：不填充任何向量。"""

    def embed_chunks(self, _chunks):
        return None


class _ChromaZeroInsert:
    def __init__(self):
        self.rolled_back: list[str] = []

    def insert(self, _chunks):
        return {"insert_count": 0, "ids": []}

    def delete_by_doc_id(self, doc_id):
        self.rolled_back.append(doc_id)


class _ChromaOk:
    def insert(self, _chunks):
        return {"insert_count": 1, "ids": ["chunk"]}

    def delete_by_doc_id(self, _doc_id):
        raise AssertionError("成功路径不应回滚")


def test_zero_insert_is_reported_as_error(tmp_path):
    target = tmp_path / "spec.txt"
    target.write_text("hello", encoding="utf-8")
    chroma = _ChromaZeroInsert()
    pipeline = IngestPipeline(parser=_Parser(), embedder=_Embedder(), chroma_client=chroma)

    result = pipeline.ingest_file(str(target))

    assert result["status"] == "error"
    assert "0 条 chunk 进入索引" in result["error"]
    assert chroma.rolled_back == ["doc-1"], "失败时应清理半成品"


def test_successful_insert_reports_counts(tmp_path):
    target = tmp_path / "spec.txt"
    target.write_text("hello", encoding="utf-8")
    pipeline = IngestPipeline(parser=_Parser(), embedder=_Embedder(), chroma_client=_ChromaOk())

    result = pipeline.ingest_file(str(target))

    assert result["status"] == "ok"
    assert result["chunk_count"] == 1
    assert result["inserted_count"] == 1
