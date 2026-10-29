# Test values are synthetic fixtures for software behavior.
"""上传入库应使用用户原始文件名，而不是带 task_id 前缀的临时文件名。"""

from pipeline.ingest import IngestPipeline
from core.parser import DocumentChunk as ParserChunk


class _FakeParser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def parse(self, file_path, file_name=None):
        self.calls.append((file_path, file_name))
        return [
            ParserChunk(
                chunk_id="c1",
                doc_id="d1",
                file_name=file_name or "placeholder.txt",
                modality="text",
                content="hello",
                searchable_text="hello",
                page=0,
                section="",
                metadata={},
            )
        ]


class _FakeEmbedder:
    def embed_chunks(self, chunks):
        return None


class _FakeChroma:
    def __init__(self):
        self.inserted = []

    def insert(self, chunks):
        self.inserted.extend(chunks)
        return {"insert_count": len(chunks)}

    def delete_by_doc_id(self, doc_id):
        return 0


def _pipeline():
    parser = _FakeParser()
    chroma = _FakeChroma()
    pipeline = IngestPipeline(parser=parser, embedder=_FakeEmbedder(), chroma_client=chroma)
    return pipeline, parser, chroma


def test_ingest_file_forwards_original_file_name(tmp_path):
    target = tmp_path / "abc123_spec.pdf"
    target.write_bytes(b"%PDF-1.4")
    pipeline, parser, chroma = _pipeline()

    result = pipeline.ingest_file(str(target), file_name="spec.pdf")

    assert result["status"] == "ok"
    assert parser.calls == [(str(target), "spec.pdf")]
    # 写入 Chroma 的元数据文件名应为用户原名
    assert chroma.inserted[0].file_name == "spec.pdf"


def test_ingest_file_defaults_to_path_name(tmp_path):
    target = tmp_path / "plain.txt"
    target.write_text("hello", encoding="utf-8")
    pipeline, parser, _ = _pipeline()

    result = pipeline.ingest_file(str(target))

    assert result["status"] == "ok"
    assert parser.calls[0][1] is None

def test_ingest_records_original_bytes_hash_and_rejects_source_changed_during_parse(tmp_path):
    import hashlib
    target = tmp_path / "source.pdf"
    target.write_bytes(b"original-source")
    pipeline, parser, chroma = _pipeline()
    assert pipeline.ingest_file(str(target))["status"] == "ok"
    assert chroma.inserted[0].metadata["source_content_sha256"] == hashlib.sha256(b"original-source").hexdigest()
    pipeline, parser, chroma = _pipeline()
    original_parse = parser.parse
    def changed_parse(*args,**kwargs):
        chunks = original_parse(*args,**kwargs)
        target.write_bytes(b"changed-source")
        return chunks
    parser.parse = changed_parse
    result = pipeline.ingest_file(str(target))
    assert result["status"] == "error" and "发生变化" in result["error"]
    assert not chroma.inserted


def test_parser_uses_override_name_for_metadata(monkeypatch, tmp_path):
    """直接验证解析器：提供 file_name 时元数据不采用路径名。"""
    from core.parser import DocumentParser

    target = tmp_path / "deadbeef_note.txt"
    target.write_text("1 概述\n内容", encoding="utf-8")

    parser = DocumentParser()
    chunks = parser.parse(str(target), file_name="note.txt")

    assert chunks, "应解析出 chunk"
    assert all(c.file_name == "note.txt" for c in chunks)
