# Test values are synthetic fixtures for software behavior.
from core.parser import DocumentChunk as ParserChunk
from pipeline.ingest import IngestPipeline


class _Parser:
    def parse(self, _path, file_name=None):
        return [ParserChunk(
            chunk_id="chunk", doc_id="doc-1",
            file_name=file_name or "spec.pdf", modality="text",
            content="content", searchable_text="content", page=1,
        )]


class _Embedder:
    def embed_chunks(self, _chunks):
        return None


class _Chroma:
    def __init__(self):
        self.rolled_back = []

    def insert(self, _chunks):
        raise RuntimeError("write failed after partial insert")

    def delete_by_doc_id(self, doc_id):
        self.rolled_back.append(doc_id)


def test_ingest_rolls_back_document_when_index_write_fails():
    chroma = _Chroma()
    pipeline = IngestPipeline(parser=_Parser(), embedder=_Embedder(), chroma_client=chroma)

    result = pipeline.ingest_file("spec.pdf")

    assert result["status"] == "error"
    assert chroma.rolled_back == ["doc-1"]
