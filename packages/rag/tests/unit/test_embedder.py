# Test values are synthetic fixtures for software behavior.
"""
Embedder 单元测试

覆盖：
    - BM25SparseEncoder 分词 / 拟合 / 编码
    - Embedder 统一入口
    - DocumentChunk 实体转换
"""
from __future__ import annotations

import pytest

from core.embedder import BM25SparseEncoder, Embedder
from db.schemas import MODALITY_IMAGE, MODALITY_TEXT, DocumentChunk


class TestBM25SparseEncoder:
    """BM25 稀疏编码器测试。"""

    @pytest.fixture
    def corpus(self):
        return [
            "ESD protection diode breakdown voltage 27V",
            "Maximum clamping voltage is 43V at 1A",
            "Package type is SOD-323 surface mount",
        ]

    def test_tokenize_chinese(self):
        """中文文本应按字或词切分。"""
        encoder = BM25SparseEncoder()
        tokens = encoder._tokenize("静电保护二极管击穿电压")
        assert len(tokens) > 0
        # 至少包含中文字符
        assert any("\u4e00" <= t <= "\u9fff" for t in tokens)

    def test_tokenize_english(self):
        """英文文本应有词级切分。"""
        encoder = BM25SparseEncoder()
        tokens = encoder._tokenize("ESD protection diode")
        assert "esd" in tokens
        assert "protection" in tokens
        assert "diode" in tokens

    def test_tokenize_empty(self):
        """空文本返回空列表。"""
        encoder = BM25SparseEncoder()
        assert encoder._tokenize("") == []
        assert encoder._tokenize("   ") == []

    def test_fit_builds_vocab(self, corpus):
        """拟合后应建立词表。"""
        encoder = BM25SparseEncoder()
        encoder.fit(corpus)
        assert encoder.is_fitted
        assert encoder.vocab_size > 0
        assert encoder._N == 3

    def test_encode_query_returns_dict(self, corpus):
        """查询编码应返回 dict[int, float]。"""
        encoder = BM25SparseEncoder()
        encoder.fit(corpus)
        vec = encoder.encode_query("ESD voltage")
        assert isinstance(vec, dict)
        # 至少有一个词命中
        assert len(vec) > 0

    def test_encode_query_unfitted_returns_empty(self):
        """未拟合时查询编码应返回空字典。"""
        encoder = BM25SparseEncoder()
        vec = encoder.encode_query("ESD voltage")
        assert vec == {}

    def test_encode_documents_returns_correct_length(self, corpus):
        """文档编码结果数量应与输入一致。"""
        encoder = BM25SparseEncoder()
        encoder.fit(corpus)
        vecs = encoder.encode_documents(corpus)
        assert len(vecs) == len(corpus)
        for v in vecs:
            assert isinstance(v, dict)

    def test_update_corpus_incremental(self, corpus):
        """增量更新语料后词表应增长。"""
        encoder = BM25SparseEncoder()
        encoder.fit(corpus[:2])
        original_size = encoder.vocab_size
        encoder.update_corpus(corpus[2:])
        assert encoder.vocab_size >= original_size

    def test_update_corpus_empty(self, corpus):
        """空增量更新不改变状态。"""
        encoder = BM25SparseEncoder()
        encoder.fit(corpus)
        original_size = encoder.vocab_size
        encoder.update_corpus([])
        assert encoder.vocab_size == original_size


class TestEmbedder:
    """Embedder 统一入口测试。"""

    def test_embed_query_returns_tuple(self):
        """embed_query 应返回 (dense, sparse) 元组。"""
        embedder = Embedder()
        # 需要先 fit BM25
        embedder.bm25.fit(["hello world test"])
        dense, sparse = embedder.embed_query("test query")
        # dense 可能为 None（无 API key），但 sparse 不应为空
        assert isinstance(sparse, dict)

    def test_embed_chunks_fills_vectors(self):
        """embed_chunks 应填充向量字段。"""
        embedder = Embedder()
        chunks = [
            DocumentChunk(
                id="c1", doc_id="d1", file_name="test.pdf",
                modality=MODALITY_TEXT,
                content="hello world", searchable_text="hello world",
                page=1,
            ),
        ]
        result = embedder.embed_chunks(chunks)
        assert len(result) == 1
        # 至少 sparse_vector 应被填充
        assert isinstance(result[0].sparse_vector, dict)

    def test_embed_chunks_empty(self):
        """空列表直接返回。"""
        embedder = Embedder()
        result = embedder.embed_chunks([])
        assert result == []

    def test_update_corpus(self):
        """update_corpus 应能增量添加语料。"""
        embedder = Embedder()
        embedder.update_corpus(["hello world"])
        assert embedder.bm25.is_fitted
        assert embedder.bm25._N == 1


class TestDocumentChunkConversion:
    """DocumentChunk → ChromaDB entity 转换测试。"""

    def test_to_chroma_entity_text(self, sample_text_chunk):
        """文本 chunk 应转换为含 dense_vector 的实体。"""
        cid, emb, meta, doc = sample_text_chunk.to_chroma_entity()
        assert len(emb) == 1024
        assert meta["modality"] == MODALITY_TEXT

    def test_to_chroma_entity_image(self, sample_image_chunk):
        """图片 chunk 应转换为含 image_vector 的实体。"""
        cid, emb, meta, doc = sample_image_chunk.to_chroma_entity()
        assert len(emb) == 1024
        assert meta["modality"] == MODALITY_IMAGE

    def test_to_text_entity_image(self, sample_image_chunk):
        """to_text_entity 应始终输出 dense_vector (1024维)。"""
        cid, emb, meta, doc = sample_image_chunk.to_text_entity()
        assert len(emb) == 1024

    def test_metadata_excludes_image_bytes(self, sample_image_chunk):
        """metadata 中不应包含 image_bytes（不可序列化）。"""
        sample_image_chunk.metadata["image_bytes"] = b"fake_bytes"
        cid, emb, meta, doc = sample_image_chunk.to_chroma_entity()
        assert "image_bytes" not in meta.get("metadata_json", "")
