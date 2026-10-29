# Test values are synthetic fixtures for software behavior.
"""
手动测试运行器 — 无需 pytest，直接运行所有测试

用法: python tests/run_tests.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from unittest.mock import patch

# 将项目根目录加入 sys.path，确保从任意目录运行都能导入 core/db 等模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

passed = 0
failed = 0
errors: list[tuple[str, str]] = []


def run_test(name: str, func):
    global passed, failed
    try:
        func()
        lines.append(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        lines.append(f"  [FAIL] {name}: {e}")
        failed += 1
        errors.append((name, traceback.format_exc()))


lines: list[str] = []


# ============================================================
# 1. Reranker 测试
# ============================================================
print("=== Reranker 测试 ===")
from core.reranker import Reranker

candidates = [
    {"id": "c1", "score": 0.5, "content": "ESD diode 27V.", "file_name": "a.pdf", "page": 1},
    {"id": "c2", "score": 0.9, "content": "Clamping 43V.", "file_name": "a.pdf", "page": 2},
    {"id": "c3", "score": 0.3, "content": "SOD-323.", "file_name": "a.pdf", "page": 3},
]


def test_empty_candidates():
    r = Reranker()
    assert r.rerank("test", []) == []


def test_clips_top_k():
    r = Reranker()
    result = r.rerank("ESD", candidates, top_k=2)
    assert len(result) == 2


def test_has_rerank_score():
    r = Reranker()
    result = r.rerank("ESD", candidates, top_k=3)
    for item in result:
        assert "rerank_score" in item


def test_sorted_desc():
    r = Reranker()
    result = r.rerank("ESD", candidates, top_k=3)
    scores = [item["rerank_score"] for item in result]
    assert scores == sorted(scores, reverse=True)


def test_cache_hit():
    r = Reranker()
    r1 = r.rerank("ESD", candidates, top_k=2)
    r2 = r.rerank("ESD", candidates, top_k=2)
    assert [x["id"] for x in r1] == [x["id"] for x in r2]


def test_different_candidates_no_collision():
    r = Reranker()
    r1 = r.rerank("ESD", candidates, top_k=2)
    diff = [candidates[0], candidates[2]]
    r2 = r.rerank("ESD", diff, top_k=2)
    assert r1 != r2, "不同候选集不应共享缓存"


def test_fallback_when_all_fail():
    """所有打分方法失败时，按原始 score 降序排列。"""
    r = Reranker()
    with patch.object(r, "_try_rerank_endpoint", return_value=None):
        with patch.object(r, "_try_embedding_similarity", return_value=None):
            result = r.rerank("ESD", candidates, top_k=3)
    assert len(result) == 3
    assert result[0]["id"] == "c2"  # score 0.9 最高


run_test("空候选集", test_empty_candidates)
run_test("截断到 top_k", test_clips_top_k)
run_test("包含 rerank_score", test_has_rerank_score)
run_test("按分数降序", test_sorted_desc)
run_test("缓存命中", test_cache_hit)
run_test("不同候选集隔离缓存", test_different_candidates_no_collision)
run_test("LLM失败降级到原始score", test_fallback_when_all_fail)

# ============================================================
# 2. BM25SparseEncoder 测试
# ============================================================
print("=== BM25SparseEncoder 测试 ===")
from core.embedder import BM25SparseEncoder

corpus = [
    "ESD protection diode breakdown voltage 27V",
    "Maximum clamping voltage is 43V at 1A",
    "Package type is SOD-323 surface mount",
]


def test_tokenize_empty():
    e = BM25SparseEncoder()
    assert e._tokenize("") == []


def test_tokenize_english():
    e = BM25SparseEncoder()
    tokens = e._tokenize("ESD protection diode")
    assert "esd" in tokens


def test_fit_builds_vocab():
    e = BM25SparseEncoder()
    e.fit(corpus)
    assert e.is_fitted
    assert e.vocab_size > 0


def test_encode_query():
    e = BM25SparseEncoder()
    e.fit(corpus)
    vec = e.encode_query("ESD voltage")
    assert isinstance(vec, dict)
    assert len(vec) > 0


def test_unfitted_returns_empty():
    e = BM25SparseEncoder()
    assert e.encode_query("ESD") == {}


def test_encode_documents_length():
    e = BM25SparseEncoder()
    e.fit(corpus)
    vecs = e.encode_documents(corpus)
    assert len(vecs) == len(corpus)


def test_update_corpus():
    e = BM25SparseEncoder()
    e.fit(corpus[:2])
    orig = e.vocab_size
    e.update_corpus(corpus[2:])
    assert e.vocab_size >= orig


run_test("空文本分词", test_tokenize_empty)
run_test("英文分词", test_tokenize_english)
run_test("拟合建立词表", test_fit_builds_vocab)
run_test("查询编码", test_encode_query)
run_test("未拟合返回空", test_unfitted_returns_empty)
run_test("文档编码长度", test_encode_documents_length)
run_test("增量更新语料", test_update_corpus)

# ============================================================
# 3. DocumentChunk 转换测试
# ============================================================
print("=== DocumentChunk 转换测试 ===")
from db.schemas import MODALITY_IMAGE, MODALITY_TEXT, DocumentChunk

text_chunk = DocumentChunk(
    id="tc1", doc_id="d1", file_name="t.pdf",
    modality=MODALITY_TEXT, content="hello", searchable_text="hello",
    page=1, dense_vector=[0.1] * 1024,
)

img_chunk = DocumentChunk(
    id="ic1", doc_id="d1", file_name="t.pdf",
    modality=MODALITY_IMAGE, content="desc", searchable_text="desc",
    page=1, image_vector=[0.2] * 1024, dense_vector=[0.1] * 1024,
)


def test_text_entity_has_dense():
    cid, emb, meta, doc = text_chunk.to_chroma_entity()
    assert len(emb) == 1024


def test_image_entity_has_image():
    cid, emb, meta, doc = img_chunk.to_chroma_entity()
    assert len(emb) == 1024


def test_to_text_entity():
    cid, emb, meta, doc = img_chunk.to_text_entity()
    assert len(emb) == 1024


def test_image_bytes_excluded():
    img_chunk.metadata["image_bytes"] = b"fake"
    cid, emb, meta, doc = img_chunk.to_chroma_entity()
    assert "image_bytes" not in meta.get("metadata_json", "")


run_test("文本实体含dense", test_text_entity_has_dense)
run_test("图片实体含image", test_image_entity_has_image)
run_test("to_text_entity", test_to_text_entity)
run_test("image_bytes排除", test_image_bytes_excluded)

# ============================================================
# 4. ChromaClient is_connected 测试
# ============================================================
print("=== ChromaClient is_connected 测试 ===")
from db.chroma_client import ChromaClientSingleton


def test_is_connected_exists():
    assert hasattr(ChromaClientSingleton, "is_connected")


def test_is_connected_property():
    assert isinstance(ChromaClientSingleton.is_connected, property)


run_test("is_connected属性存在", test_is_connected_exists)
run_test("is_connected是property", test_is_connected_property)

# ============================================================
# 5. QueryPipeline off-by-one 验证
# ============================================================
print("=== QueryPipeline off-by-one 验证 ===")
import inspect
from pipeline.query import QueryPipeline


def test_rerank_condition_uses_ge():
    """验证重排序条件使用 >= 而非 >"""
    source = inspect.getsource(QueryPipeline.query)
    assert ">= config.RERANK_TOP_K" in source, "query 方法应使用 >= 条件"
    source2 = inspect.getsource(QueryPipeline.query_stream)
    assert ">= config.RERANK_TOP_K" in source2, "query_stream 方法应使用 >= 条件"


run_test("重排序条件使用>=", test_rerank_condition_uses_ge)

# ============================================================
# 结果汇总
# ============================================================
total = passed + failed
summary = [
    f"\n{'='*50}",
    f"  测试结果: {passed} 通过, {failed} 失败, {total} 总计",
    f"{'='*50}",
]
lines.extend(summary)
for line in lines:
    print(line)

if errors:
    print("\n失败详情:")
    for name, tb in errors:
        print(f"\n--- {name} ---")
        print(tb[-500:])

# 也将结果写入文件以便读取
with open(str(Path(__file__).resolve().parent.parent / "test_result.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
    if errors:
        f.write("\n\n失败详情:\n")
        for name, tb in errors:
            f.write(f"\n--- {name} ---\n")
            f.write(tb[-500:])

sys.exit(0 if failed == 0 else 1)
