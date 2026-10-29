# Test values are synthetic fixtures for software behavior.
"""
pytest 全局配置与共享 fixtures

提供：
    - Mock 客户端（ChromaDB / OneAPI），避免测试依赖外部服务
    - 测试用 DocumentChunk 工厂函数
    - 模块级别的 mock 注入
    - ``--run-real-api`` CLI 选项，用于运行需要真实 LLM API 的测试
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ============================================================
# CLI 选项：--run-real-api
# ============================================================

def pytest_addoption(parser: pytest.Parser) -> None:
    """添加 --run-real-api 选项，默认跳过需要真实 API 的测试。"""
    parser.addoption(
        "--run-real-api",
        action="store_true",
        default=False,
        help="运行需要真实 LLM API 调用的测试",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item],
) -> None:
    """收集时跳过标记了 real_api 的测试（除非传了 --run-real-api）。"""
    if config.getoption("--run-real-api"):
        # 用户显式要求运行，不做跳过
        return
    skip_real_api = pytest.mark.skip(reason="需要 --run-real-api 标志运行")
    for item in items:
        if "real_api" in item.keywords:
            item.add_marker(skip_real_api)


# ============================================================
# 测试数据工厂
# ============================================================

@pytest.fixture
def sample_text_chunk():
    """标准文本 chunk 工厂。"""
    from db.schemas import DocumentChunk, MODALITY_TEXT

    return DocumentChunk(
        id="test_chunk_001",
        doc_id="test_doc_001",
        file_name="test.pdf",
        modality=MODALITY_TEXT,
        content="ESD protection diode with 27V breakdown voltage.",
        searchable_text="ESD protection diode with 27V breakdown voltage.",
        page=1,
        section="Electrical Characteristics",
        dense_vector=[0.1] * 1024,
        sparse_vector={0: 0.5, 1: 0.3},
        metadata={"language": "en", "heading_level": 2},
    )


@pytest.fixture
def sample_image_chunk():
    """标准 image chunk 工厂。"""
    from db.schemas import DocumentChunk, MODALITY_IMAGE

    return DocumentChunk(
        id="test_img_001",
        doc_id="test_doc_001",
        file_name="test.pdf",
        modality=MODALITY_IMAGE,
        content="![vector_p1_0](image)\n焊盘布局图，展示 SOT-23 封装建议焊盘尺寸。",
        searchable_text="焊盘布局图，展示 SOT-23 封装建议焊盘尺寸。",
        page=1,
        section="Package Information",
        dense_vector=[0.1] * 1024,
        image_vector=[0.2] * 1024,
        metadata={
            "language": "zh",
            "image_format": "png",
            "image_id": "vector_p1_0",
            "embedding_status": "ok",
        },
    )


@pytest.fixture
def sample_candidates():
    """模拟检索候选集。"""
    return [
        {
            "id": "c1",
            "score": 0.85,
            "content": "ESD protection diode breakdown voltage is 27V.",
            "file_name": "datasheet.pdf",
            "page": 3,
            "modality": "text",
        },
        {
            "id": "c2",
            "score": 0.72,
            "content": "Maximum clamping voltage is 43V at 1A.",
            "file_name": "datasheet.pdf",
            "page": 4,
            "modality": "text",
        },
        {
            "id": "c3",
            "score": 0.65,
            "content": "Package type is SOD-323.",
            "file_name": "datasheet.pdf",
            "page": 1,
            "modality": "text",
        },
    ]


# ============================================================
# Mock 基础设施
# ============================================================

@pytest.fixture
def mock_chroma_client():
    """模拟 ChromaClientSingleton。"""
    mock = MagicMock()
    mock._client = MagicMock()  # 模拟已连接状态
    mock.is_connected = True
    mock.insert.return_value = {"insert_count": 1, "ids": ["test_id"]}
    mock.hybrid_search.return_value = [
        {"id": "c1", "score": 0.85, "entity": {
            "id": "c1", "content": "ESD protection...", "file_name": "test.pdf",
            "page": 1, "modality": "text", "searchable_text": "ESD...",
            "doc_id": "d1", "section": "Overview", "metadata": {},
        }},
        {"id": "c2", "score": 0.72, "entity": {
            "id": "c2", "content": "Clamping voltage...", "file_name": "test.pdf",
            "page": 2, "modality": "text", "searchable_text": "Clamping...",
            "doc_id": "d1", "section": "Specs", "metadata": {},
        }},
    ]
    mock.list_documents.return_value = [
        {"file_name": "test.pdf", "chunk_count": 5},
    ]
    mock.get_collection_stats.return_value = {"name": "document_chunks", "row_count": 5}
    mock.ensure_loaded.return_value = None
    mock.delete_by_file_name.return_value = 3
    mock.image_search.return_value = []
    return mock


@pytest.fixture
def mock_openai_client():
    """模拟 OpenAI 客户端（用于 Generator / Reranker / Parser）。"""
    mock = MagicMock()

    # 模拟 chat.completions.create 返回
    mock_choice = MagicMock()
    mock_choice.message.content = "Based on the datasheet, the breakdown voltage is 27V."
    mock_choice.index = 0

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock.chat.completions.create.return_value = mock_completion
    return mock


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """自动注入环境变量，确保测试不依赖 .env 文件。"""
    monkeypatch.setenv("ONEAPI_BASE_URL", "https://models.example.com/v1")
    monkeypatch.setenv("ONEAPI_API_KEY", "mock-api-key")
    monkeypatch.setenv("CHROMA_DB_PATH", "./mock_chroma_data")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")  # 减少测试日志噪音


# ============================================================
# Mock 生成器 & 检索器（供工具层测试使用）
# ============================================================

@pytest.fixture
def mock_generator():
    """模拟 Generator.generate()，返回固定 JSON 参数表。"""
    mock = MagicMock()
    mock.generate.return_value = (
        '{"反向击穿电压": "27V", "钳位电压": "43V", "封装类型": "SOD-323"}'
    )
    return mock


@pytest.fixture
def mock_retriever():
    """模拟 Retriever，返回固定 chunk 列表。"""
    mock = MagicMock()
    mock.retrieve_all_by_doc.return_value = [
        {
            "id": "c1", "doc_id": "doc_a", "file_name": "doc_a.pdf",
            "page": 1, "modality": "text",
            "content": "Breakdown voltage: 27V",
            "searchable_text": "Breakdown voltage: 27V",
            "section": "Electrical", "metadata": {},
        },
        {
            "id": "c2", "doc_id": "doc_a", "file_name": "doc_a.pdf",
            "page": 2, "modality": "text",
            "content": "Package: SOD-323",
            "searchable_text": "Package: SOD-323",
            "section": "Package", "metadata": {},
        },
    ]
    mock.retrieve.return_value = [
        {"id": "c1", "score": 0.9, "doc_id": "doc_a", "file_name": "doc_a.pdf",
         "page": 1, "modality": "text", "content": "ESD protection...",
         "searchable_text": "ESD protection...", "section": "Overview", "metadata": {}},
    ]
    return mock


@pytest.fixture
def mock_chroma():
    """模拟 ChromaClientSingleton.get_all_by_doc_id。"""
    mock = MagicMock()
    mock.get_all_by_doc_id.return_value = [
        {
            "id": "c1", "doc_id": "doc_a", "file_name": "doc_a.pdf",
            "page": 1, "modality": "text",
            "content": "Breakdown voltage: 27V", "section": "Electrical",
        },
        {
            "id": "c2", "doc_id": "doc_a", "file_name": "doc_a.pdf",
            "page": 2, "modality": "text",
            "content": "Package: SOD-323", "section": "Package",
        },
    ]
    return mock
