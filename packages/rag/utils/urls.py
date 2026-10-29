"""
URL 构建工具：统一拼接 embedding / rerank 端点，避免 embedder / reranker 重复实现。
"""

from config import config


def build_embeddings_url() -> str:
    """拼接完整的 embeddings URL。

    ONEAPI_BASE_URL 可能带 /v1，也可能不带；
    EMBEDDINGS_PATH 默认 /v1/embeddings。
    若 BASE_URL 已以 /v1 结尾，PATH 自动去掉 /v1 前缀，避免 /v1/v1/embeddings。
    """
    base = (config.ONEAPI_BASE_URL or "").rstrip("/")
    path = config.EMBEDDINGS_PATH or "/v1/embeddings"
    if path.startswith("/v1/") and base.endswith("/v1"):
        path = path[len("/v1"):]
    return f"{base}{path}"


def build_rerank_url() -> str:
    """拼接重排 URL。

    - 阿里云 qwen3-rerank：compatible-api/v1/reranks（扁平请求体，
      响应 results[].index / relevance_score）
    - 其他网关：/v1/rerank（Jina/Cohere 风格），自动去重尾部的 /v1。
    """
    base = (config.RERANK_BASE_URL or "").rstrip("/")
    # 阿里云 qwen3-rerank 走 compatible-api/v1/reranks；其他网关按 Jina/Cohere 的 /v1/rerank
    if base.endswith("/compatible-api/v1"):
        return f"{base}/reranks"
    if base.endswith("/v1"):
        return f"{base}/rerank"
    return f"{base}/v1/rerank"
