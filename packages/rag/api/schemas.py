"""
多模态 RAG 系统 — API 层 Pydantic 模型 (Task 7)

设计原则：
    - 兼容 OpenAI Chat Completions 请求/响应字段
    - 业务接口（ingest / search / documents）模型精简
    - 字段允许扩展（Extra.allow），对前向兼容友好
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# 通用基类
# ============================================================

class _Base(BaseModel):
    """统一开启前向兼容：忽略未知字段。"""
    model_config = ConfigDict(extra="allow", protected_namespaces=())


# ============================================================
# OpenAI Chat Completions 兼容模型
# ============================================================

class ChatMessage(_Base):
    """单条消息；content 简化为 str/None。"""
    role: str
    content: Optional[str] = None


class ToolFunctionSpec(_Base):
    """tool.function 子结构。"""
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class ToolDefinition(_Base):
    """OpenAI tool 定义。"""
    type: str = "function"
    function: ToolFunctionSpec


class ToolCallFunction(_Base):
    """tool_call.function 子结构。"""
    name: str
    arguments: str = "{}"


class ToolCall(_Base):
    """OpenAI tool_call 项。"""
    id: str
    type: str = "function"
    function: ToolCallFunction


class ChatCompletionRequest(_Base):
    """OpenAI Chat Completions 兼容请求体。"""
    model: str
    messages: list[ChatMessage]
    tools: Optional[list[ToolDefinition]] = None
    stream: bool = False
    temperature: float = 0.2
    # 业务扩展：可指定 RAG top_k / 是否启用 rerank
    top_k: Optional[int] = Field(None, ge=1, le=20)
    use_rerank: Optional[bool] = None
    language: Optional[str] = Field(None, pattern="^zh$")


class Choice(_Base):
    """单条 choice。"""
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(_Base):
    """token 用量（这里填 0 即可，真实场景由上游网关提供）。"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(_Base):
    """OpenAI Chat Completions 兼容响应体。"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
    # ---- 证据链路扩展字段 ----
    run_id: str = ""
    answer_status: str = "ok"
    citations: list[dict] = Field(default_factory=list)


class ChatCompletionChunk(_Base):
    """流式响应的单个 chunk。"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict[str, Any]] = Field(default_factory=list)


# ============================================================
# 业务接口模型
# ============================================================

class IngestResponse(_Base):
    """摄入结果响应（同步模式，兼容旧接口）。"""
    status: str
    file_name: str = ""
    doc_id: str = ""
    chunk_count: int = 0


class IngestAcceptedResponse(_Base):
    """异步摄入接受响应：立即返回，后台处理。"""
    task_id: str
    status: str = "processing"  # processing | done | error
    file_name: str = ""


class IngestStatusResponse(_Base):
    """异步摄入状态查询响应。"""
    task_id: str
    status: str                  # processing | done | error
    file_name: str = ""
    progress: str = ""           # 进度描述，如 "解析中..." / "向量化中..."
    result: Optional[dict] = None  # 完成时包含 doc_id / chunk_count
    error: Optional[str] = None    # 失败时的错误信息


class SearchRequest(_Base):
    """检索请求体。"""
    query: str
    top_k: Optional[int] = Field(None, ge=1, le=20)
    filter: Optional[str] = None
    language: Optional[str] = Field(None, pattern="^zh$")
    use_rerank: bool = True


class SearchResultItem(_Base):
    """单条检索结果。"""
    id: str
    score: float = 0.0
    file_name: str = ""
    page: int = 0
    modality: str = "text"
    content: str = ""


class SearchResponse(_Base):
    """检索结果响应。"""
    query: str
    results: list[SearchResultItem]
    answer: Optional[str] = None
    language: str = "en"
    used_rerank: bool = False


class DocumentListItem(_Base):
    """文档列表项。"""
    file_name: str
    chunk_count: int
    # 供 /compare/* 等按 doc_id 定位文档的接口使用（旧客户端可忽略）
    doc_id: str = ""


class DocumentListResponse(_Base):
    """文档列表响应。"""
    total: int
    documents: list[DocumentListItem]


class DeleteResponse(_Base):
    """删除结果响应。"""
    status: str
    deleted_count: int = 0
    file_name: Optional[str] = None


class ErrorResponse(_Base):
    """统一错误响应。"""
    error: str
    detail: Optional[str] = None


class HealthResponse(_Base):
    """健康检查响应。"""
    status: str
    version: str
    chroma_connected: bool
    document_count: int
    chunk_count: int


# ============================================================
# 元器件替换判断接口模型
# ============================================================

class ComponentCompareRequest(_Base):
    """元器件替换判断请求体。"""
    doc_a: str = Field(..., description="文档 A 的 doc_id")
    doc_b: str = Field(..., description="文档 B 的 doc_id")
    params_template: dict = Field(..., description="参数表模板 JSON，含类别/参数名/单位/容差等")
    strict_mode: bool = True
    language: str = Field(default="zh", pattern="^zh$")
    criteria_override: dict[str, Any] = Field(default_factory=dict)
    evidence_version: str = ""


class ParamComparison(_Base):
    """单条参数对比详情。"""
    param: str = Field(..., description="参数名，如 'VBR'")
    category: str = Field(..., description="参数所属类别，如 'electrical'")
    doc_a_value: Optional[str] = Field(None, description="文档 A 中该参数的值")
    doc_b_value: Optional[str] = Field(None, description="文档 B 中该参数的值")
    match: bool = Field(..., description="是否匹配")
    within_tolerance: bool = False
    note: str = ""


class ComponentCompareResponse(_Base):
    """元器件替换判断结果响应。"""
    conclusion: str = Field(..., description="结论：可替换 / 有条件替换 / 不可替换")
    conclusion_detail: str = Field(..., description="结论详细说明")
    params_comparison: list[ParamComparison] = Field(
        default_factory=list, description="逐参数对比详情"
    )
    failed_params: list[str] = Field(
        default_factory=list, description="未在文档中找到的参数"
    )
    # ---- 证据链路扩展字段 ----
    directional_conclusions: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict] = Field(default_factory=list)
    answer_status: str = "ok"
    run_id: str = ""
    evidence_version: str = ""
    coverage: dict = Field(default_factory=dict)


# ============================================================
# EMC 标准对比
# ============================================================

class EmcCompareRequest(_Base):
    """EMC 标准对比请求"""
    query: str
    doc_a: str
    doc_b: str
    language: str = "zh"
    comparison_mode: str = Field(default="standard", pattern="^(standard|test_item)$")
    test_item: str = ""
    section_hint: str = ""
    evidence_version: str = ""


class EmcCompareResponse(_Base):
    """EMC 标准对比响应"""
    answer: str
    doc_a: dict
    doc_b: dict
    compare_route: str = ""
    sections: list[dict] = []
    sources: list[dict] = []
    image_chunks: list[dict] = []
    # ---- 证据链路扩展字段 ----
    comparison_items: list[dict] = []
    citations: list[dict] = []
    answer_status: str = "ok"
    run_id: str = ""
    evidence_version: str = ""
    coverage: dict = Field(default_factory=dict)
    comparison_matrix: list[dict] = Field(default_factory=list)


# ============================================================
# EMC 表格可视化
# ============================================================

class VisualizeEmcRequest(_Base):
    """EMC 表格可视化请求"""
    query: str
    doc_id: str | None = None
    doc_a: str | None = None
    doc_b: str | None = None
    top_k: int = 5
    table_type: str | None = None


class VisualizeEmcResponse(_Base):
    """EMC 表格可视化响应"""
    html_bundle: str
    table_type: str
    datasets: list[dict] = []
    sources: list[dict] = []


__all__ = [
    "ChatMessage",
    "ToolFunctionSpec",
    "ToolDefinition",
    "ToolCallFunction",
    "ToolCall",
    "ChatCompletionRequest",
    "Choice",
    "Usage",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "IngestResponse",
    "IngestAcceptedResponse",
    "IngestStatusResponse",
    "SearchRequest",
    "SearchResultItem",
    "SearchResponse",
    "DocumentListItem",
    "DocumentListResponse",
    "DeleteResponse",
    "ErrorResponse",
    "HealthResponse",
    "ComponentCompareRequest",
    "ParamComparison",
    "ComponentCompareResponse",
    "EmcCompareRequest",
    "EmcCompareResponse",
    "VisualizeEmcRequest",
    "VisualizeEmcResponse",
]
