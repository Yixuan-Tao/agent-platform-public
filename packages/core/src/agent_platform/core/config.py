"""全局配置（pydantic-settings，从仓库根目录 .env / 环境变量读取）。

LLM API 统一配置入口（全部服务共用）：
- LLM_API_KEY / LLM_BASE_URL：OpenAI 兼容网关密钥与地址
- CHAT_MODEL / VL_MODEL / EMBED_MODEL / RERANK_MODEL：各角色模型
- MODEL_OVERRIDES：按 Agent 覆盖模型（JSON）
旧变量名（OPENAI_API_KEY / OPENAI_BASE_URL / DEFAULT_MODEL）保留为兼容别名。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[5]


class Settings(BaseSettings):
    """统一配置。所有包通过 get_settings() 读取。"""

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "agent-platform"
    env: str = "dev"

    # 网关
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "dev-key-change-me"
    # 前端地址：网关根路径索引里回给浏览器，避免直接把 API 端口当页面访问
    frontend_url: str = "http://localhost:3000"

    # ===== 认证 / 审计 / LAN 安全 =====
    lan_enabled: bool = False
    tls_cert_file: str = ""
    tls_key_file: str = ""
    initial_admin_username: str = ""
    initial_admin_password: str = ""
    admin_api_key: str = ""
    model_gateway_allowed_hosts: str = ""

    # ===== LLM（统一配置，全项目共用；兼容旧变量名）=====
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY"),
    )
    llm_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"),
    )
    chat_model: str = Field(
        default="qwen3.7-plus",
        validation_alias=AliasChoices("CHAT_MODEL", "DEFAULT_MODEL"),
    )
    vl_model: str = Field(
        default="qwen3-vl-plus",
        validation_alias=AliasChoices("VL_MODEL", "QWEN_VL_MODEL"),
    )
    embed_model: str = Field(
        default="text-embedding-v4",
        validation_alias=AliasChoices("EMBED_MODEL", "TEXT_EMBED_MODEL"),
    )
    embedding_gateway_url: str = Field(default="", validation_alias="ONEAPI_BASE_URL")
    embedding_gateway_key: str = Field(default="", validation_alias="ONEAPI_API_KEY")
    image_embed_model: str = Field(default="tongyi-embedding-vision-flash")
    rerank_model: str = Field(default="qwen3-rerank")
    model_overrides: dict[str, str] = Field(default_factory=dict)
    llm_max_retries: int = 3
    llm_timeout: float = 60.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_seconds: int = 60

    # 数据库
    database_url: str = "postgresql+asyncpg://agent:agent@localhost:5432/agent_platform"

    # 记忆
    redis_url: str = "redis://localhost:6379/0"
    use_fakeredis: bool = True
    short_memory_ttl_seconds: int = 3600
    # 长期记忆 few-shot（可选增强，默认开启；检索失败/超限自动降级为空，等价现状）
    memory_few_shot_enabled: bool = True
    memory_few_shot_k: int = 3
    memory_few_shot_min_score: float = 0.5
    memory_few_shot_max_tokens: int = 500

    # RAG 服务
    rag_service_url: str = "http://localhost:8010"
    # 网关调用 RAG 的服务凭据：
    # - 本机模式作为 X-Local-Token（RAG 的 require_local_token）
    # - LAN 模式作为 Bearer（RAG 的 require_admin）
    rag_service_token: str = ""
    rag_request_timeout: float = 120.0
    # Opt-in protocol v1: freeze logical corpus and delegate durable model/token grants.
    rag_runtime_contract_enabled: bool = False
    rag_delegate_model_requests: int = Field(default=8, ge=1, le=32)
    rag_delegate_tokens: int = Field(default=32000, ge=1, le=100000)
    chroma_dir: str = "chroma_data"

    # Jira
    jira_url: str = ""
    jira_user: str = ""
    jira_token: str = ""

    # 工具模块（运行时网关 / 动态注册 / MCP 客户端 / 插件）
    tool_call_timeout: float = 30.0
    plugin_dir: str = ""
    tool_config_json: str = Field(default="", validation_alias=AliasChoices("TOOL_CONFIG_JSON", "TOOL_CONFIG"))
    mcp_servers_json: str = Field(default="", validation_alias=AliasChoices("MCP_SERVERS_JSON", "MCP_SERVERS"))

    # 编排（Router-as-ReAct 双模式，默认关闭，保持现有单 Agent 路由）
    use_router_react: bool = False
    max_agent_calls: int = 4
    plan_mode: bool = True
    max_plan_steps: int = 6
    # 回答流式：开启后按模型产出的 token 增量下发（首字延迟更真实）；
    # 此时质检只记录不回滚，因为内容已经发送给用户。默认关闭。
    answer_streaming: bool = False
    # 每次请求允许调用的子 Agent 次数上限（Router-ReAct / Plan 共用）
    agent_call_budget: int = 8

    # Langfuse（可选）
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # 日志
    log_level: str = "INFO"
    log_dir: str = "logs"
    # 运行数据目录（access.db / audit.db 等）。相对路径按 apps/api 目录解析，
    # 容器中通过 RUNTIME_DIR 指向挂载卷，避免重建后账号与审计丢失。
    runtime_dir: str = "runtime_data"
    runs_enabled: bool = False
    chat_via_runs: bool = False
    experiments_enabled: bool = False
    sandbox_enabled: bool = False
    semantic_memory_enabled: bool = False
    tool_discovery_enabled: bool = False
    default_run_profile: str = "B0"
    service_tool_scopes: list[str] = []
    user_tool_scopes: list[str] = ["jira:read", "jira:create"]

    def llm_kwargs(self) -> dict[str, Any]:
        """构造 AsyncOpenAI 客户端参数。"""
        return {
            "api_key": self.llm_api_key or "sk-none",
            "base_url": self.llm_base_url,
            "timeout": self.llm_timeout,
        }

    def resolve_model(self, agent_name: str | None = None) -> str:
        """按 Agent 覆盖模型，未覆盖则用默认模型。"""
        if agent_name and agent_name in self.model_overrides:
            return self.model_overrides[agent_name]
        return self.chat_model

    # ---- 兼容别名（旧代码引用）----
    @property
    def openai_api_key(self) -> str:
        return self.llm_api_key

    @property
    def openai_base_url(self) -> str:
        return self.llm_base_url

    @property
    def default_model(self) -> str:
        return self.chat_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
