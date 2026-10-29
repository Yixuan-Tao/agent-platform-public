"""
全局配置管理 — 通过 _Config 延迟加载，避免导入时固化环境变量
"""
import os
from pathlib import Path
from dotenv import load_dotenv


def locate_repo_env(start: Path) -> "Path | None":
    """自 ``start`` 向上查找最近的 ``.env``；找不到返回 None。

    本地布局是 ``<repo>/packages/rag/config.py``，仓库根 ``.env`` 位于
    ``parents[2]``；但容器里本文件被复制成 ``/app/config.py``，层级不足，
    直接取 ``parents[2]`` 会抛 IndexError，导致 RAG 容器启动即退出。
    因此改为逐级向上探测，容器内由环境变量提供配置。
    """
    for parent in start.resolve().parents:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


class _Config:
    """延迟加载配置类：属性在首次访问时从环境变量读取，便于测试注入。"""

    def __init__(self) -> None:
        # 优先级（高→低）：进程环境变量 > 本地 packages/rag/.env > 仓库根 .env（统一 LLM 配置）
        load_dotenv()
        repo_env = locate_repo_env(Path(__file__))
        if repo_env is not None:
            load_dotenv(repo_env)

    # --- OneAPI 网关 ---
    @property
    def ONEAPI_BASE_URL(self) -> str:
        return os.getenv("ONEAPI_BASE_URL") or os.getenv(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    @property
    def ONEAPI_API_KEY(self) -> str:
        return (
            os.getenv("ONEAPI_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY", "")
        )

    @property
    def CHAT_BASE_URL(self) -> str:
        """??????????/Query ??/?????????????
        ??? ONEAPI ???????????? deepseek ?? API??"""
        return os.getenv("CHAT_BASE_URL") or self.ONEAPI_BASE_URL

    @property
    def CHAT_API_KEY(self) -> str:
        return os.getenv("CHAT_API_KEY") or self.ONEAPI_API_KEY

    @property
    def CHAT_IS_DEEPSEEK(self) -> bool:
        """??????? DeepSeek ?? API??? extra_body ??????"""
        return "api.deepseek.com" in (self.CHAT_BASE_URL or "")

    @property
    def CHAT_EXTRA_BODY(self) -> dict:
        """?????????deepseek ?? API ? thinking ???????
        ???DashScope?? enable_thinking?"""
        if self.CHAT_IS_DEEPSEEK:
            return {"thinking": {"type": "disabled"}}
        return {"enable_thinking": False}

    @property
    def MODEL_GATEWAY_ALLOWED_HOSTS(self) -> list[str]:
        """可信模型网关主机白名单；为空时跳过主机校验（开发模式）。"""
        raw = os.getenv("MODEL_GATEWAY_ALLOWED_HOSTS", "")
        return [host.strip() for host in raw.split(",") if host.strip()]

    @property
    def LAN_ENABLED(self) -> bool:
        """局域网共享模式：强制 HTTPS、要求 Bearer 会话、写操作需管理员。"""
        return os.getenv("LAN_ENABLED", "false").lower() in ("true", "1", "yes")

    @property
    def INITIAL_ADMIN_USERNAME(self) -> str:
        return os.getenv("INITIAL_ADMIN_USERNAME", "")

    @property
    def INITIAL_ADMIN_PASSWORD(self) -> str:
        return os.getenv("INITIAL_ADMIN_PASSWORD", "")

    @property
    def ADMIN_API_KEY(self) -> str:
        """可选的自动化管理员密钥；只从环境读取，移除或轮换即可失效。"""
        return os.getenv("ADMIN_API_KEY", "")

    @property
    def TLS_CERT_FILE(self) -> str:
        return os.getenv("TLS_CERT_FILE", "")

    @property
    def TLS_KEY_FILE(self) -> str:
        return os.getenv("TLS_KEY_FILE", "")

    # --- 端点路径 ---
    @property
    def CHAT_PATH(self) -> str:
        return os.getenv("CHAT_PATH", "/v1/chat/completions")

    @property
    def EMBEDDINGS_PATH(self) -> str:
        return os.getenv("EMBEDDINGS_PATH", "/v1/embeddings")

    @property
    def MODELS_PATH(self) -> str:
        return os.getenv("MODELS_PATH", "/v1/models")

    # --- 模型名称 ---
    @property
    def QWEN_VL_MODEL(self) -> str:
        return os.getenv("QWEN_VL_MODEL") or os.getenv("VL_MODEL", "qwen3-vl-plus")

    @property
    def QWEN_CHAT_MODEL(self) -> str:
        return os.getenv("QWEN_CHAT_MODEL") or os.getenv("CHAT_MODEL", "qwen3.7-plus")

    @property
    def TEXT_EMBED_MODEL(self) -> str:
        return os.getenv("TEXT_EMBED_MODEL") or os.getenv("EMBED_MODEL", "text-embedding-v4")

    @property
    def IMAGE_EMBED_MODEL(self) -> str:
        # 百炼多模态向量：2026-03-06 版支持 dimension=1024（默认 1152）
        return os.getenv("IMAGE_EMBED_MODEL", "tongyi-embedding-vision-plus-2026-03-06")

    @property
    def DASHSCOPE_API_KEY(self) -> str:
        """DashScope 原生协议专用密钥：优先 DASHSCOPE_API_KEY，其次 LLM_API_KEY / ONEAPI_API_KEY。"""
        return (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("ONEAPI_API_KEY", "")
        )

    @property
    def DASHSCOPE_WORKSPACE_ID(self) -> str:
        """百炼业务空间 ID（控制台可见）；非空时拼接 maas.aliyuncs.com 新域名原生端点。"""
        return os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()

    @property
    def DASHSCOPE_MULTIMODAL_URL(self) -> str:
        """多模态 Embedding 原生端点（不支持 OpenAI 兼容接口）。
        优先级：显式 DASHSCOPE_MULTIMODAL_URL > 新业务空间域名 > 旧 dashscope 域名。
        """
        explicit = os.getenv("DASHSCOPE_MULTIMODAL_URL")
        if explicit:
            return explicit.rstrip("/")
        ws = self.DASHSCOPE_WORKSPACE_ID
        if ws:
            return (
                f"https://{ws}.cn-beijing.maas.aliyuncs.com/api/v1/services/"
                "embeddings/multimodal-embedding/multimodal-embedding"
            )
        return "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"

    @property
    def ENABLE_VISUAL_SEARCH(self) -> bool:
        """是否在主检索链路中并入图像向量检索（文搜图/图搜图）；默认关闭，小样本验证后再开启。"""
        return os.getenv("ENABLE_VISUAL_SEARCH", "false").lower() in ("true", "1", "yes")

    # --- ChromaDB ---
    @property
    def CHROMA_DB_PATH(self) -> str:
        return os.getenv("CHROMA_DB_PATH", "./chroma_data")

    # --- 向量维度 ---
    DENSE_EMBED_DIM = 1024

    @property
    def IMAGE_EMBED_DIM(self) -> int:
        """多模态向量维度：tongyi-embedding-vision-plus-2026-03-06 支持 1152/1024/512/256/128/64。"""
        return int(os.getenv("IMAGE_EMBED_DIM", "1024"))

    # --- 服务配置 ---
    @property
    def HOST(self) -> str:
        # 显式设置优先（Docker 部署传 HOST=0.0.0.0）；未设置时默认仅本机访问，
        # 局域网共享（LAN_ENABLED=true）才默认 0.0.0.0
        explicit = os.getenv("HOST")
        if explicit:
            return explicit
        if os.getenv("LAN_ENABLED", "false").lower() not in ("true", "1", "yes"):
            return "127.0.0.1"
        return "0.0.0.0"

    @property
    def RUNTIME_DIR(self) -> str:
        return os.getenv("RUNTIME_DIR", "./runtime_data")

    @property
    def MAX_UPLOAD_BYTES(self) -> int:
        return int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

    @property
    def MAX_TOP_K(self) -> int:
        return int(os.getenv("MAX_TOP_K", "20"))

    @property
    def MAX_INGEST_TASKS(self) -> int:
        """活跃摄入任务上限（含排队任务），超限返回 429。"""
        return int(os.getenv("MAX_INGEST_TASKS", "20"))

    @property
    def PORT(self) -> int:
        return int(os.getenv("PORT", "8010"))

    @property
    def LOG_LEVEL(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")

    # --- 重排序 ---
    @property
    def RERANK_MODEL(self) -> str:
        return os.getenv("RERANK_MODEL", "qwen3-rerank")

    # 重排输入是否拼"《文件名》 章节：…"语境前缀（修复表格块无语境问题）
    @property
    def RERANK_CONTEXT_PREFIX_ENABLED(self) -> bool:
        return os.getenv("RERANK_CONTEXT_PREFIX_ENABLED", "true").lower() in ("true", "1", "yes")

    @property
    def RERANK_BASE_URL(self) -> str:
        """Rerank gateway; default derived from ONEAPI_BASE_URL (compatible-mode -> compatible-api)."""
        explicit = os.getenv("RERANK_BASE_URL")
        if explicit:
            return explicit.rstrip("/")
        base = (self.ONEAPI_BASE_URL or "").rstrip("/")
        if base.endswith("/compatible-mode/v1"):
            return base[: -len("/compatible-mode/v1")] + "/compatible-api/v1"
        return base

    @property
    def EMBED_BATCH_SIZE(self) -> int:
        """Max inputs per embeddings request; default 10 (Aliyun text-embedding-v4 limit)."""
        return int(os.getenv("EMBED_BATCH_SIZE", "10"))

    # --- 检索参数 ---
    RETRIEVAL_TOP_K = 40
    # 对比/综合分析场景：召回更多候选，避免关键文档被挤出 top-K
    COMPARE_RETRIEVAL_TOP_K = 60
    RERANK_TOP_K = 5
    # 综合分析场景：保留更多候选给 LLM，提供更全面的上下文
    COMPREHENSIVE_RERANK_TOP_K = 8
    MAX_CHUNK_TOKENS = 768

    # --- 两阶段对比（参数提取 + 对比）专用配置 ---
    # 参数提取 prompt 的最大上下文 token 数（避免爆 LLM 窗口）
    EXTRACT_MAX_CONTEXT_TOKENS = 12000
    # 全量检索的分页大小（ChromaDB collection.get 单次 limit）
    CHROMA_FETCH_PAGE_SIZE = 100
    # 提取阶段允许的最大 chunk 数（兜底，防止超长文档 OOM）
    MAX_CHUNKS_PER_DOC_FOR_EXTRACT = 500

    # --- 两阶段对比（标准试验项提取 → 对比生成）新增配置 ---
    # 对比/枚举题最终上下文数（Anthropic 实测 20 > 10 > 5）
    @property
    def COMPARE_FINAL_TOP_K(self) -> int:
        return int(os.getenv("COMPARE_FINAL_TOP_K", "28"))

    # 阶段A：每个标准组（族/版本）提取 profile 时每文件检索的 chunk 数
    @property
    def PROFILE_PER_FILE_TOP_K(self) -> int:
        return int(os.getenv("PROFILE_PER_FILE_TOP_K", "4"))

    # 阶段A：每组进入提取的 chunk 上限（阶段B 上下文集 = 各组并集，需保持一致以便引用可核验）
    @property
    def PROFILE_CHUNKS_PER_GROUP(self) -> int:
        return int(os.getenv("PROFILE_CHUNKS_PER_GROUP", "14"))

    # 阶段A：单组参与提取的原始候选上限（重排前）
    @property
    def PROFILE_MAX_RAW_CHUNKS_PER_GROUP(self) -> int:
        return int(os.getenv("PROFILE_MAX_RAW_CHUNKS_PER_GROUP", "48"))

    # 单试验项对比（BCI/ESD/RE…）：每族定向检索 chunk 数
    @property
    def ITEM_TARGETED_TOP_K(self) -> int:
        return int(os.getenv("ITEM_TARGETED_TOP_K", "6"))

    # 阶段A profile 缓存 TTL（秒）/ 上限
    @property
    def PROFILE_CACHE_TTL(self) -> float:
        return float(os.getenv("PROFILE_CACHE_TTL", "300"))

    @property
    def PROFILE_CACHE_MAX(self) -> int:
        return int(os.getenv("PROFILE_CACHE_MAX", "64"))

    # 两阶段对比总开关；False 回退单次对比生成
    @property
    def TWO_STAGE_COMPARE_ENABLED(self) -> bool:
        return os.getenv("TWO_STAGE_COMPARE_ENABLED", "true").lower() in ("true", "1", "yes")

    # 生成端迭代补检（回答出现缺失标记时定向二次检索重写）
    @property
    def RETRY_MAX_ROUNDS(self) -> int:
        return int(os.getenv("RETRY_MAX_ROUNDS", "1"))

    @property
    def RETRY_TARGETED_TOP_K(self) -> int:
        return int(os.getenv("RETRY_TARGETED_TOP_K", "6"))

    # 生成温度（数值/标准类取低值）
    @property
    def GENERATION_TEMPERATURE(self) -> float:
        return float(os.getenv("GENERATION_TEMPERATURE", "0.2"))

    # --- Requery (query rewriting + multi-query fusion) ---
    # 总开关：False 时完全跳过 requery 步骤（保留单 query 检索原流程）
    @property
    def REQUERY_ENABLED(self) -> bool:
        return os.getenv("REQUERY_ENABLED", "true").lower() in ("true", "1", "yes")

    # 短查询阈值：query 字符数 < 该值时优先触发改写
    @property
    def REQUERY_MIN_QUERY_LENGTH(self) -> int:
        return int(os.getenv("REQUERY_MIN_QUERY_LENGTH", "30"))

    # 改写生成的额外 query 数（不含原 query）；建议 1-2，避免稀释检索信号
    @property
    def REQUERY_N_VARIANTS(self) -> int:
        return int(os.getenv("REQUERY_N_VARIANTS", "2"))

    # RRF 融合常数 k（参考 Cormack et al. 2009，60 是经验最优值）
    REQUERY_FUSION_K: int = 60
    # 改写结果缓存 TTL（秒）
    REQUERY_CACHE_TTL: float = 300.0
    # 改写缓存最大条目数（超过时按 FIFO 淘汰一半）
    REQUERY_CACHE_MAX: int = 128
    # 口语化词正则：query 命中即视为口语化，绕过长度阈值触发改写
    REQUERY_COLLOQUIAL_RE: str = r"怎么|如何|为什么|是什么|哪.{0,2}个|why|how|which|what|多少"

    # --- 切块策略（heading-aware + parent-child） ---
    # 子 chunk token 上限（child chunk 按此切分）
    CHUNK_MAX_TOKENS: int = 768
    # 表格类 chunk 的 token 上限（保留完整 Markdown 时需更大窗口）
    TABLE_CHUNK_MAX_TOKENS: int = 3000
    # 表格 chunk 的字符硬上限：split_by_tokens 按英文约 4 字符/token 估算时，
    # 3000 token 可能产生 ~12000 字符的英文表格块，超出嵌入模型的有效长度。
    # 统一按 1.5 字符/token 的保守比例再切一刀，保证单块不超过 ~4500 字符。
    TABLE_CHUNK_MAX_CHARS: int = 4500
    # EMC 文档专用文本 chunk 上限（参数密集，适当增大窗口）
    EMC_CHUNK_MAX_TOKENS: int = 1200
    # 是否在表格 chunk 的 content 中保留原始表格 Markdown（True 时 content 含 [原始表格] + [表格摘要]）
    ENABLE_TABLE_FULL_CONTENT: bool = True
    # 父 chunk token 上限（超过时按段落切分为多个 parent）
    PARENT_MAX_TOKENS: int = 2000
    # 跨页同标题合并开关（True：H1/H2 跨页合并为单一 chunk）
    ENABLE_HEADING_MERGE: bool = True
    # 表格独立成 chunk（True：表格不与文本聚合，独立入库）
    ENABLE_TABLE_ISOLATION: bool = True
    # 父子双层结构开关（True：父 chunk 仅做上下文，不入向量库）
    ENABLE_PARENT_CHILD: bool = True
    # 父 chunk 是否入向量库（False：parent 仅做 child 的上下文锚点）
    INCLUDE_PARENT_IN_INDEX: bool = False
    # heading 切块最小层级（H1+H2 切块，H3 及以下不切）
    CHUNK_HEADING_MIN_LEVEL: int = 2
    # child chunk 检索时附加的 parent context 长度（字符数）
    PARENT_CONTEXT_SNIPPET_CHARS: int = 600

    # --- 章节结构识别（标准类大文档对比路由用） ---
    # 总开关：True 时启用 heading 文本的章节编号/主题识别，写入 DocumentChunk.section_*
    ENABLE_SECTION_RECOGNITION: bool = True
    # 章节主题词路由表：topic → 触发该 topic 的关键词列表
    # 用于 query 解析阶段识别"用户在问哪一类章节"，路由对比流程聚焦相关 chunk
    SECTION_TOPIC_KEYWORDS: dict[str, list[str]] = {
        # --- 抗扰度类（Immunity）---
        "immunity.bci": [
            "BCI", "大电流注入", "Bulk Current Injection", "电流注入抗扰度",
            "电流注入", "注入电流", "BCI 测试", "BCI test", "BCI test method",
            "ISO 11451", "ISO 11452", "11451", "11452",
        ],
        "immunity.radiated": [
            "辐射抗扰度", "Radiated Immunity", "Radiated Susceptibility", "RI",
            "辐射敏感度", "抗辐射干扰", "辐射骚扰抗扰度", "RI 测试",
            "ALSE", "电波暗室", "Anechoic Chamber", "Strip Line", "带状线",
            "辐射抗扰度测试", "辐射发射", "radiated immunity", "radiated emission",
        ],
        "immunity.esd": [
            "ESD", "静电放电", "Electrostatic Discharge", "静电放电抗扰度",
            "ESD 测试", "ESD test", "IEC 61000-4-2", "61000-4-2",
            "静电保护", "静电抗扰度", "ESD 抗扰度",
        ],
        "immunity.conducted": [
            "传导抗扰度", "Conducted Immunity", "Conducted Susceptibility", "CI",
            "传导敏感度", "传导骚扰抗扰度", "传导干扰",
        ],
        # --- 发射类（Emission）---
        "emission.radiated": [
            "辐射发射", "Radiated Emission", "RE", "辐射骚扰", "辐射干扰",
            "辐射发射限值", "Radiated Emission Limit", "CISPR 25", "CISPR25",
        ],
        "emission.conducted": [
            "传导发射", "Conducted Emission", "CE", "传导骚扰", "传导干扰",
            "传导发射限值", "Conducted Emission Limit", "电压法", "电流探头法",
        ],
        # --- 限值类（Limits）---
        "limits.class_a": [
            "Class A", "Class-A", "A 等级", "等级 A", "限值 A",
        ],
        "limits.class_b": [
            "Class B", "Class-B", "B 等级", "等级 B", "限值 B",
        ],
        "limits.threshold": [
            "限值", "限值表", "Limit", "Limits", "Threshold", "阈值", "门限",
            "限值要求", "限值范围",
        ],
        # --- 测试条件（Test Setup / Method）---
        "test_setup": [
            "测试设置", "测试配置", "Test Setup", "Test Configuration",
            "试验设置", "试验配置", "测试布置", "测试环境",
        ],
        "test_method": [
            "测试方法", "试验方法", "Test Method", "Testing Method",
            "测试步骤", "Test Procedure", "测试流程", "Test Process",
            "测试要求", "Test Requirement",
        ],
    }
    # 章节主题的中文标签：把 topic 编码转成对比 prompt 中可读的中文描述
    SECTION_TOPIC_LABELS: dict[str, str] = {
        "immunity.bci": "BCI 大电流注入抗扰度",
        "immunity.radiated": "辐射抗扰度（RI）",
        "immunity.esd": "ESD 静电放电抗扰度",
        "immunity.conducted": "传导抗扰度（CI）",
        "emission.radiated": "辐射发射（RE）",
        "emission.conducted": "传导发射（CE）",
        "limits.class_a": "Class A 限值",
        "limits.class_b": "Class B 限值",
        "limits.threshold": "限值表",
        "test_setup": "测试设置",
        "test_method": "测试方法",
    }

    # --- 对比场景章节路由（按 query 命中的 topic 路由对比，未命中时降级全量） ---
    # 总开关：False 时完全等价于修改前的 compare() 行为
    ENABLE_COMPARE_SECTION_ROUTING: bool = True
    # 单 section 对比的最大 chunk 数（兜底，防止单 section 异常膨胀导致 OOM）
    MAX_CHUNKS_PER_SECTION_FOR_EXTRACT: int = 80

    # --- LLM 调用超时 ---
    @property
    def CHAT_TIMEOUT(self) -> float:
        return float(os.getenv("CHAT_TIMEOUT", "30"))

    # --- 裁判上下文上限（与生成上下文一致，避免引用无法核验） ---
    @property
    def JUDGE_CONTEXTS_MAX(self) -> int:
        return int(os.getenv("JUDGE_CONTEXTS_MAX", "28"))

    # --- PDF 解析 ---
    @property
    def LITEPARSE_OCR_ENABLED(self) -> bool:
        return os.getenv("LITEPARSE_OCR_ENABLED", "false").lower() in ("true", "1", "yes")

    @property
    def LITEPARSE_OCR_FAILURE_FATAL(self) -> bool:
        return os.getenv("LITEPARSE_OCR_FAILURE_FATAL", "false").lower() in ("true", "1", "yes")

    # --- Langfuse ---
    @property
    def LANGFUSE_PUBLIC_KEY(self) -> str:
        return os.getenv("LANGFUSE_PUBLIC_KEY", "")

    @property
    def LANGFUSE_SECRET_KEY(self) -> str:
        return os.getenv("LANGFUSE_SECRET_KEY", "")

    @property
    def LANGFUSE_HOST(self) -> str:
        return os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


config = _Config()
