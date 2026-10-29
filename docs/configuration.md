# 配置说明（LLM API 统一入口）

整个项目的配置统一放在**仓库根目录 `.env`**（复制自 `.env.example`），网关、RAG 知识服务、记忆层全部从同一份配置读取。

## LLM API 统一变量（全项目共用）

| 变量 | 默认值 | 用途 |
|---|---|---|
| `LLM_API_KEY` | 空 | OpenAI 兼容网关密钥（OpenAI / OneAPI / DashScope / FastGPT / DeepSeek 均可） |
| `LLM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 网关地址（阿里云百炼默认） |
| `CHAT_MODEL` | `qwen3.7-plus` | 聊天 / Agent 默认模型 |
| `VL_MODEL` | `qwen3-vl-plus` | 图片理解（RAG 多模态解析） |
| `EMBED_MODEL` | `text-embedding-v4` | 文本向量 |
| `IMAGE_EMBED_MODEL` | `tongyi-embedding-vision-flash` | 图片向量 |
| `RERANK_MODEL` | `qwen3-rerank` | 检索重排 |
| `RERANK_BASE_URL` | 由 `LLM_BASE_URL` 推导 | 重排专用网关（compatible-mode → compatible-api） |
| `EMBED_BATCH_SIZE` | `10` | 单次 embedding 请求最大输入条数 |
| `CHAT_TIMEOUT` | `120` | RAG 生成/改写调用超时（秒；默认 30，阿里云大报告场景建议 120） |
| `MODEL_OVERRIDES` | `{}` | 按 Agent 覆盖聊天模型，JSON 对象 |

其余非 LLM 配置（数据库、Redis、端口、Jira 等）也都在根 `.env`，见 `.env.example` 注释。

## 阿里云百炼（DashScope）配置

```dotenv
LLM_API_KEY=sk-xxx            # 或使用 DASHSCOPE_API_KEY
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL=qwen3.7-plus
VL_MODEL=qwen3-vl-plus
EMBED_MODEL=text-embedding-v4
RERANK_MODEL=qwen3-rerank
RERANK_BASE_URL=https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-api/v1
```

- **域名**：旧域名 `https://dashscope.aliyuncs.com/compatible-mode/v1` 无需 WorkspaceId；生产建议迁移到新业务空间域名 `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`（北京）或新加坡 `.../ap-southeast-1.maas.aliyuncs.com/...`。
- **文本向量**：`text-embedding-v4`（默认 1024 维，`dimensions` 可调；单请求最大 10 行，对应 `EMBED_BATCH_SIZE=10`）。
- **重排**：`qwen3-rerank` 走 `compatible-api/v1/reranks`（不是 `compatible-mode`），请求体扁平 `{model, query, documents, top_n}`，响应 `results[].index/relevance_score`；代码会按 `LLM_BASE_URL` 自动推导，也可用 `RERANK_BASE_URL` 显式指定。若网关不可用，自动降级到 embedding 余弦相似度 / 原始分数。
- **视觉对话**：`qwen3-vl-plus` 等 Qwen-VL 模型走 OpenAI 兼容 `chat/completions`。
- **图片向量**：`tongyi-embedding-vision` / `qwen3-vl-embedding` 等多模态 Embedding **不支持 OpenAI 兼容接口**；在 DashScope 域名下代码自动禁用图片向量（走文本向量/零向量降级），不影响其他功能。
- **API Key**：百炼控制台创建，官方环境变量名为 `DASHSCOPE_API_KEY`（`sk-` 开头）。

## 兼容旧变量

不填统一变量时，以下旧变量仍生效（优先级：统一变量 > 旧变量 > 默认值）：

- 网关：`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `DEFAULT_MODEL`
- RAG 服务：`ONEAPI_BASE_URL` / `ONEAPI_API_KEY` / `QWEN_CHAT_MODEL` / `QWEN_VL_MODEL` / `TEXT_EMBED_MODEL` / `IMAGE_EMBED_MODEL` / `RERANK_MODEL`

## 配置读取优先级

进程环境变量（Docker/Shell）> 本地 `packages/rag/.env`（仅 RAG 服务）> 仓库根 `.env` > 默认值

## 常见网关示例

阿里云百炼：

```dotenv
LLM_API_KEY=sk-...
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL=qwen3.7-plus
VL_MODEL=qwen3-vl-plus
EMBED_MODEL=text-embedding-v4
RERANK_MODEL=qwen3-rerank
RERANK_BASE_URL=https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-api/v1
```

OpenAI：

```dotenv
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4o-mini
```

OneAPI / 企业内网网关：

```dotenv
LLM_API_KEY=sk-...
LLM_BASE_URL=https://models.example.com/v1
CHAT_MODEL=qwen3.7-plus
VL_MODEL=qwen3-vl-plus
EMBED_MODEL=text-embedding-v4
RERANK_MODEL=qwen3-rerank
```

DeepSeek：

```dotenv
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.deepseek.com/v1
CHAT_MODEL=deepseek-chat
```

> 注意：若网关不提供 VL / Embedding / Rerank 能力，RAG 的多模态解析与重排会降级（见 `packages/rag` 日志），聊天仍可用。

## 评测裁判（DeepSeek LLM-as-Judge）

评测集运行器（`packages/rag/evaluation/run_testset.py --judge`）会用独立的 DeepSeek 模型
（默认 `deepseek-chat`，OpenAI 兼容接口 `https://api.deepseek.com`）按评分规则对每条 RAG
回答打分：忠实度 / 相关性 / 完整性 / 引用质量 / 总分（0-5），并给出是否通过（阈值
`JUDGE_PASS_THRESHOLD`，默认 4.0）与改进建议。

```dotenv
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

运行方式：

```powershell
cd packages/rag
python evaluation/run_testset.py --testset evaluation/testset_v4.json --judge
```
