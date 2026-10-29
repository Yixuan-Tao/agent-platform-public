# 一键部署（Docker Compose）

## 快速开始

```powershell
# 1. 在仓库根目录准备环境变量
copy .env.example .env
# 编辑 .env，至少填写 LLM_API_KEY / LLM_BASE_URL（统一 LLM 配置，见 docs/configuration.md）
# 并设置 POSTGRES_PASSWORD（无默认值，空密码会导致 postgres 启动失败，属于有意为之）
# 以及 RAG_SERVICE_TOKEN（网关调用 RAG 写操作的服务凭据）
# 默认即为阿里云百炼：LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# 2. 启动（注意 --env-file 指向仓库根 .env：compose 默认只读取 compose 文件所在目录）
docker compose --env-file .env -f deploy/docker-compose.yml up -d --build

# 3. 健康检查
curl http://localhost:8000/health
curl http://localhost:8010/health
```

## 服务清单

| 服务 | 端口 | 说明 |
|---|---|---|
| api | 8000 | FastAPI 网关（SSE 对话、会话、反馈、文档代理） |
| web | 3000 | React/Next.js 前端 |
| rag | 8010 | 多模态 RAG 知识服务（内置 ChromaDB） |
| postgres | 仅容器网络 | 会话/消息/反馈/审计数据（不发布宿主机端口） |
| redis | 仅容器网络 | 短期记忆缓存（不发布宿主机端口） |

## 必要的环境变量

| 变量 | 作用 | 说明 |
|---|---|---|
| `POSTGRES_PASSWORD` | 数据库口令 | 无默认值；必须设置，否则 postgres 拒绝启动 |
| `RAG_SERVICE_TOKEN` | 网关 → RAG 服务凭据 | 同时作为 RAG 的 `LOCAL_API_TOKEN`，两端保持一致 |
| `RAG_INTERNAL_URL` | 容器内网关访问 RAG 的地址 | 默认 `http://rag:8010`；LAN 模式设 `https://rag:8010`。容器内**不要**复用宿主机的 `RAG_SERVICE_URL` |
| `API_HOST_PORT` / `RAG_HOST_PORT` / `WEB_HOST_PORT` | 宿主端口 | 默认 `8000` / `8010` / `3000`。**若本机已有服务占用 8000**（例如 `workspace/project` 的 API），务必改这里，否则浏览器 `localhost:8000` 会打到错误的服务 |
| `NEXT_PUBLIC_API_URL` | 前端调用的网关地址 | 构建期内联，改端口后需重建 web 镜像：`docker compose ... up -d --build web` |
| `RUNTIME_DIR` | 运行数据目录 | 容器内为 `/data/runtime_data`，落在持久卷上 |
| `API_KEY` | 服务侧 `X-API-Key` | 仅代表 `service` 角色，不能访问账号管理接口 |
| `ANSWER_STREAMING` | 回答流式 | 默认 `false`。开启后按 token 增量下发（实测首 token 140.6s → 1.48s），此时质检只记录不回滚 |
| `AGENT_CALL_BUDGET` | 单请求子 Agent 调用上限 | 默认 `8` |

## 局域网共享（LAN，HTTPS + 账号鉴权）

默认仅本机访问。团队局域网共享时需要启用 HTTPS 与账号会话：

```powershell
# 1. 生成自签名证书（RAG 服务入口，api 网关可复用同一对证书）
cd packages/rag
python scripts/generate_tls_cert.py --cert runtime_data/tls/server.crt --key runtime_data/tls/server.key

# 2. 在仓库根 .env 配置（同时作用于 rag 与 api 两个服务）
LAN_ENABLED=true
TLS_CERT_FILE=runtime_data/tls/server.crt
TLS_KEY_FILE=runtime_data/tls/server.key
RAG_INTERNAL_URL=https://rag:8010
INITIAL_ADMIN_USERNAME=admin
INITIAL_ADMIN_PASSWORD=replace-with-strong-password
# 可选：供 CI/脚本使用的自动化管理员密钥
ADMIN_API_KEY=replace-with-rotatable-key
# 可选：可信模型网关主机白名单（非空时启动自检强制 HTTPS + 主机校验）
MODEL_GATEWAY_ALLOWED_HOSTS=dashscope.aliyuncs.com

# 3. 重新部署；首次信任前先核验证书指纹（服务日志会打印 SHA-256 指纹）
docker compose --env-file .env -f deploy/docker-compose.yml up -d --build
```

局域网模式下：RAG 服务与网关强制 HTTPS，控制台/网关要求 Bearer 会话登录，
文档摄入、删除、反馈等写操作需登录（管理操作用例见 `/api/v1/auth/*`），
写操作审计归属到真实用户（SQLite：`runtime_data/audit.db`）。

## 阿里云百炼（默认）

- 网关：`LLM_BASE_URL` 默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`，生产推荐新业务空间域名 `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`。
- API Key：`LLM_API_KEY` 或 `DASHSCOPE_API_KEY`（百炼控制台创建，`sk-` 开头）。
- 模型：`CHAT_MODEL=qwen3.7-plus`、`EMBED_MODEL=text-embedding-v4`、`RERANK_MODEL=qwen3-rerank`（重排走 `compatible-api`，可用 `RERANK_BASE_URL` 覆盖）。
- 图片向量：阿里云多模态 Embedding 不支持 OpenAI 兼容，代码自动降级，不影响其他功能。

## 常见问题

- **Chroma 数据卷权限**：容器以 root 运行，数据卷 `rag_data` 挂载到 `/data`；如需改用用户，自行调整 Dockerfile。
- **身份模型**：`X-API-Key` 现在是 `service` 角色，只能调用业务接口；账号管理需要
  `ADMIN_API_KEY` 或管理员会话。前端默认走 `/api/v1/auth/login` 登录。
- **数据持久化**：账号与审计写入 `RUNTIME_DIR`（容器内 `/data/runtime_data`），
  容器重建后仍然保留；文档与向量数据在 `rag_data` 卷。
- **模型不通**：检查根 `.env` 的 `LLM_BASE_URL` 与 `LLM_API_KEY`；网关与 RAG 服务共用这一份统一配置，旧变量 `ONEAPI_*`/`OPENAI_*` 仅作兼容。
- **本地开发（不用 Docker）**：按根 README 的步骤用 venv 分别启动 rag、api、web。
