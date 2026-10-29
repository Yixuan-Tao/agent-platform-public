# Agent Platform

一个用于学习和开发的多 Agent / RAG 平台，包含 FastAPI 网关、Next.js 界面、工具治理、持久化任务和独立 RAG 服务。

## 功能与目录

| 目录 | 内容 |
| --- | --- |
| `apps/api` | 认证、会话、SSE、文档代理、任务和实验接口 |
| `apps/web` | 对话、任务和实验界面 |
| `packages/core` | 配置、数据库、执行契约和模型调用治理 |
| `packages/orchestrator` | 路由、ReAct、规划执行、预算、恢复与审批 |
| `packages/tools` | 工具契约、权限、MCP、RAG/Jira 适配和执行隔离 |
| `packages/agents` | IT 支持、硬件问答、文档对比和信息采集示例 |
| `packages/rag` | 文档解析、检索、重排、引用与证据管理 |
| `packages/memory` / `packages/evaluation` | 记忆隔离与合成任务评测 |
| `skills` / `migrations` / `deploy` | 技能模板、数据库迁移和部署配置 |

## 公开版内容

仓库从当前源码建立全新的历史，包含通用实现和软件测试。演示文档族、参数模板和平台评测任务均为合成数据。仓库不提供知识库语料；使用 RAG 前，需要自行摄入有权使用的文档。

原始文件、模型回答、历史成绩、专用别名表、账号、密钥、向量库、数据库、日志和本机配置均不随仓库分发。离线测试验证软件行为；真实模型质量、外部工单服务和生产部署需在自己的环境中另行验证。

## 开始使用

需要 Python 3.11+、Node.js 20 和 pnpm 9。以下命令分别在仓库根目录运行。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r deploy/requirements.txt
Copy-Item .env.example .env
```

在 `.env` 填入自己的模型配置、管理员凭据和服务令牌。示例中的开发密钥与数据库账号仅用于本地开发，应在部署时替换。详见 [配置说明](docs/configuration.md)。

RAG 服务在单独终端运行，并需安装自己的依赖：

```powershell
pip install -r packages/rag/requirements.txt
Set-Location packages/rag
python main.py
```

网关在仓库根目录的另一终端运行：

```powershell
$env:PYTHONPATH = 'packages/core/src;packages/tools/src;packages/memory/src;packages/orchestrator/src;packages/agents/src;packages/skills/src;packages/evaluation/src;apps/api'
python -m uvicorn app.main:app --port 8000
```

前端：

```powershell
Set-Location apps/web
pnpm install --frozen-lockfile
pnpm dev
```

也可在完成 `.env` 配置后使用 `docker compose --env-file .env -f deploy/docker-compose.yml up -d --build`。部署说明见 [deploy/README.md](deploy/README.md)。

## 验证

```powershell
python scripts/check_public_content.py
python -m pytest packages/core/tests packages/orchestrator/tests packages/tools/tests packages/memory/tests packages/evaluation/tests apps/api/tests -q
```

RAG 单元测试在 `packages/rag` 运行 `python -m pytest tests/unit -q`。前端在 `apps/web` 运行 `pnpm typecheck` 和 `pnpm build`。PostgreSQL 专项集成测试需要单独的临时数据库，CI 配置见 `.github/workflows/platform-regression.yml`。

更多说明：[架构](docs/architecture.md)、[API](docs/api.md)、[评测](packages/rag/evaluation/README.md)、[公开内容规则](PUBLIC_CONTENT.md)。
