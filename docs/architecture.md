# 架构说明

## 分层

| 层 | 职责 | 位置 |
|---|---|---|
| 接入层 | 统一 API、SSE 流式、鉴权、前端 | `apps/api`、`apps/web` |
| 编排层 | 意图路由、任务分配、状态管理、反思校验 | `packages/orchestrator` |
| Agent 层 | 领域专家能力单元（Manifest + 工具 + Prompt） | `packages/agents/*` |
| 知识层 | 多模态解析、混合检索、重排、评测 | `packages/rag` |
| 支撑层 | 记忆、工具注册、MCP、插件沙箱 | `packages/memory`、`packages/tools` |
| 治理层 | 审计日志、反馈闭环、可观测性 | core + api |

## 请求主链路

```
用户 → 前端(SSE) → 网关 /api/v1/chat
  → 语义路由（关键词 + LLM 兜底）
  → Supervisor（LangGraph，从注册表选 Agent）
  → 专家 Agent（ReAct 循环，经 MCP 工具层调用 RAG / Jira / 采集等）
  → 反思校验（可选，最多重做 2 轮）
  → 流式返回 + 会话持久化 + 审计日志
```

## 端口约定

- api: `8000`，rag: `8010`，web: `3000`，postgres: `5432`，redis: `6379`
