# 工具调用模块评审与优化方案

> 本文档记录对 agent-platform 工具调用（Tool Calling）模块的代码评审结论，基于实际代码分析 + 业界最佳实践调研，整理出当前存在的问题与可落地的优化方向。

## 实现状态（2026-08-04）

✅ 已落地：`ToolGateway` 运行时网关（schema 校验 + ACL + 超时/重试/熔断 + 结构化错误 +
`execute_many` 并行）；`BaseTool.required_role`；Agent 工具调用接入网关；MCP 服务端走网关并新增
`tools/call_many`；`mcp/client.py` 客户端（stdio / Streamable HTTP，原生实现）；`PluginTool`
沙箱插件接入；`ToolRegistry.register_from_config` 动态注册；`main.py` 合并 MCP/插件/声明式工具。

⏳ 待办：真实外部 MCP 服务器联调（当前单测用假服务器验证协议）。

## 一、背景与范围

本对话围绕 agent-platform 的四大要素（规划 / 记忆 / 工具 / 行动）展开，重点深入分析了 **工具（Tools）模块** 的实现与可扩展性，并结合主流 LLM 工具调用最佳实践（OpenAI / Anthropic / Gemini / MCP / LangGraph）进行了对比。

涉及核心文件：

| 文件 | 职责 |
|------|------|
| `packages/tools/src/agent_platform/tools/base.py` | `BaseTool` 契约 + `ToolRegistry` 注册表 |
| `packages/tools/src/agent_platform/tools/mcp/server.py` | MCP 服务端（暴露工具给外部） |
| `packages/tools/src/agent_platform/tools/sandbox.py` | 插件沙箱（子进程隔离） |
| `packages/agents/src/agent_platform/agents/common.py` | `build_react_agent` 工具挂载 + ReAct 循环 |
| `apps/api/app/main.py` | 工具注册的接入点 |

## 二、已具备的能力（值得保留的基线）

1. **统一契约**：所有工具继承 `BaseTool`，统一 `name / description / args_schema / execute()`。
2. **注册表模式**：`ToolRegistry` 线程安全、按名注册/查找，集中管理工具。
3. **"模型只请求、运行时执行"**：LLM 只产出结构化调用请求，真正执行由后端 `execute()` 完成，符合安全边界。
4. **MCP 服务端**：能把 `ToolRegistry` 包装为 JSON-RPC 服务暴露给外部（`tools/list` / `tools/call`）。
5. **降级容错**：langgraph 缺失时降级为 `SimpleAgent`；外部依赖失败时工具返回可读错误信息。

## 三、发现的问题

### 1. 不支持外部 Skills 接入
- 整个 `tools` 包没有任何 skills 的发现 / 加载 / 接入机制。
- 外部能力只能通过"手写 `BaseTool` 子类 + 手动注册"接入，无声明式 / 动态发现。

### 2. MCP 只有服务端，没有客户端
- `McpToolServer` 是**反向**暴露——把本平台工具提供给外部。
- 全项目**没有 MCP 客户端实现**（无 `ClientSession` / `StreamableHttp` / `fastmcp`），`pyproject.toml` 也未引入 `mcp` SDK。
- 因此**无法作为客户端调用外部第三方 MCP 服务器**。

### 3. 插件沙箱存在但未接入调用链
- [sandbox.py](../packages/tools/src/agent_platform/tools/sandbox.py) 的 `SandboxedPlugin` 用子进程 + JSON IPC 隔离运行，但**未继承 `BaseTool`**，也**未被 `main.py` 使用**，无法被 Agent 调用，是孤立代码。

### 4. 注册表为静态硬编码注册，不支持动态发现
- 工具在 `main.py` 的 `lifespan` 中**显式实例化并注册**，无热加载 / 动态发现 / 配置注入。
- 注册表是进程内单例对象，不持久化。

### 5. 与最佳实践对比的差距

| 最佳实践 | 当前状态 | 差距 |
|---------|---------|------|
| LLM 只请求、运行时执行 | ✅ 已具备 | — |
| 调用前 Schema 严格校验 | ⚠️ 有 `args_schema` 但未在调用前统一校验 | 缺 `additionalProperties:false`、enum/format 等约束执行 |
| ACL 权限校验 | ❌ 无 | 无法按权限过滤工具 |
| 结构化错误返回 | ⚠️ 部分工具返回 `ok:false` | 未统一错误码 / retryable |
| 超时 / 重试 / 熔断 | ⚠️ 仅 LLM 层有 | 工具层缺失 |
| 并行工具调用 | ❌ 无 | 不支持一次多工具并行 |
| MCP 客户端接入 | ❌ 无 | 无法接入外部工具 |
| 危险操作二次确认 | ❌ 无 | 不可逆操作无确认机制 |

## 四、优化方向

### 方向一：工具调用运行时网关（校验 + 授权 + 结构化错误）
在 `ToolRegistry` 与 `execute()` 之间增加一层网关，统一做：

1. **Schema 校验**：调用前用 pydantic 校验参数（类型、必填、enum、format），失败返回结构化错误让模型修正。
2. **ACL 权限校验**：根据调用方（用户/Agent）权限动态过滤可用工具，白名单校验。
3. **结构化错误**：统一返回 `{ok, error_code, message, retryable}`，避免把原始异常丢给模型。
4. **超时 / 重试 / 熔断**：工具层复用 LLM 层的 `CircuitBreaker` 与 tenacity 重试策略。

### 方向二：集成 MCP 客户端（接入外部工具）
引入官方 `mcp` SDK 或 `langchain-mcp-adapters`：
- 支持 stdio / Streamable HTTP / SSE 三种传输。
- 使用 `MultiServerMCPClient` 把外部 MCP 服务器拉取为 LangChain 工具，再包装进现有 `ToolRegistry`。
- 这样外部第三方工具即可像本地工具一样被 Agent 调用，实现"工具热插拔 + 跨模型标准化"。

### 方向三：接入插件沙箱到调用链
让 `SandboxedPlugin` 适配 `BaseTool` 接口，注册进 `ToolRegistry`，使 Agent 能调用隔离环境的用户插件。

### 方向四：注册表动态化
支持从配置 / 目录 / 远程注册中心动态加载工具，并暴露 `tools/list` 供 LLM 发现，替代硬编码注册。

### 方向五：支持并行工具调用
代理层支持一次返回多个 `tool_calls`，对相互独立的工具并发执行，降低多工具任务的延迟。

## 五、优先级建议

| 优先级 | 优化项 | 理由 |
|--------|--------|------|
| P0 | 运行时网关（校验 + 结构化错误 + 超时） | 直接提升可靠性，是工具调用的地基 |
| P0 | ACL 权限校验 | 安全底线，防止模型越权 |
| P1 | MCP 客户端接入 | 解锁外部工具生态，标准化方向 |
| P1 | 并行工具调用 | 明显降低延迟 |
| P2 | 插件沙箱接入 / 注册表动态化 | 提升可扩展性，非阻塞 |

## 六、结论

项目已具备工具调用的**正确骨架**（契约 + 注册表 + "模型请求/运行执行"分离 + ReAct 循环），当前主要短板集中在**执行前的校验/授权网关、结构化错误、并行调用、MCP 客户端接入**四块。优先补齐 P0 两项即可显著提升工具模块的可靠性与安全性。
