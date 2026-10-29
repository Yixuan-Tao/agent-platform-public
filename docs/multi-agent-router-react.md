# 实施方案：Router-as-ReAct 多 Agent 编排（含 Plan-and-Execute）

## 实现状态（2026-08-04）

✅ 已落地：`AgentManifest.handoff/input_hint/output_hint`；`AgentTool`（contextvars 深度/循环
控制）；`router_react.py` Router ReAct 图；`planner_execute.py`（ComplexityJudge + PLAN/EXECUTE/
REPLAN）；`route_and_run_react()` 双模式入口（快速路径 + 复杂度分派）；`chat.py` 配置切换；
4 个 Agent 已配置 handoff；编排测试（快速路径/跨领域/深度/循环/并发/Plan 全链路）。

⏳ 待办：`USE_ROUTER_REACT=true` 上线后按测试集验证跨领域问答效果。

## 1. Summary（目标）

将现有"路由选一个 Agent"的编排，升级为**由主 Agent 以工具方式自主调度多个子 Agent**。主 Agent 支持**两种执行模式**，按问题复杂度自动选择：

- **ReAct 模式**：用于简单/跨领域问题，LLM 自主思考→调用子 Agent→再思考→汇总。
- **Plan-and-Execute 模式**：用于**长/复杂问题**，先由 Planner 生成完整步骤计划，再逐步执行、每步可调用子 Agent，并由 Replanner 在步骤间动态调整计划。

保留现有单 Agent 路由作为快速路径，新增执行模式与现有 `route_and_run` 共存，通过配置开关切换。

## 2. Current State Analysis（现状）

- 编排主流程：[service.py](../packages/orchestrator/src/agent_platform/orchestrator/service.py) 的 `route_and_run` 通过 `pick_agent` 选出一个 Agent（`agent_hint` → 关键词语义路由 → LLM 兜底），然后只执行该一个 Agent 的图，再经 `ReflectionLoop` 反思。

- 语义路由：[router.py](../packages/orchestrator/src/agent_platform/orchestrator/router.py) `SemanticRouter.route()` 按 `INTENT_KEYWORDS` 命中数打分；`ensure_llm_route` 用 LLM 兜底。

- 注册表：[registry.py](../packages/orchestrator/src/agent_platform/orchestrator/registry.py) `AgentRegistry` 管理 Manifest + 构建器，`build()` 懒构建并缓存 Agent 图。

- Agent 构建：[agents/common.py](../packages/agents/src/agent_platform/agents/common.py) `build_react_agent` 用 `ChatOpenAI(model=llm_router.resolve_model(name), **llm_router.settings.llm_kwargs())` 构造模型，`create_react_agent` 构建 ReAct 图。

- 图调用：[supervisor.py](../packages/orchestrator/src/agent_platform/orchestrator/supervisor.py) `_invoke_agent` 兼容 dict/字符串返回，提取 `final_answer`。

- 工具契约：[tools/base.py](../packages/tools/src/agent_platform/tools/base.py) `BaseTool` + `ToolRegistry`。

- 数据模型：[core/models.py](../packages/core/src/agent_platform/core/models.py) `AgentManifest`（含 name/description/capabilities/tools/model_override/enabled）。

- 状态：[orchestrator/state.py](../packages/orchestrator/src/agent_platform/orchestrator/state.py) `AgentState`。

- 依赖事实：`tools` 包只依赖 httpx/bs4/feedparser/pydantic，**不依赖** orchestrator；`orchestrator` 依赖 langgraph/langchain-openai/langchain-core/openai/pydantic，**不依赖** tools。`AgentTool` 需依赖 `registry` 与 `_invoke_agent`（均在 orchestrator），故 `AgentTool` 应放 **orchestrator 包**，并让 orchestrator 新增对 `agent-platform-tools` 的依赖以复用 `BaseTool`，避免循环依赖。

- 测试约定：`tests/test_service.py` 用 `FakeLLM`/`FakeGraph`/`_agent_builder` 注入 fake 依赖，`pytest` 异步用例。

## 3. Proposed Changes（改动清单）

### 3.1 数据模型：`AgentManifest` 增加协作契约

**文件**：`packages/core/src/agent_platform/core/models.py`

- 新增字段：

  - `handoff: list[str] = []`：允许本 Agent 调用的子 Agent 名单（空 = 不暴露给路由层）。

  - `input_hint: str = ""`：调用本 Agent 需传什么（给 LLM 的说明）。

  - `output_hint: str = ""`：本 Agent 返回什么（给 LLM 的说明）。

### 3.2 新增 `AgentTool`（放 orchestrator 包）

**文件（新建）**：`packages/orchestrator/src/agent_platform/orchestrator/agent_tool.py`

- 继承 `agent_platform.tools.base.BaseTool`。

- 构造参数：`agent_name`、`registry`、`llm_router`、`tool_registry`、`max_depth`。

- `name = f"agent__{agent_name}"`（唯一，避免与普通工具冲突）。

- `execute(task)`：内部 `registry.build(...)` 构建目标图 → `_invoke_agent` 运行 → 返回 answer。

- 深度/循环控制用 `contextvars.ContextVar`（按请求隔离，避免并发串扰）：

  - 每次进入 `depth += 1`；`depth > max_depth` 返回"已达深度上限，请基于现有信息回答"。

  - 维护调用链集合；目标 Agent 已在链上则返回"禁止循环调用"。

  - `finally` 中恢复 depth / 弹出调用链。

### 3.3 新增 Router 图

**文件（新建）**：`packages/orchestrator/src/agent_platform/orchestrator/router_react.py`

- `ROUTER_PROMPT`：约束 Router 自主判断是否拆分子任务、可调用多个 Agent、必须整合结果输出最终回答。

- `build_router_graph(registry, llm_router, max_agent_calls=4)`：

  - 取 `registry.list_manifests()` 中 `handoff` 非空的 Agent，包装成 `AgentTool`。

  - `ChatOpenAI(model=llm_router.resolve_model("router"), **llm_router.settings.llm_kwargs())`，`temperature=0` 保证调度稳定。

  - `create_react_agent(model, agent_tools, prompt=ROUTER_PROMPT)`。

- 提供 `get_router_graph()` 懒构建并缓存（模块级单例，仿 `get_agent_registry`）。

### 3.4 主 Agent 编排器接入（双模式入口）

**文件**：`packages/orchestrator/src/agent_platform/orchestrator/service.py`

- `__init__` 新增参数：`use_router_react: bool = False`、`max_agent_calls: int = 4`、`plan_mode: bool = True`（是否启用 Plan-and-Execute）、`max_plan_steps: int = 6`。

- 新增 `route_and_run_react(session_id, user_message, agent_hint=None)` 异步生成器：

  1. **快速路径**：`agent_hint` 合法，或 `SemanticRouter.route` 置信度 ≥0.9 时直接命中单个 Agent，走原单 Agent 执行逻辑（`ReflectionLoop`）。
  2. **复杂度判定**：一般路径下调用 `ComplexityJudge`（见 3.8）判断 `simple` / `complex`。
     - `simple` → 产出 `router_start` 事件，跑 **ReAct 图**（3.3）。
     - `complex` → 产出 `plan_start` 事件，跑 **Plan-and-Execute 图**（3.8）。
  3. 按 `chunk_size` 切块产出 `token` 事件，最后 `done`（answer / agent="router" / attempts）。

- 保留原 `route_and_run` 不变。

### 3.5 事件流与 API 接入

**文件**：`apps/api/app/routes/chat.py`

- 按 `orchestrator.use_router_react` 选择调用 `route_and_run_react` 或 `route_and_run`。

- 新增 `router_start`、`agent_call` 事件类型（可选，用于前端展示调用链）。

### 3.6 依赖调整

**文件**：`packages/orchestrator/pyproject.toml`

- `dependencies` 增加 `"agent-platform-tools"`（本地包，供 `AgentTool` import `BaseTool`）。

### 3.7 测试

**文件（新建）**：`packages/orchestrator/tests/test_router_react.py`

- 复用 `FakeLLM`/`FakeGraph`/`_agent_builder` 风格。

- 用例：

  1. 单领域命中关键词 → 走快速路径，不触发 Router ReAct。
  2. 跨领域 → Router 图被调用并返回结果。
  3. 深度超限：构造 `AgentTool(max_depth=1)` 嵌套调用，断言不无限递归。
  4. 循环调用：A→A 返回"禁止循环"。
  5. 并发安全：`asyncio.gather` 同时多请求，断言 depth/调用链互不串扰。

### 3.8 Plan-and-Execute 子图（处理长/复杂问题）

**文件（新建）**：`packages/orchestrator/src/agent_platform/orchestrator/planner_execute.py`

主 Agent 的 **Plan-and-Execute 模式**用 LangGraph `StateGraph` 构建，节点结构：

```
PLAN → EXECUTE(第1步) → REPLAN → EXECUTE(第2步) → ... → END
```

- **PLAN**：LLM 把用户长问题拆成有序步骤 `steps: list[str]`（≤ `max_plan_steps`）。
- **EXECUTE**：执行当前步骤 `current_step`，工具集 = `AgentTool` 列表 + 共享工具；结果追加到 `step_results`。
- **REPLAN**：检查 `current_step` 是否已到最后；满足则 `END`，否则 `Command(goto="execute")` 继续下一步；若某步结果异常，可改写剩余步骤（动态调整计划）。

**关键实现细节**：
- 复用 `AgentTool`（3.2）作为 EXECUTE 的工具，子 Agent 仍以工具方式被调用，与 ReAct 模式共享同一套工具。
- 状态复用 `AgentState`，并新增字段（见下 3.9）。
- Planner / Replanner 用 `llm_router.chat`（`temperature=0`），结构化输出步骤列表。
- 提供 `build_plan_execute_graph(registry, llm_router, tool_registry, max_plan_steps)` 与模块级懒加载缓存 `get_plan_execute_graph()`。

**复杂度判定 `ComplexityJudge`**（同文件内）：
- 规则：启发式——问题长度超阈值、含多个领域关键词、或含"对比/分析/方案/报告"等复杂指令词 → `complex`；否则 `simple`。
- 可选扩展：用 LLM 判定（`temperature=0`，输出 `simple`/`complex`），默认用规则，避免额外调用成本。

### 3.9 状态扩展：`AgentState`

**文件**：`packages/orchestrator/src/agent_platform/orchestrator/state.py`

- 为 Plan-and-Execute 新增字段：
  - `steps: list[str]`：当前计划步骤。
  - `current_step: int`：当前执行到第几步。
  - `step_results: list[str]`：每步执行结果。
  - `mode: str`（`"react"` / `"plan_execute"`）：当前执行模式。

## 4. Assumptions & Decisions（假设与决策）

- **决策**：`AgentTool` 放 orchestrator 包而非 tools 包，避免 tools→orchestrator 循环依赖；orchestrator 新增对 tools 的依赖。

- **决策**：深度/循环控制用 `contextvars` 按请求隔离，保证并发安全。

- **决策**：Router 决策 `temperature=0`，且保留 `SemanticRouter` 作为快速路径，兼顾稳定性与成本。

- **决策**：主 Agent 支持 ReAct + Plan-and-Execute 双模式，由 `ComplexityJudge`（默认规则启发式）判定，避免简单问题误走高成本的 Plan 流程。

- **决策**：Plan-and-Execute 的 EXECUTE 复用 `AgentTool`（3.2），与 ReAct 共享同一套子 Agent 工具，不重复实现。

- **假设**：`router` 无独立 Manifest，`resolve_model("router")` 走默认模型（config 中无专门覆盖时返回默认）。

- **假设**：`handoff` 默认空，逐步放开；先小范围验证（如 `it_support` 可选 `rag` 相关 Agent）。

- **兼容**：`mode="router_react"` 默认关闭，不影响现有单 Agent 路由；`plan_mode` 默认开启，仅在 `use_router_react=true` 时生效。

## 5. Verification（验证）

1. `cd agent-platform`，对 `packages/orchestrator` 运行 `pytest tests/`，确认新增用例全绿、旧用例不回归。
2. 启动 API（`python run.py`），用 `curl` 以 `use_router_react=true` 发跨领域请求，检查 SSE 事件流含 `router_start` 与 `done`。
3. 单领域请求验证走快速路径（不产生 `router_start`）。
4. 发长/复杂问题，验证走 `plan_start` → Plan-and-Execute 流程，SSE 含步骤进度。
5. 手工构造深度/循环用例，确认不无限递归、不循环调用。

## 6. 实施顺序（分步）

1. `AgentManifest` 加字段（3.1）。
2. `AgentTool` 实现 + contextvars 控制（3.2）。
3. `router_react.py`：ROUTER_PROMPT + `build_router_graph` + 缓存（3.3）。
4. `AgentState` 扩展 Plan-and-Execute 字段（3.9）。
5. `planner_execute.py`：PLAN/EXECUTE/REPLAN 节点 + `ComplexityJudge` + `build_plan_execute_graph` + 缓存（3.8）。
6. `service.py` 接入 `route_and_run_react` 双模式分派 + 配置（3.4）。
7. `chat.py` 事件流接入（3.5）。
8. orchestrator `pyproject.toml` 加依赖（3.6）。
9. 编写并运行测试（3.7，补 Plan-and-Execute 场景：PLAN 生成步骤、逐步执行、REPLAN 调整、步骤数超限）。
