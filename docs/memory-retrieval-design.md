# 长期记忆检索接入方案（安全防膨胀）

## 实现状态（2026-08-04）

✅ 已落地：`LongTermMemory.search_filtered()`（阈值 + 条数 + token 三重控制）、
`_estimate_tokens()`、配置项 `MEMORY_FEW_SHOT_*`、编排层 `route_and_run()` few-shot 注入
（检索失败/超限降级为空），并补单元测试与编排层降级测试。

## 1. 背景与现状

### 1.1 现状问题

记忆分两层（见 `packages/memory`）：

- **短期记忆** `ShortTermMemory`：会话级历史，组装上下文时固定取最近 10 条（`recent(k=10)`），有硬上限，不会膨胀。
- **长期记忆** `LongTermMemory`：把"被点赞的问答对"持久化到 ChromaDB，文档上写明用于后续会话的 **few-shot**。

但通过代码检索确认，**`LongTermMemory.search()` 在生产链路中从未被调用**：

- 唯一调用点在 `packages/memory/tests/test_memory.py`（测试）。
- 编排层 `OrchestratorService.route_and_run()` 只注入了短期记忆，**没有读取长期记忆**。

因此当前长期记忆是"**只写不读**"的空转状态：反馈被写入，却从未被塞回上下文，设计文档（`docs/architecture.md`、RAG 评测集）声称的"few-shot 复用"能力实际未生效。

### 1.2 接入的两类风险

接入长期记忆检索后，随着反馈量增长会出现两类问题：

1. **上下文膨胀**：每次请求都注入多条 QA，不做任何裁剪，反馈越多上下文越长，最终撑爆模型窗口。
2. **污染与误导**：检索只按 score 排序、无相关性阈值，不相关的旧反馈也会被注入，反而干扰模型回答（负面 few-shot）。

本文档的目标是**在不引入膨胀和污染的前提下，把长期记忆安全地接进上下文**。

## 2. 目标与非目标

### 2.1 目标

- 在 `route_and_run()` 中将长期记忆检索结果作为 few-shot 注入上下文。
- 注入量有**硬性上限**（条数 + token 数），与反馈总量无关。
- 只有**相关性达标**的反馈才会被注入，防止污染。
- 全程可降级、可回滚，不影响现有对话链路。

### 2.2 非目标

- 不做跨会话的身份/偏好记忆（那是用户画像，超出范围）。
- 不改造短期记忆的滑动窗口策略（当前固定 10 条已够用）。
- 不引入新的向量数据库（沿用现有 ChromaDB + 本地哈希嵌入）。

## 3. 设计总览

接入点在 `OrchestratorService.route_and_run()`，在组装短期记忆之后、构建 Agent 图之前，插入长期记忆检索：

```
组装短期记忆（最近 10 条）
        │
        ▼
长期记忆检索（search + 相关性阈值 + token 裁剪）   ← 新增
        │  仅注入 score 达标且累计 token 未超限的 QA
        ▼
构建 Agent 图 → 反思循环 → 流式返回
```

关键设计原则：**检索是"可选增强"，不是"必要输入"**。即使检索为空、失败、超限，也必须优雅跳过，不影响主流程。

## 4. 详细设计

### 4.1 检索入口：给 `LongTermMemory` 增加带阈值的方法

现有 `search()` 只按 score 排序，无阈值过滤。为保持兼容，新增一个带过滤的查询方法：

```python
def search_filtered(
    self,
    query: str,
    *,
    k: int = 3,
    min_score: float = 0.5,
    max_tokens: int = 500,
) -> list[dict]:
    """检索相似 QA，串行应用：相关性阈值过滤 → 条数上限 → token 上限。

    返回的列表必然满足：
      - 条数 <= k
      - 得分 >= min_score（低于阈值直接丢弃）
      - 累计 token 数 <= max_tokens（超出即截断）
    """
    results = self.search(query, k=k)
    # 1) 相关性阈值过滤
    passed = [r for r in results if r["score"] >= min_score]
    # 2) token 累计裁剪（按得分降序，score 越高越优先）
    out: list[dict] = []
    used = 0
    for r in passed:
        approx_tokens = _estimate_tokens(r["question"] + r["answer"])
        if used + approx_tokens > max_tokens:
            break
        out.append(r)
        used += approx_tokens
    return out
```

### 4.2 token 估算（无外部依赖）

由于长期记忆完全离线、不引 LLM，用轻量近似即可，不必精确：

```python
def _estimate_tokens(text: str) -> int:
    """中文按字符数近似，英文按 4 字符 1 token 近似，取保守上界。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + other // 4 + 1
```

### 4.3 编排层接入：安全缝合

在 `route_and_run()` 中，短期记忆之后、`registry.build()` 之前插入：

```python
# 长期记忆 few-shot（可选增强）：检索相似反馈，注入为 system 提示
few_shot = ""
if self.memory_service is not None:
    try:
        hits = self.memory_service.long_term.search_filtered(
            user_message, k=3, min_score=0.5, max_tokens=500
        )
        if hits:
            few_shot = "参考历史优质问答（可借鉴其表述与结构）：\n" + "\n".join(
                f"- Q: {h['question']}\n  A: {h['answer']}" for h in hits
            )
    except Exception:
        few_shot = ""  # 检索失败一律降级为空，不影响主流程
```

在 `generate()` 组装消息时，把 `few_shot` 并入 system 提示：

```python
async def generate(feedback: str) -> str:
    msgs = list(messages)
    system_parts = []
    if few_shot:
        system_parts.append(few_shot)
    if feedback:
        system_parts.append(f"质检反馈，请据此改进回答：{feedback}")
    if system_parts:
        msgs = msgs + [{"role": "system", "content": "\n".join(system_parts)}]
    ...
```

### 4.4 参数与开关（可配置、可回滚）

为保证"安全接入"，把本次新增行为全部收敛到配置，默认开启但支持一键关闭：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `memory.few_shot.enabled` | `true` | 总开关，false 时完全不检索不注入 |
| `memory.few_shot.k` | `3` | 最多注入条数 |
| `memory.few_shot.min_score` | `0.5` | 相关性阈值，低于则丢弃 |
| `memory.few_shot.max_tokens` | `500` | 注入上下文 token 上限 |

> 回滚手段：环境变量把 `enabled` 置为 `false` 即可，无需改代码、无需发版。

## 5. 防膨胀的量化保证

本次设计保证**上下文注入量与反馈总量无关**，恒满足：

```
注入量 <= k 条（默认 3）
注入量 <= min_score 命中数（默认 0.5 阈值）
注入量 <= max_tokens token（默认 500）
```

- 短期记忆：固定 10 条，不受反馈量影响。
- 长期记忆：固定 ≤ 3 条 / ≤ 500 token，且只注入得分 ≥ 0.5 的。

因此无论反馈累积到多少，单次请求注入的上下文长度都有**常数上界**，不会随反馈量线性增长。

## 6. 风险与降级策略

| 场景 | 处理 |
|---|---|
| ChromaDB 不可用 | `search_filtered` 抛异常 → 捕获后 `few_shot=""`，走纯短期记忆 |
| 检索结果全不相关 | 阈值过滤后为空 → 不注入，等价于现状 |
| 结果过多 | 条数 + token 双重截断，只取头部 |
| 单条反馈质量差 | 由 score 阈值 + 人工可撤销（`undo`）兜底 |
| 上线后效果不佳 | 开关 `enabled=false` 一键回滚 |

## 7. 测试与验证

### 7.1 单元测试（`packages/memory/tests/test_memory.py` 扩展）

1. **阈值过滤**：低于 `min_score` 的结果不返回。
2. **条数上限**：`k` 生效，返回不超过 `k` 条。
3. **token 上限**：构造长问答，验证累计 token 超限即截断。
4. **降级**：`search_filtered` 抛异常时，编排层 `few_shot` 为空，主流程不中断。

### 7.2 集成验证（手工/脚本）

- 构造一条被点赞的问答 → 触发 `record(up)` → 等延迟入库 → 发相似问题，确认响应中出现该 QA 的表述风格。
- 验证极端场景：反馈库中注入大量不相关/超长 QA，确认单次请求上下文长度仍在预算内。

## 8. 实施步骤

1. 在 `long_term.py` 增加 `_estimate_tokens()` 与 `search_filtered()`。
2. 在 `service.py` 的 `build_memory_service()` 中读取并暴露 few-shot 配置开关。
3. 在 `orchestrator/service.py` 的 `route_and_run()` 中接入检索与注入。
4. 补充单元测试（见第 7 节）。
5. 联调验证 + 上线，默认开启，保留 `enabled=false` 回滚开关。

## 9. 结论

- 本方案通过**阈值过滤 + 条数上限 + token 上限**三重控制，确保长期记忆接入后上下文注入量有**常数上界**，反馈再多也不会膨胀。
- 通过**失败降级为空 + 一键开关**，保证接入是安全、可回滚的，不影响现有对话链路。
- 补齐了"长期记忆只写不读"的功能缺口，让反馈沉淀真正复用于后续会话。
