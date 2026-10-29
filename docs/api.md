# API 文档

网关地址 `http://localhost:8000`。除 `/health`、`/docs`、`/redoc` 外，所有请求需带请求头 `X-API-Key: <API_KEY>`。

## 对话（SSE 流式）

```
POST /api/v1/chat
Content-Type: application/json
X-API-Key: <key>
```

请求体：

```json
{"session_id": null, "message": "VPN 无法连接，报修", "agent_hint": null}
```

响应为 SSE 事件流，事件类型：

| type | 说明 |
|---|---|
| `agent_switch` | 已路由到某 Agent，携带 `agent` |
| `token` | 回答分片，携带 `content` |
| `done` | 完成，携带 `answer`/`agent`/`attempts` |
| `error` | 出错，携带 `message` |
| `session` | 末尾回传 `session_id`（新建会话时使用） |

## 其他接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 网关与 RAG 服务健康状态 |
| GET | `/api/v1/sessions?user_id=default` | 会话列表 |
| GET | `/api/v1/agents` | 已注册 Agent 清单 |
| POST | `/api/v1/feedback` | 反馈（up/down/retry） |
| POST | `/api/v1/feedback/undo` | 撤销好评入库 |
| POST | `/api/v1/ingest` | 上传文档到知识库（multipart `file`） |
| GET | `/api/v1/ingest/status/{id}` | 摄入进度 |
| GET | `/api/v1/documents` | 已索引文档 |
| DELETE | `/api/v1/documents?ids=...` | 删除文档 |

## SSE 事件示例

```
data: {"type":"agent_switch","agent":"it_support"}

data: {"type":"token","content":"您好，请提供设备类型、操作系统和错误信息…"}

data: {"type":"done","answer":"…","agent":"it_support","attempts":1}

data: {"type":"session","session_id":"abc123"}
```
