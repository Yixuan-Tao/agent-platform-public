"""统一 pydantic schema：聊天、反馈、Agent Manifest、会话。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(min_length=1)
    agent_hint: Optional[str] = None
    # 兼容字段：服务端以认证身份为准，忽略客户端传入值（防止越权伪造归属）。
    user_id: str = Field(default="", description="已废弃，服务端忽略")


class ChatResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]
    agent: Optional[str] = None


class Feedback(BaseModel):
    run_id: Optional[str] = None
    session_id: str
    message_id: str
    feedback_type: Literal["up", "down", "retry"]
    reason: Optional[str] = None
    correction: Optional[str] = None


class AgentManifest(BaseModel):
    """Agent 能力描述，注册进 AgentRegistry 并生成路由表。"""

    name: str
    description: str
    capabilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    model_override: Optional[str] = None
    enabled: bool = True
    # 协作契约（Router-as-ReAct）：本 Agent 允许调用的子 Agent 名单（空 = 不暴露给路由层）
    handoff: list[str] = Field(default_factory=list)
    input_hint: str = ""
    output_hint: str = ""


class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageInfo(BaseModel):
    """持久化消息（用于会话历史回放与反馈溯源）。"""

    message_id: str
    session_id: str
    role: Role
    content: str
    agent: str = ""
    created_at: datetime
