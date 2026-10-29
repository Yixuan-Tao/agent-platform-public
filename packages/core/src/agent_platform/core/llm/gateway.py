"""统一模型网关校验与 gateway_unavailable 失败语义。

对齐 intern_rag 新版 `core/model_gateway.py` 的能力，面向平台 core 配置：
    - 启动自检：HTTPS 强制、可信主机白名单（配置非空时）、API Key 非空
    - 健康检查：``gateway_status()`` 供 /health 上报
    - 失败语义：``GatewayUnavailableError`` 供调用方判定降级

开发模式（未配置 MODEL_GATEWAY_ALLOWED_HOSTS）跳过主机白名单校验，只做基础检查。
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from agent_platform.core.config import Settings


class GatewayUnavailableError(RuntimeError):
    """模型网关不可用（配置非法 / 熔断 / 连接失败）时抛出的统一异常。"""


@dataclass
class GatewayStatus:
    ok: bool
    reason: str = ""


def validate_gateway(settings: Settings) -> GatewayStatus:
    """校验模型网关配置；返回 ok=False 时不阻塞启动，仅记录降级状态。"""
    base_url = (settings.llm_base_url or "").rstrip("/")
    if not base_url:
        return GatewayStatus(False, "未配置 LLM_BASE_URL")
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.hostname:
        return GatewayStatus(False, "模型网关必须使用 HTTPS URL")
    allowed = [host.strip().lower() for host in settings.model_gateway_allowed_hosts.split(",") if host.strip()]
    if allowed and parsed.hostname.lower() not in allowed:
        return GatewayStatus(False, f"模型网关主机 {parsed.hostname} 不在允许列表")
    if not settings.llm_api_key:
        return GatewayStatus(False, "缺少模型网关 API Key")
    return GatewayStatus(True, "ok")


def gateway_status(settings: Settings) -> dict:
    status = validate_gateway(settings)
    return {
        "ok": status.ok,
        "base_url": settings.llm_base_url,
        "model": settings.chat_model,
        "reason": status.reason,
    }
