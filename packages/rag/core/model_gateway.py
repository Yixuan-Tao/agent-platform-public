"""可信内网 OpenAI-compatible 网关的统一配置边界。"""
from urllib.parse import urlparse

import httpx
from openai import OpenAI


class GatewayConfigurationError(ValueError):
    pass


class ModelGateway:
    def __init__(self, base_url: str, api_key: str, allowed_hosts: list[str]):
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise GatewayConfigurationError("模型网关必须使用 HTTPS URL")
        if parsed.hostname.lower() not in {host.lower() for host in allowed_hosts}:
            raise GatewayConfigurationError("模型网关主机不在允许列表")
        if not api_key:
            raise GatewayConfigurationError("缺少模型网关 API Key")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key

    def create_chat_client(self, *, timeout: float, max_retries: int = 0) -> OpenAI:
        """构造受统一网关配置约束的 OpenAI-compatible 客户端。

        API Key 只在客户端内存中使用，不向调用方暴露；所有 Chat/Vision 调用
        通过此方法创建，避免各模块各自拼接未校验的网关地址。
        """
        return OpenAI(
            base_url=self.base_url,
            api_key=self._api_key,
            timeout=timeout,
            max_retries=max_retries,
            http_client=httpx.Client(
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
            ),
        )

    def endpoint(self, path: str) -> str:
        """返回经校验网关的 API 端点，避免出现重复的 ``/v1``。"""
        suffix = path if path.startswith("/") else f"/{path}"
        if self.base_url.endswith("/v1") and suffix.startswith("/v1/"):
            suffix = suffix[3:]
        return f"{self.base_url}{suffix}"

    def headers(self) -> dict[str, str]:
        """为 HTTP-compatible 端点提供认证头，不泄漏密钥到日志或响应。"""
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
