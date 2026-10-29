"""平台网关生产启动入口：局域网模式（LAN_ENABLED=true）强制 HTTPS。"""

import os
from pathlib import Path

import uvicorn

from agent_platform.core.config import get_settings


def tls_options() -> dict:
    """局域网模式返回 uvicorn ssl 参数；本机模式返回空（明文 HTTP）。"""
    settings = get_settings()
    if not settings.lan_enabled:
        return {}
    cert, key = Path(settings.tls_cert_file), Path(settings.tls_key_file)
    if not settings.tls_cert_file or not settings.tls_key_file or not cert.is_file() or not key.is_file():
        raise RuntimeError("局域网模式必须配置存在的 TLS_CERT_FILE 和 TLS_KEY_FILE")
    return {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}


if __name__ == "__main__":
    settings = get_settings()
    host = os.getenv("API_HOST") or settings.api_host
    port = int(os.getenv("API_PORT") or settings.api_port)
    uvicorn.run(
        "app.main:app", host=host, port=port,
        reload=False, log_level="info", **tls_options(),
    )
