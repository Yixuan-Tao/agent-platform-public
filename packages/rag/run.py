"""生产启动入口：局域网模式强制 HTTPS，本机模式走明文 HTTP。"""
import uvicorn
from config import config
from utils.tls import tls_options

if __name__ == "__main__":
    uvicorn.run(
        "main:app", host=config.HOST, port=config.PORT,
        reload=False, log_level="info", **tls_options(),
    )
