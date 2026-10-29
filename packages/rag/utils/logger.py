"""
统一日志模块 — 支持控制台 + 文件双输出
"""
import logging
import sys
from pathlib import Path
from config import config

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(name: str) -> logging.Logger:
    """创建带统一格式的 logger 实例"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    file_handler = logging.FileHandler(LOG_DIR / "rag.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# 默认 logger
log = setup_logger("rag")
