"""loguru 日志配置。"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO", log_dir: str | None = None) -> None:
    """移除默认 handler，输出到 stderr（可选同时写文件）。"""
    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=True, backtrace=False, diagnose=False)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logger.add(
            Path(log_dir) / "app.log",
            rotation="10 MB",
            retention=7,
            level=level,
            enqueue=True,
            encoding="utf-8",
        )
