"""
统一重试模块 — 为 OneAPI 网关调用提供指数退避 + 抖动重试。

设计原则：
  - 429 (Rate Limit): 指数退避，尊重 Retry-After 响应头
  - 5xx (Server Error): 指数退避，短间隔
  - 网络异常 (ConnectionError/Timeout): 指数退避
  - 支持同步函数和 httpx 请求两种模式
"""
from __future__ import annotations

import random
import time
from functools import wraps
from typing import Callable, TypeVar

import httpx

from utils.logger import log
from utils.runtime_budget import model_post

F = TypeVar("F", bound=Callable)

# 默认重试配置
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # 基础等待秒数
DEFAULT_MAX_DELAY = 60.0  # 最大等待秒数
DEFAULT_JITTER = 0.3  # 抖动系数 (0~1)，在 base 基础上 ±30% 随机


def _calc_delay(attempt: int, base_delay: float, max_delay: float, jitter: float) -> float:
    """计算本次重试等待时间：指数退避 + 随机抖动。"""
    delay = min(base_delay * (2 ** attempt), max_delay)
    # 添加随机抖动，避免惊群效应
    jitter_range = delay * jitter
    delay += random.uniform(-jitter_range, jitter_range)
    return max(0.1, delay)


def _parse_retry_after(headers: dict) -> float | None:
    """从响应头解析 Retry-After（秒数或 HTTP-date）。"""
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        # HTTP-date 格式，解析为秒数
        try:
            from email.utils import parsedate_to_datetime
            retry_dt = parsedate_to_datetime(raw.strip())
            return max(0, (retry_dt.timestamp() - time.time()))
        except Exception:
            return None


def retry_on_rate_limit(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
    retry_on_5xx: bool = True,
):
    """装饰器：为同步函数添加 429/5xx 重试。

    被装饰函数应抛出异常来表示失败，装饰器会捕获并重试。
    适用于 openai SDK 调用等场景。

    Args:
        max_retries: 最大重试次数
        base_delay: 基础等待秒数
        max_delay: 最大等待秒数
        jitter: 抖动系数
        retry_on_5xx: 是否对 5xx 错误也重试
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    # 判断是否可重试
                    retryable = _is_retryable_error(e)
                    if not retryable or attempt >= max_retries:
                        if attempt > 0:
                            log.warning(
                                "%s 重试 %d 次后仍失败: %s",
                                func.__name__, attempt, e,
                            )
                        raise

                    # 解析 Retry-After 或使用指数退避
                    retry_after = _get_retry_after_from_exc(e)
                    if retry_after is not None:
                        delay = min(retry_after, max_delay)
                    else:
                        delay = _calc_delay(attempt, base_delay, max_delay, jitter)

                    log.warning(
                        "%s 第 %d/%d 次重试 (429/5xx), 等待 %.1fs: %s",
                        func.__name__, attempt + 1, max_retries, delay, str(e)[:100],
                    )
                    time.sleep(delay)
            # 理论上不会到这里，但作为兜底
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


def httpx_retry_post(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
    timeout: float = 30.0,
) -> Callable[..., httpx.Response]:
    """带重试的 httpx.post 封装。

    对 429/5xx/网络错误自动重试，返回 Response 或抛出最后一次异常。

    Args:
        max_retries: 最大重试次数
        base_delay: 基础等待秒数
        max_delay: 最大等待秒数
        jitter: 抖动系数
        timeout: 请求超时秒数

    Returns:
        可调用对象，签名与 httpx.post 一致
    """

    def _post(url: str, **kwargs) -> httpx.Response:
        kwargs.setdefault("timeout", timeout)
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                resp = model_post(url, **kwargs)
                status = resp.status_code

                # 429 限流 → 重试
                if status == 429:
                    retry_after = _parse_retry_after(dict(resp.headers))
                    delay = min(retry_after or base_delay, max_delay) if retry_after else _calc_delay(attempt, base_delay, max_delay, jitter)
                    if attempt < max_retries:
                        log.warning(
                            "httpx POST %s 返回 429, 第 %d/%d 次重试, 等待 %.1fs",
                            url[:80], attempt + 1, max_retries, delay,
                        )
                        time.sleep(delay)
                        continue
                    return resp  # 最后一次不重试，返回原始响应

                # 5xx 服务端错误 → 重试
                if 500 <= status < 600:
                    if attempt < max_retries:
                        delay = _calc_delay(attempt, base_delay, max_delay, jitter)
                        log.warning(
                            "httpx POST %s 返回 %d, 第 %d/%d 次重试, 等待 %.1fs",
                            url[:80], status, attempt + 1, max_retries, delay,
                        )
                        time.sleep(delay)
                        continue
                    return resp

                return resp

            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException,
                    httpx.RemoteProtocolError, httpx.NetworkError) as e:
                last_exc = e
                if attempt < max_retries:
                    delay = _calc_delay(attempt, base_delay, max_delay, jitter)
                    log.warning(
                        "httpx POST %s 网络异常, 第 %d/%d 次重试, 等待 %.1fs: %s",
                        url[:80], attempt + 1, max_retries, delay, str(e)[:80],
                    )
                    time.sleep(delay)
                    continue
                raise

            except Exception:
                # 其他异常不重试，直接抛出
                raise

        # 所有重试耗尽
        if last_exc:
            raise last_exc
        # 返回最后一次响应（理论上已被 return 捕获）
        raise RuntimeError("httpx_retry_post: 不可达分支")

    return _post


def _is_retryable_error(exc: Exception) -> bool:
    """判断 openai SDK 异常是否可重试（429 / 5xx / 网络错误）。"""
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__

    # 429 限流
    if "429" in exc_str or "rate limit" in exc_str or "too many requests" in exc_str:
        return True
    # 5xx 服务端错误
    if any(f" {code}" in exc_str or f"error code: {code}" in exc_str or f"status {code}" in exc_str
           for code in range(500, 600)):
        return True
    # 网络/连接错误
    if exc_type in ("APIConnectionError", "APITimeoutError", "ConnectionError",
                    "Timeout", "ReadTimeout", "ConnectTimeout"):
        return True
    if "connection" in exc_str or "timeout" in exc_str:
        return True

    return False


def _get_retry_after_from_exc(exc: Exception) -> float | None:
    """从 openai SDK 异常中尝试提取 Retry-After 信息。"""
    # openai SDK 的 RateLimitError 通常有 response 属性
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", {}) or {}
        return _parse_retry_after(headers)
    # 尝试从 body 中解析
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        retry = body.get("retry_after") or body.get("Retry-After")
        if retry:
            try:
                return float(retry)
            except (ValueError, TypeError):
                pass
    return None
