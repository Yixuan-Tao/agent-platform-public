"""网页采集与 RSS 工具（供 info_collect Agent 使用）。"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup

from agent_platform.tools.base import BaseTool

_ALLOWED_SCHEMES = {"http", "https"}


class UnsafeUrlError(ValueError):
    """目标 URL 指向不允许的协议或内网地址。"""


def _is_blocked_host(host: str) -> bool:
    """拒绝内网/环回/链路本地/保留地址，降低 SSRF 风险。

    域名会先解析为 IP 再判断；解析失败时保守拒绝。
    """
    if not host:
        return True
    candidates: list[str] = []
    try:
        candidates.append(str(ipaddress.ip_address(host)))
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return True
        for info in infos:
            sockaddr = info[4]
            if sockaddr and sockaddr[0]:
                candidates.append(str(sockaddr[0]))
    if not candidates:
        return True
    for raw in candidates:
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            return True
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    return False


def ensure_public_url(url: str) -> str:
    """校验 URL 可安全采集；不合法时抛 :class:`UnsafeUrlError`。"""
    parsed = urlparse(url or "")
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"仅支持 http/https，收到: {parsed.scheme or '(空)'}")
    if not parsed.hostname:
        raise UnsafeUrlError("URL 缺少主机名")
    if _is_blocked_host(parsed.hostname):
        raise UnsafeUrlError(f"拒绝采集内网或保留地址: {parsed.hostname}")
    return url


class WebCollectTool(BaseTool):
    name = "web_collect"
    description = "抓取指定 URL 的网页正文文本（用于信息采集、竞品调研）。"
    args_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的网页地址"},
            "max_chars": {"type": "integer", "description": "返回文本最大长度，默认 8000"},
        },
        "required": ["url"],
    }

    async def execute(self, url: str, max_chars: int = 8000, **kwargs: Any) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) agent-platform"}
        try:
            ensure_public_url(url)
        except UnsafeUrlError as exc:
            return f"网页抓取被拒绝（{url}）: {exc}"
        try:
            # 重定向交给 httpx，但每一跳都校验目标，避免跳转到内网
            async with httpx.AsyncClient(
                timeout=30.0, follow_redirects=False
            ) as client:
                target = url
                for _ in range(5):
                    resp = await client.get(target, headers=headers)
                    if resp.is_redirect and resp.headers.get("location"):
                        target = str(httpx.URL(target).join(resp.headers["location"]))
                        ensure_public_url(target)
                        continue
                    break
                else:
                    return f"网页抓取失败（{url}）: 重定向次数过多"
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = " ".join(soup.get_text(separator=" ").split())
        except Exception as exc:  # noqa: BLE001
            return f"网页抓取失败（{url}）: {exc}"
        if not text:
            return f"网页 {url} 无可见正文。"
        return text[: int(max_chars)]


class RssFetchTool(BaseTool):
    name = "rss_fetch"
    description = "读取 RSS/Atom 订阅源，返回最新条目（标题+链接+摘要），用于行业信息采集。"
    args_schema = {
        "type": "object",
        "properties": {
            "feed_url": {"type": "string", "description": "RSS/Atom 地址"},
            "max_items": {"type": "integer", "description": "返回条数，默认 10"},
        },
        "required": ["feed_url"],
    }

    async def execute(self, feed_url: str, max_items: int = 10, **kwargs: Any) -> str:
        try:
            ensure_public_url(feed_url)
        except UnsafeUrlError as exc:
            return f"RSS 读取被拒绝（{feed_url}）: {exc}"
        try:
            feed = await self._fetch(feed_url)
        except Exception as exc:  # noqa: BLE001
            return f"RSS 读取失败（{feed_url}）: {exc}"
        entries = feed.entries[: int(max_items)]
        if not entries:
            return f"订阅源 {feed_url} 没有条目。"
        lines = [f"订阅源: {feed.feed.get('title', feed_url)}，共 {len(entries)} 条："]
        for i, e in enumerate(entries, 1):
            summary = (e.get("summary") or e.get("description") or "")[:200]
            lines.append(f"{i}. {e.get('title', '无标题')}\n   {e.get('link', '')}\n   {summary}")
        return "\n".join(lines)

    async def _fetch(self, feed_url: str):
        import asyncio

        return await asyncio.to_thread(feedparser.parse, feed_url)
