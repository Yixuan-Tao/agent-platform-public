"""逐个探测已配置模型是否可用，用于换模型/换网关后的自检。

只输出模型名、端点主机、状态与耗时，绝不打印密钥。

用法（packages/rag 目录）：
    python scripts/check_model_availability.py
    python scripts/check_model_availability.py --json metrics/model_availability.json

退出码：全部可用为 0，存在不可用项为 1（便于接入健康检查/CI）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config  # noqa: E402
from utils.urls import build_embeddings_url, build_rerank_url  # noqa: E402


@dataclass
class Probe:
    role: str
    model: str
    host: str
    ok: bool
    status: str
    latency_s: float
    detail: str = ""
    # required=True 表示该角色缺失会直接影响主链路；诊断项不计入退出码
    required: bool = True


def _host(url: str) -> str:
    parsed = urlparse(url)
    return parsed.hostname or "(invalid)"


def _join_chat_url(base: str) -> str:
    """拼接 chat completions URL，避免 base 已含 /v1 时出现 /v1/v1/...。

    与 utils.urls.build_embeddings_url 同一套归一化规则。
    """
    base = (base or "").rstrip("/")
    path = config.CHAT_PATH or "/v1/chat/completions"
    if path.startswith("/v1/") and base.endswith("/v1"):
        path = path[len("/v1"):]
    return f"{base}{path}"


def _probe(
    role: str,
    url: str,
    model: str,
    key: str,
    *,
    payload: dict,
    timeout: float = 60.0,
    required: bool = True,
) -> Probe:
    started = time.time()
    if not key:
        return Probe(role, model, _host(url), False, "missing_key", 0.0, "未配置 API Key", required)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
    except Exception as exc:  # noqa: BLE001
        return Probe(
            role, model, _host(url), False, "unreachable", round(time.time() - started, 3),
            f"{type(exc).__name__}: {exc}", required,
        )
    latency = round(time.time() - started, 3)
    if resp.status_code == 200:
        return Probe(role, model, _host(url), True, "ok", latency, required=required)
    detail = resp.text[:200].replace("\n", " ")
    return Probe(
        role, model, _host(url), False, f"http_{resp.status_code}", latency, detail, required
    )


def probe_all() -> list[Probe]:
    results: list[Probe] = []
    chat_base = config.CHAT_BASE_URL
    chat_key = config.CHAT_API_KEY
    chat_url = _join_chat_url(chat_base)
    chat_payload = {
        "model": config.QWEN_CHAT_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
        **config.CHAT_EXTRA_BODY,
    }
    results.append(_probe("chat", chat_url, config.QWEN_CHAT_MODEL, chat_key, payload=chat_payload))

    # 视觉模型（多模态解析依赖）
    results.append(
        _probe(
            "vision@chat_gateway",
            chat_url,
            config.QWEN_VL_MODEL,
            chat_key,
            payload={
                "model": config.QWEN_VL_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
                **config.CHAT_EXTRA_BODY,
            },
            # 诊断项：CHAT_BASE_URL 指向非百炼网关时，VL 不该走这里；
            # 产品代码的 VL 路径是 ONEAPI_BASE_URL（见 core/parser.py）
            required=False,
        )
    )

    # 视觉模型的实际归属网关（统一 LLM 网关）：用于区分
    # "百炼侧模型不可用" 与 "CHAT_BASE_URL 指向的网关不支持该模型"。
    # 产品实际使用的视觉路径：解析器用 ONEAPI_BASE_URL 调 VL
    llm_gateway_url = _join_chat_url(config.ONEAPI_BASE_URL)
    results.append(
        _probe(
            "vision",
            llm_gateway_url,
            config.QWEN_VL_MODEL,
            config.ONEAPI_API_KEY,
            payload={
                "model": config.QWEN_VL_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
                "enable_thinking": False,
            },
        )
    )

    # 文本向量（检索依赖）
    results.append(
        _probe(
            "embedding",
            build_embeddings_url(),
            config.TEXT_EMBED_MODEL,
            config.ONEAPI_API_KEY,
            payload={"model": config.TEXT_EMBED_MODEL, "input": ["ping"]},
        )
    )

    # 重排（可选增强，失败会降级但不影响主链路）
    results.append(
        _probe(
            "rerank",
            build_rerank_url(),
            config.RERANK_MODEL,
            config.ONEAPI_API_KEY,
            payload={
                "model": config.RERANK_MODEL,
                "query": "ping",
                "documents": ["ping"],
                "top_n": 1,
            },
        )
    )

    # 图片向量（百炼原生多模态协议，不支持 OpenAI 兼容）
    results.append(
        _probe(
            "image_embedding",
            config.DASHSCOPE_MULTIMODAL_URL,
            config.IMAGE_EMBED_MODEL,
            config.DASHSCOPE_API_KEY,
            payload={
                "model": config.IMAGE_EMBED_MODEL,
                "input": {"contents": [{"text": "ping"}]},
                "parameters": {"dimension": config.IMAGE_EMBED_DIM},
            },
            timeout=120.0,
            # 图像向量是可选增强（ENABLE_VISUAL_SEARCH 默认关闭）
            required=False,
        )
    )
    return results


def list_models(base_url: str, key: str, role: str) -> tuple[Probe, list[str]]:
    """列出网关可用模型，用于确认"是模型名不对还是模型确实没有"。"""
    url = f"{base_url.rstrip('/')}/models"
    started = time.time()
    if not key:
        return Probe(role, "-", _host(url), False, "missing_key", 0.0, "未配置 API Key"), []
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers={"Authorization": f"Bearer {key}"})
    except Exception as exc:  # noqa: BLE001
        return (
            Probe(role, "-", _host(url), False, "unreachable",
                  round(time.time() - started, 3), f"{type(exc).__name__}: {exc}"),
            [],
        )
    latency = round(time.time() - started, 3)
    if resp.status_code != 200:
        return (
            Probe(role, "-", _host(url), False, f"http_{resp.status_code}", latency,
                  resp.text[:200].replace("\n", " ")),
            [],
        )
    try:
        data = resp.json()
    except ValueError:
        return Probe(role, "-", _host(url), False, "bad_json", latency), []
    ids = sorted(
        str(item.get("id", "")) for item in (data.get("data") or []) if item.get("id")
    )
    return Probe(role, "-", _host(url), True, "ok", latency, f"{len(ids)} 个模型"), ids


def main() -> int:
    parser = argparse.ArgumentParser(description="模型可用性自检")
    parser.add_argument("--json", help="把结果写入 JSON 文件")
    parser.add_argument("--list-models", action="store_true", help="同时列出各网关可用模型")
    args = parser.parse_args()

    probes = probe_all()

    catalogs: dict[str, list[str]] = {}
    if args.list_models:
        seen_bases: set[str] = set()
        for role, base, key in (
            ("catalog@chat_gateway", config.CHAT_BASE_URL, config.CHAT_API_KEY),
            ("catalog@llm_gateway", config.ONEAPI_BASE_URL, config.ONEAPI_API_KEY),
        ):
            if base in seen_bases:
                continue
            seen_bases.add(base)
            probe, ids = list_models(base, key, role)
            probe.required = False
            probes.append(probe)
            catalogs[role] = ids
    width = max(len(p.role) for p in probes)
    for p in probes:
        flag = "OK  " if p.ok else "FAIL"
        tag = "" if p.required else "（诊断项，不影响退出码）"
        print(
            f"[{flag}] {p.role:<{width}}  model={p.model:<34} host={p.host:<32} "
            f"status={p.status:<14} {p.latency_s}s {tag}"
        )
        if p.detail:
            print(f"        └─ {p.detail}")

    failed = [p for p in probes if not p.ok and p.required]
    warnings = [p for p in probes if not p.ok and not p.required]
    payload = {
        "chat_base_url_host": _host(config.CHAT_BASE_URL),
        "results": [asdict(p) for p in probes],
        "failed": [p.role for p in failed],
        "warnings": [p.role for p in warnings],
        "catalogs": catalogs,
    }
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入 {out}")

    if failed:
        print("\n不可用角色：" + ", ".join(f"{p.role}({p.model})" for p in failed))
        return 1
    if warnings:
        print(
            "\n诊断项未通过（不阻塞主链路）："
            + ", ".join(f"{p.role}({p.status})" for p in warnings)
        )
    print("\n全部模型可用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
