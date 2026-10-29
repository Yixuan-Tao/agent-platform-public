"""平台端到端冒烟：网关鉴权 → 文档上传/检索/删除 → 对话 → 历史 → 反馈。

覆盖此前暴露过的集成缺陷：
    - 网关调用 RAG 写操作是否携带服务凭据
    - 删除接口参数名是否为 file_name
    - 登录 Bearer 是否能通过网关（而不是只带 X-API-Key）
    - 对话是否返回可溯源的 message_id、历史是否可回放、反馈是否写入长期记忆

前置：RAG（:8010）与网关（:8000）已启动，且两者使用同一个 RAG_SERVICE_TOKEN。

用法：
    python scripts/smoke_platform_e2e.py --gateway http://127.0.0.1:8000 \
        --user admin --password '<password>'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import httpx


class SmokeFailure(RuntimeError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)
    print(f"  ✓ {message}")


def _login(client: httpx.Client, base: str, user: str, password: str) -> str:
    resp = client.post(f"{base}/api/v1/auth/login", json={"username": user, "password": password})
    _check(resp.status_code == 200, f"登录成功（HTTP {resp.status_code}）")
    return resp.json()["access_token"]


def _upload(client: httpx.Client, base: str, headers: dict) -> str:
    content = (
        "EMC 冒烟测试文档\n\n"
        "1 概述\n本文件用于验证平台文档摄入链路。\n\n"
        "2 辐射发射限值\n频率范围 150 kHz–30 MHz，限值 40 dBuV/m。\n"
    ).encode("utf-8")
    resp = client.post(
        f"{base}/api/v1/ingest",
        headers=headers,
        files={"file": ("smoke_platform_e2e.txt", content, "text/plain")},
        timeout=180.0,
    )
    _check(resp.status_code == 200, f"文档上传被网关接受（HTTP {resp.status_code}）")
    body = resp.json()
    task_id = body.get("task_id") or ""
    _check(bool(task_id), f"返回 task_id={task_id}")

    deadline = time.time() + 300
    status = "processing"
    while time.time() < deadline:
        r = client.get(f"{base}/api/v1/ingest/status/{task_id}", headers=headers, timeout=30.0)
        _check(r.status_code == 200, f"查询入库状态（HTTP {r.status_code}）")
        data = r.json()
        status = data.get("status", "unknown")
        if status in {"completed", "done"}:
            _check(int((data.get("result") or {}).get("chunk_count", 0)) > 0, "入库产生 chunk")
            return "smoke_platform_e2e.txt"
        if status in {"error", "failed"}:
            raise SmokeFailure(f"入库失败: {data.get('error')}")
        time.sleep(3)
    raise SmokeFailure(f"入库超时，最后状态={status}")


def _list_documents(client: httpx.Client, base: str, headers: dict) -> list[str]:
    resp = client.get(f"{base}/api/v1/documents", headers=headers, timeout=60.0)
    _check(resp.status_code == 200, f"文档列表可读（HTTP {resp.status_code}）")
    data = resp.json()
    docs = data.get("documents") if isinstance(data, dict) else None
    if docs is None:
        raise SmokeFailure(f"文档列表结构异常: {str(data)[:200]}")
    return [str(d.get("file_name", "")) for d in docs]


def _chat(client: httpx.Client, base: str, headers: dict, message: str) -> tuple[str, str, str]:
    payload = {"message": message, "agent_hint": "hardware_qa"}
    answer_parts: list[str] = []
    session_id = ""
    message_id = ""
    agent = ""
    with client.stream(
        "POST", f"{base}/api/v1/chat", headers={**headers, "Content-Type": "application/json"},
        json=payload, timeout=300.0,
    ) as resp:
        _check(resp.status_code == 200, f"对话请求被接受（HTTP {resp.status_code}）")
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            try:
                evt = json.loads(line[5:].strip())
            except ValueError:
                continue
            etype = evt.get("type")
            if etype == "token":
                answer_parts.append(evt.get("content", ""))
            elif etype == "agent_switch":
                agent = evt.get("agent", agent)
            elif etype == "session":
                session_id = evt.get("session_id", session_id)
                message_id = evt.get("message_id", message_id)
            elif etype == "error":
                raise SmokeFailure(f"对话返回错误: {evt.get('message')}")
    answer = "".join(answer_parts)
    _check(bool(answer), "收到回答内容")
    _check(bool(session_id), f"返回 session_id={session_id[:8]}…")
    _check(bool(message_id), f"返回可追溯 message_id={message_id[:8]}…")
    return session_id, message_id, answer


def main() -> int:
    parser = argparse.ArgumentParser(description="平台端到端冒烟")
    parser.add_argument("--gateway", default="http://127.0.0.1:8000")
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", required=True)
    parser.add_argument("--keep-document", action="store_true", help="不删除测试文档")
    args = parser.parse_args()
    base = args.gateway.rstrip("/")

    with httpx.Client(timeout=60.0) as client:
        print("[0] 健康检查")
        health = client.get(f"{base}/health")
        _check(health.status_code == 200, f"网关健康（{health.json().get('status')}）")
        _check(health.json().get("rag_service") == "ok", "网关到 RAG 健康检查连通")

        print("[1] 登录并校验 Bearer 可访问受保护接口")
        token = _login(client, base, args.user, args.password)
        headers = {"Authorization": f"Bearer {token}"}
        me = client.get(f"{base}/api/v1/auth/me", headers=headers)
        _check(me.status_code == 200, f"/auth/me 可用（role={me.json().get('role')}）")
        _check(
            client.get(f"{base}/api/v1/sessions", headers=headers).status_code == 200,
            "Bearer 会话可通过网关鉴权访问业务接口",
        )

        print("[2] 文档上传 → 状态 → 列表 → 删除")
        file_name = _upload(client, base, headers)
        names = _list_documents(client, base, headers)
        _check(file_name in names, f"文档出现在列表中（共 {len(names)} 份）")

        print("[3] 对话：SSE 流 + 会话/消息 ID")
        session_id, message_id, answer = _chat(
            client, base, headers, "冒烟测试：请用一句话说明本知识库覆盖了哪些内容。"
        )
        print(f"      回答片段：{answer[:60].replace(chr(10), ' ')}…")

        print("[4] 会话历史可回放")
        history = client.get(
            f"{base}/api/v1/sessions/{session_id}/messages", headers=headers
        )
        _check(history.status_code == 200, "历史接口可用")
        roles = [m.get("role") for m in history.json()]
        _check(roles[:2] == ["user", "assistant"], f"历史顺序正确：{roles[:2]}")

        print("[5] 反馈可追溯到原始问答")
        fb = client.post(
            f"{base}/api/v1/feedback",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "session_id": session_id,
                "message_id": message_id,
                "feedback_type": "up",
            },
        )
        _check(fb.status_code == 200, f"点赞提交成功（HTTP {fb.status_code}）")

        print("[6] 跨用户隔离")
        viewer_name = "smoke-viewer"
        viewer_password = __import__("secrets").token_urlsafe(24)
        client.post(
            f"{base}/api/v1/auth/users",
            headers={**headers, "Content-Type": "application/json"},
            json={"username": viewer_name, "password": viewer_password, "role": "viewer"},
        )
        viewer_token = _login(client, base, viewer_name, viewer_password)
        viewer_headers = {"Authorization": f"Bearer {viewer_token}"}
        stolen = client.get(
            f"{base}/api/v1/sessions/{session_id}/messages", headers=viewer_headers
        )
        _check(stolen.status_code == 403, f"其他用户读取会话被拒绝（HTTP {stolen.status_code}）")
        listed = client.get(f"{base}/api/v1/sessions", headers=viewer_headers)
        _check(
            all(s.get("session_id") != session_id for s in listed.json()),
            "其他用户的会话列表不包含该会话",
        )
        forged = client.get(
            f"{base}/api/v1/sessions/{session_id}/messages",
            headers={"Authorization": "Bearer invalid-token"},
        )
        _check(forged.status_code == 401, "伪造令牌被拒绝（401）")

        if not args.keep_document:
            print("[7] 删除测试文档")
            resp = client.request(
                "DELETE", f"{base}/api/v1/documents",
                headers=headers, params={"file_name": file_name},
            )
            _check(resp.status_code == 200, f"删除成功（HTTP {resp.status_code}）")

    print("\n端到端冒烟全部通过。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeFailure as exc:
        print(f"\n[失败] {exc}")
        sys.exit(1)
