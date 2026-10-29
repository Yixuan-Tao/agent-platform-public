# Test values are synthetic fixtures for software behavior.
"""部署配置不变量测试：把此前踩过的坑固化成断言。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE = REPO_ROOT / "deploy" / "docker-compose.yml"
API_DOCKERFILE = REPO_ROOT / "deploy" / "Dockerfile.api"


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_compose_is_valid_yaml(compose):
    assert set(compose["services"]) >= {"rag", "postgres", "redis", "api", "web"}


def test_api_persists_runtime_dir_on_volume(compose):
    """账号/审计库必须落在挂载卷内，否则容器重建即丢失。"""
    api = compose["services"]["api"]
    assert api["environment"]["RUNTIME_DIR"] == "/data/runtime_data"
    mounts = [v for v in api.get("volumes", [])]
    assert any(v.endswith(":/data") for v in mounts), mounts


def test_api_holds_rag_service_token(compose):
    api_env = compose["services"]["api"]["environment"]
    rag_env = compose["services"]["rag"]["environment"]
    assert "RAG_SERVICE_TOKEN" in api_env
    assert "RAG_SERVICE_URL" in api_env
    # 两端必须来自同一变量，否则非 LAN 模式写操作会 401
    assert rag_env.get("LOCAL_API_TOKEN") == api_env["RAG_SERVICE_TOKEN"]


def test_api_rag_url_does_not_leak_host_dev_value(compose):
    """容器内必须走 Compose DNS。

    宿主 .env 里 RAG_SERVICE_URL 通常是 http://localhost:8010（本地开发用），
    一旦被容器继承就会指向 api 容器自身，表现为文档接口 502。
    """
    url = compose["services"]["api"]["environment"]["RAG_SERVICE_URL"]
    assert "${RAG_SERVICE_URL" not in url, "不应复用宿主机 RAG_SERVICE_URL"
    assert "${RAG_INTERNAL_URL:-http://rag:8010}" == url


def test_orchestration_knobs_are_passed_to_api(compose):
    """README/.env.example 里承诺的编排开关必须在容器里真正生效。"""
    api_env = compose["services"]["api"]["environment"]
    for key in (
        "ANSWER_STREAMING",
        "AGENT_CALL_BUDGET",
        "USE_ROUTER_REACT",
        "PLAN_MODE",
        "MAX_PLAN_STEPS",
        "MAX_AGENT_CALLS",
    ):
        assert key in api_env, f"compose 未把 {key} 传给 api 容器"
    assert api_env["ANSWER_STREAMING"] == "${ANSWER_STREAMING:-false}"


def test_databases_not_published_to_host(compose):
    for name in ("postgres", "redis"):
        assert "ports" not in compose["services"][name], f"{name} 不应发布宿主机端口"


def test_host_ports_are_configurable(compose):
    """宿主端口必须可覆盖：本机其它服务（如 qiuzhao-desk）常占用 8000。"""
    api_ports = compose["services"]["api"]["ports"]
    web_ports = compose["services"]["web"]["ports"]
    rag_ports = compose["services"]["rag"]["ports"]
    assert api_ports == ["${API_HOST_PORT:-8000}:8000"]
    assert web_ports == ["${WEB_HOST_PORT:-3000}:3000"]
    assert rag_ports == ["${RAG_HOST_PORT:-8010}:8010"]
    # 前端构建期需要注入 API 地址（Next.js 会内联 NEXT_PUBLIC_*）
    web_build_args = compose["services"]["web"]["build"]["args"]
    assert "NEXT_PUBLIC_API_URL" in web_build_args
    dockerfile = (REPO_ROOT / "apps" / "web" / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG NEXT_PUBLIC_API_URL" in dockerfile
    assert dockerfile.index("ARG NEXT_PUBLIC_API_URL") < dockerfile.index("RUN npm run build")


def test_postgres_password_has_no_weak_default(compose):
    env = compose["services"]["postgres"]["environment"]
    value = str(env["POSTGRES_PASSWORD"])
    assert value.strip() == "${POSTGRES_PASSWORD:-}", "不应提供默认口令"
    # DATABASE_URL 必须复用同一批变量，避免口令不一致
    url = compose["services"]["api"]["environment"]["DATABASE_URL"]
    assert "${POSTGRES_PASSWORD:-}" in url
    assert "${POSTGRES_DB:-agent_platform}" in url


def test_api_image_uses_tls_aware_entrypoint():
    """直接 uvicorn 会绕过 run.py 的 LAN/TLS 处理。"""
    text = API_DOCKERFILE.read_text(encoding="utf-8")
    assert 'CMD ["python", "run.py"]' in text
    assert "uvicorn" not in text.split("CMD")[-1]


def test_dockerignore_files_exist_and_exclude_secrets():
    for rel in (".dockerignore", "apps/web/.dockerignore", "packages/rag/.dockerignore"):
        path = REPO_ROOT / rel
        assert path.is_file(), f"缺少 {rel}"
        body = path.read_text(encoding="utf-8")
        assert ".env" in body, f"{rel} 未排除 .env"
        assert any(
            token in body for token in ("chroma_data", "node_modules", "runtime_data")
        ), f"{rel} 未排除运行数据/依赖"
