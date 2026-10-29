"""输出不含密钥和原始文档内容的本机运行诊断。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import config
from core.model_gateway import ModelGateway
from utils.local_runtime import TaskStore
from utils.tls import tls_options


def main() -> int:
    report = {"lan_enabled": config.LAN_ENABLED, "host": config.HOST, "port": config.PORT}
    try:
        ModelGateway(config.ONEAPI_BASE_URL, config.ONEAPI_API_KEY, config.MODEL_GATEWAY_ALLOWED_HOSTS)
        report["model_gateway"] = "ok"
    except Exception as exc:  # configuration diagnosis intentionally exposes only message
        report["model_gateway"] = f"error: {exc}"
    try:
        tls_options()
        report["tls"] = "ok" if config.LAN_ENABLED else "not_required"
    except Exception as exc:
        report["tls"] = f"error: {exc}"
    store = TaskStore(Path(config.RUNTIME_DIR) / "runtime.db")
    report["runtime_db"] = str(store.path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
