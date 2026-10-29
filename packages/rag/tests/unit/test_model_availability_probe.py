# Test values are synthetic fixtures for software behavior.
"""模型自检脚本的 URL 归一化测试（避免 /v1/v1 与误报不可用）。"""

from importlib import util
from pathlib import Path
import sys

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_model_availability.py"
spec = util.spec_from_file_location("check_model_availability", _SCRIPT)
module = util.module_from_spec(spec)
assert spec.loader is not None
# dataclass 解析注解时需要模块已注册进 sys.modules
sys.modules[spec.name] = module
spec.loader.exec_module(module)

join = module._join_chat_url


def test_join_chat_url_avoids_double_v1_when_base_ends_with_v1():
    assert (
        join("https://dashscope.aliyuncs.com/compatible-mode/v1")
        == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )


def test_join_chat_url_keeps_v1_when_base_has_no_version():
    assert join("https://api.deepseek.com") == "https://api.deepseek.com/v1/chat/completions"


def test_join_chat_url_is_idempotent_for_trailing_slash():
    assert join("https://api.deepseek.com/") == join("https://api.deepseek.com")


def test_probe_marks_diagnostics_as_not_required():
    """诊断项不应影响退出码：required=False 的失败只作为提示。"""
    probe = module._probe(
        "diagnostic",
        "http://127.0.0.1:1/v1/chat/completions",
        "m",
        "k",
        payload={},
        timeout=0.5,
        required=False,
    )
    assert probe.ok is False
    assert probe.required is False
    assert probe.status in {"unreachable", "http_0"}
