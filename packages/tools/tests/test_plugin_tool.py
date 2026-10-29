# Test values are synthetic fixtures for software behavior.
"""插件工具（沙箱适配）测试。"""

from __future__ import annotations

import json

from agent_platform.tools.gateway import ToolError
from agent_platform.tools.plugin_tool import PluginTool, scan_plugins


async def test_plugin_tool_execute():
    code = (
        "import json,sys; d=json.load(sys.stdin); "
        "print(json.dumps({'ok':True,'double':d['args']['x']*2}))"
    )
    tool = PluginTool("double", "翻倍", ["python", "-c", code])
    result = json.loads(await tool.execute(x=21))
    assert result["double"] == 42


async def test_plugin_tool_error():
    code = (
        "import json,sys; d=json.load(sys.stdin); "
        "print(json.dumps({'ok':False,'error':'boom'}))"
    )
    tool = PluginTool("boom", "失败插件", ["python", "-c", code])
    try:
        await tool.execute(x=1)
        assert False, "应抛 ToolError"
    except ToolError as exc:
        assert exc.error_code == "PLUGIN_ERROR"


async def test_scan_plugins(tmp_path):
    pdir = tmp_path / "plugins"
    (pdir / "demo").mkdir(parents=True)
    (pdir / "demo" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "demo_plugin",
                "description": "demo",
                "command": [
                    "python",
                    "-c",
                    "import json,sys; d=json.load(sys.stdin); print(json.dumps({'ok':True}))",
                ],
            }
        ),
        encoding="utf-8",
    )
    plugins = scan_plugins(str(pdir))
    assert len(plugins) == 1
    assert plugins[0].name == "demo_plugin"
    assert await plugins[0].execute()  # 可正常调用
