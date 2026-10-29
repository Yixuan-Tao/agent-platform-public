# Test values are synthetic fixtures for software behavior.
from types import SimpleNamespace
import httpx
import pytest
from agent_platform.tools.builtin.jira_tool import JiraCreateTicketTool
from agent_platform.core.config import Settings
from agent_platform.tools.result import ToolResult

@pytest.mark.parametrize("returned_title,expected", [("VPN", "success"), ("other user's issue", "partial")])
async def test_jira_v3_adf_and_actual_title_verification(monkeypatch, returned_title, expected):
    captured = []
    settings = Settings(_env_file=None, jira_url="https://jira.example", jira_user="fixture", jira_token="test-only")
    monkeypatch.setattr("agent_platform.tools.builtin.jira_tool.get_settings", lambda: settings)
    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, **kwargs):
            captured.append(kwargs["json"])
            return httpx.Response(201, json={"key":"IT-1"}, request=httpx.Request("POST", url))
        async def get(self, url, **kwargs):
            return httpx.Response(200, json={"key":"IT-1", "fields":{"summary":returned_title}}, request=httpx.Request("GET", url))
    monkeypatch.setattr("agent_platform.tools.builtin.jira_tool.httpx.AsyncClient", Client)
    result = await JiraCreateTicketTool().execute("VPN", "连接失败", priority="高")
    assert result.tool_result.status == expected
    assert captured[0]["fields"]["description"]["type"] == "doc"
    assert captured[0]["fields"]["priority"]["name"] == "High"
