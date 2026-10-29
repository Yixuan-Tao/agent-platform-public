# Test values are synthetic fixtures for software behavior.
import pytest
from agent_platform.evaluation.dataset import dataset
from agent_platform.evaluation.tools import FixtureTool
from agent_platform.tools.base import BaseTool

class Tool(BaseTool):
    name = "emc_compare"
    async def execute(self, **kwargs):
        raise AssertionError("Real service must never run in fixture evaluation")

@pytest.mark.parametrize("scenario", ["空检索", "缺失条款", "文档不存在"])
async def test_missing_evidence_is_a_real_fixture_failure(scenario):
    task = next(task for task in dataset() if task["scenario"] == scenario)
    result = await FixtureTool(Tool(), task, {}).execute()
    assert result.status == "partial" and not result.evidence_refs
    assert str(task["expected"]["value_a"]) + " dBµV/m" not in task["goal"]

async def test_unknown_write_fixture_never_reports_created_ticket():
    task = next(task for task in dataset() if task["scenario"] == "外部结果未知")
    tool = Tool()
    tool.name = "jira_create_ticket"
    tickets = {}
    result = await FixtureTool(tool, task, tickets).execute(title="VPN")
    assert result.status == "unknown" and not tickets and not result.external_refs and not result.retryable
