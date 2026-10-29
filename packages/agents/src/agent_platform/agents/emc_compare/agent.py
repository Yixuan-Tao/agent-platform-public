from __future__ import annotations

from typing import Any

from agent_platform.agents.common import build_react_agent, pick_tools
from agent_platform.agents.emc_compare.prompts import SYSTEM_PROMPT
from agent_platform.tools.builtin.emc_service import (
    EmcCompareTool,
    EmcComponentCheckTool,
    EmcVisualizeTool,
)

SHARED_TOOLS = ["rag_search"]
# 使用调用 RAG 服务的完整实现（矩阵/引用/真实图表），而不是本地简化逻辑
DOMAIN_TOOLS = [EmcCompareTool(), EmcComponentCheckTool(), EmcVisualizeTool()]


def build_emc_compare_agent(llm_router: Any, tool_registry: Any) -> Any:
    tools = pick_tools(tool_registry, SHARED_TOOLS) + DOMAIN_TOOLS
    return build_react_agent(llm_router, "emc_compare", SYSTEM_PROMPT, tools)
