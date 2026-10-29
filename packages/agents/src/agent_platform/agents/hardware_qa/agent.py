from __future__ import annotations

from typing import Any

from agent_platform.agents.common import build_react_agent, pick_tools
from agent_platform.agents.hardware_qa.prompts import SYSTEM_PROMPT

TOOL_NAMES = ["rag_search", "emc_visualize"]


def build_hardware_qa_agent(llm_router: Any, tool_registry: Any) -> Any:
    tools = pick_tools(tool_registry, TOOL_NAMES)
    return build_react_agent(llm_router, "hardware_qa", SYSTEM_PROMPT, tools)
