from __future__ import annotations

from typing import Any

from agent_platform.agents.common import build_react_agent, pick_tools
from agent_platform.agents.info_collect.prompts import SYSTEM_PROMPT

TOOL_NAMES = ["web_collect", "rss_fetch"]


def build_info_collect_agent(llm_router: Any, tool_registry: Any) -> Any:
    tools = pick_tools(tool_registry, TOOL_NAMES)
    return build_react_agent(llm_router, "info_collect", SYSTEM_PROMPT, tools)
