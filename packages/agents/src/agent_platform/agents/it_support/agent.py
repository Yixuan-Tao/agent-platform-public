from __future__ import annotations

from typing import Any

from agent_platform.agents.common import build_react_agent, pick_tools
from agent_platform.agents.it_support.prompts import SYSTEM_PROMPT

TOOL_NAMES = ["rag_search", "jira_create_ticket", "jira_get_ticket", "escalate_human"]


def build_it_support_agent(llm_router: Any, tool_registry: Any) -> Any:
    tools = pick_tools(tool_registry, TOOL_NAMES)
    return build_react_agent(llm_router, "it_support", SYSTEM_PROMPT, tools)
