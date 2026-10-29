"""专家 Agent 注册入口。"""

from __future__ import annotations

from typing import Any

from loguru import logger

from agent_platform.agents.emc_compare.agent import build_emc_compare_agent
from agent_platform.agents.emc_compare.manifest import MANIFEST as EMC_MANIFEST
from agent_platform.agents.hardware_qa.agent import build_hardware_qa_agent
from agent_platform.agents.hardware_qa.manifest import MANIFEST as HW_MANIFEST
from agent_platform.agents.info_collect.agent import build_info_collect_agent
from agent_platform.agents.info_collect.manifest import MANIFEST as INFO_MANIFEST
from agent_platform.agents.it_support.agent import build_it_support_agent
from agent_platform.agents.it_support.manifest import MANIFEST as IT_MANIFEST


def register_all(registry: Any) -> None:
    """把四个专家 Agent 注册进 AgentRegistry。"""
    for manifest, builder in (
        (IT_MANIFEST, build_it_support_agent),
        (HW_MANIFEST, build_hardware_qa_agent),
        (EMC_MANIFEST, build_emc_compare_agent),
        (INFO_MANIFEST, build_info_collect_agent),
    ):
        try:
            registry.register(manifest, builder)
        except ValueError as exc:
            logger.warning("跳过重复注册 {}: {}", manifest.name, exc)
