from agent_platform.core.models import AgentManifest

MANIFEST = AgentManifest(
    name="emc_compare",
    description=(
        "EMC 标准对比专家：对两份标准文档做结构化对比（章节路径）、"
        "元件替换参数判断、限值表可视化（Type_A-D 图表）。"
    ),
    capabilities=["emc", "compare", "component_check", "visualize"],
    tools=["emc_compare", "emc_component_check", "emc_visualize"],
    handoff=["hardware_qa"],
    input_hint="需要对比的标准/文档或需要判断的元件替换问题（含标准名、型号、参数）",
    output_hint="EMC 对比结论：章节路径、参数差异表、替换判断与可视化图表说明",
)
