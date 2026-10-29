from agent_platform.core.models import AgentManifest

MANIFEST = AgentManifest(
    name="hardware_qa",
    description=(
        "硬件工程问答专家：基于知识库回答元器件/型号/参数/替代问题，"
        "结合数据可视化工具呈现结果，支持反馈学习。"
    ),
    capabilities=["hardware", "qa", "parameter", "visualization"],
    tools=["rag_search", "emc_visualize"],
    handoff=["emc_compare"],
    input_hint="需要从知识库查询的硬件/元器件问题（型号、参数、替代、应用）",
    output_hint="基于知识库的答案：参数数值、来源引用、必要时附可视化结论",
)
