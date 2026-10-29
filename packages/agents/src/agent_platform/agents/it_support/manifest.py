from agent_platform.core.models import AgentManifest

MANIFEST = AgentManifest(
    name="it_support",
    description=(
        "企业 IT 支持专家：解答 IT FAQ、故障报修（Incident）、服务请求（Service Request），"
        "自动采集信息并创建 Jira 工单，复杂问题转人工。"
    ),
    capabilities=["faq", "incident", "service_request", "ticket", "human_escalation"],
    tools=["rag_search", "jira_create_ticket", "escalate_human"],
    handoff=["hardware_qa"],
    input_hint="需要 IT 支持处理的问题描述（故障现象、报错、账号、设备信息等）",
    output_hint="IT 支持的处理结论：故障类型、排查步骤、工单信息或人工升级建议",
)
