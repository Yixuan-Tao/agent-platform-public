from agent_platform.core.models import AgentManifest

MANIFEST = AgentManifest(
    name="info_collect",
    description=(
        "行业信息采集与洞察 Agent：抓取网页/RSS 资讯，过滤去噪，"
        "提炼关键洞察并生成结构化分析报告。"
    ),
    capabilities=["collect", "rss", "web", "filter", "insight", "report"],
    tools=["web_collect", "rss_fetch"],
    handoff=[],
    input_hint="需要采集的行业/竞品主题、来源范围与报告要求",
    output_hint="结构化洞察报告：来源、时间范围、关键发现与结论建议",
)
