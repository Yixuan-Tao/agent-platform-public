"""语义路由：关键词表 + LLM 兜底（微软 Semantic Router 模式）。"""

from __future__ import annotations

from typing import Any

from agent_platform.core.models import ChatMessage

# 意图关键词表：Agent 名 -> 关键词列表（模块常量，便于测试）
INTENT_KEYWORDS: dict[str, list[str]] = {
    "it_support": [
        "vpn", "网络", "软件", "安装", "账号", "权限", "故障", "报修",
        "工单", "打印", "蓝屏", "it", "IT", "申请", "电脑", "办公软件",
    ],
    "hardware_qa": [
        "硬件", "参数", "型号", "datasheet", "规格", "元器件", "替代",
        "引脚", "电气", "负载", "电压", "电流",
    ],
    "emc_compare": [
        "emc", "EMC", "标准对比", "辐射", "发射", "抗扰", "esd", "ESD",
        "测试项目", "限值", "频谱", "传导",
    ],
    "info_collect": [
        "行业", "竞品", "市场", "报告", "调研", "资讯", "新闻", "收集",
        "分析", "动态", "趋势",
    ],
}


class SemanticRouter:
    """按关键词命中数选 Agent；置信度低时由调用方决定 LLM 兜底。"""

    def route(self, text: str) -> tuple[str | None, float]:
        best_agent: str | None = None
        best_hits = 0
        for agent, keywords in INTENT_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits > best_hits:
                best_hits = hits
                best_agent = agent
        if best_agent is None or best_hits <= 0:
            return None, 0.0
        confidence = min(1.0, best_hits * 0.5)
        return best_agent, confidence

    async def ensure_llm_route(
        self, text: str, llm: Any, agents: list[str], agent_descriptions: dict[str, str] | None = None
    ) -> str | None:
        """关键词未命中或置信度低时，用 LLM 判定应交给哪个 Agent。"""
        if not agents:
            return None
        agent_descriptions = agent_descriptions or {}
        lines = "\n".join(
            f"- {a}: {agent_descriptions.get(a, '')}" for a in agents
        )
        prompt = (
            "你是任务分发器。根据用户请求选择最合适的 Agent，只输出 Agent 名称（不要任何其他内容）。\n"
            "仅当请求与平台业务完全无关（如天气、闲聊）时输出 unsupported，不得强行路由；"
            "行业动态/资讯采集属 info_collect 能力范围，应正常路由。\n"
            f"可选 Agent：\n{lines}\n\n用户请求：{text}\nAgent 名称："
        )
        try:
            answer = await llm.chat(
                [ChatMessage(role="user", content=prompt)], temperature=0.0
            )
        except Exception:  # noqa: BLE001
            return None
        answer = (answer or "").strip()
        for agent in agents:
            if agent in answer:
                return agent
        return None
