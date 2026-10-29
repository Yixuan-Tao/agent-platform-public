"""EMC 领域工具（简化版业务逻辑）。

.. deprecated::
    平台 Agent 已改为使用 ``agent_platform.tools.builtin.emc_service`` 中的
    服务适配层（调用 RAG 的 ``/compare/*`` 完整能力）。本模块仅保留给
    离线/无 RAG 服务场景与历史测试使用，新代码请勿依赖。
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent_platform.tools.base import BaseTool


def _extract_sections(text: str) -> list[dict[str, Any]]:
    """按标题提取章节（简单版）。"""
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+(?:\.\d+)*)\s+(.+)$", line)
        if m:
            current = {"path": m.group(1), "title": m.group(2).strip(), "lines": []}
            sections.append(current)
        elif current is not None:
            current["lines"].append(line.strip())
    return sections


class EmcCompareTool(BaseTool):
    name = "emc_compare"
    description = "对比两份 EMC 标准文本：按章节路径对齐，输出差异与结论。"
    args_schema = {
        "type": "object",
        "properties": {
            "text_a": {"type": "string", "description": "标准文档 A 的文本"},
            "text_b": {"type": "string", "description": "标准文档 B 的文本"},
        },
        "required": ["text_a", "text_b"],
    }

    async def execute(self, text_a: str, text_b: str, **kwargs: Any) -> str:
        sec_a = {s["path"]: s for s in _extract_sections(text_a)}
        sec_b = {s["path"]: s for s in _extract_sections(text_b)}
        only_a = [p for p in sec_a if p not in sec_b]
        only_b = [p for p in sec_b if p not in sec_a]
        both = [p for p in sec_a if p in sec_b]
        diffs = []
        for p in both:
            if sec_a[p]["title"] != sec_b[p]["title"]:
                diffs.append(f"{p} 标题不同: A={sec_a[p]['title']} / B={sec_b[p]['title']}")
        result = {
            "章节数": {"A": len(sec_a), "B": len(sec_b)},
            "仅A有": only_a[:20],
            "仅B有": only_b[:20],
            "标题差异": diffs[:20],
            "结论": "两份标准存在差异，建议人工复核详细限值。",
        }
        return json.dumps(result, ensure_ascii=False, indent=2)


class EmcComponentCheckTool(BaseTool):
    name = "emc_component_check"
    description = "基于参数模板判断元件能否替代：逐字段比对并给出结论（含容差）。"
    args_schema = {
        "type": "object",
        "properties": {
            "template": {"type": "string", "description": "参数模板 JSON（原元件要求）"},
            "candidate": {"type": "string", "description": "候选元件参数 JSON"},
        },
        "required": ["template", "candidate"],
    }

    async def execute(self, template: str, candidate: str, **kwargs: Any) -> str:
        try:
            tpl = json.loads(template)
            cand = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return f"参数不是合法 JSON: {exc}"
        lines: list[str] = []
        for key, req in tpl.items():
            got = cand.get(key)
            if got is None:
                lines.append(f"[缺失] {key}: 候选元件没有该参数（要求 {req}）")
                continue
            if isinstance(req, dict) and "min" in req and "max" in req:
                ok = req["min"] <= float(got) <= req["max"]
                lines.append(
                    f"[{'通过' if ok else '不通过'}] {key}: 要求 {req['min']}~{req['max']}，实测 {got}"
                )
            else:
                lines.append(f"[信息] {key}: 要求 {req}，候选 {got}")
        return "\n".join(lines)


class EmcVisualizeTool(BaseTool):
    name = "emc_visualize"
    description = "把限值表/参数表渲染为 ECharts 可视化 HTML（Type_A-D 简化版），返回可直接嵌入的 HTML 片段。"
    args_schema = {
        "type": "object",
        "properties": {
            "chart_type": {"type": "string", "description": "Type_A/B/C/D"},
            "data_json": {"type": "string", "description": "图表数据 JSON"},
        },
        "required": ["chart_type", "data_json"],
    }

    async def execute(self, chart_type: str, data_json: str, **kwargs: Any) -> str:
        try:
            data = json.loads(data_json)
        except json.JSONDecodeError as exc:
            return f"数据不是合法 JSON: {exc}"
        series = data.get("series", data)
        html = (
            "<div style='padding:12px;border:1px solid #ddd;border-radius:8px'>"
            f"<b>EMC {chart_type} 可视化</b>（ECharts 占位，正式版嵌入图表库）<br/>"
            f"<pre>{json.dumps(series, ensure_ascii=False)[:2000]}</pre></div>"
        )
        return html
