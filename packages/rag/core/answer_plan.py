"""同步与流式回答共用的中文意图编排。"""
from __future__ import annotations

from dataclasses import dataclass
import re


_COMPARE = re.compile(r"对比|比较|区别|差异|不同|限值", re.IGNORECASE)
_SUBSTITUTION = re.compile(r"替换|替代|互换|代换|能换|可以换", re.IGNORECASE)
_EMC = re.compile(r"EMC|电磁兼容|标准|BCI|ESD|RE|CE|RI|CI|辐射|传导", re.IGNORECASE)
_TEST_ITEM_ALIASES = {
    "BCI": ("BCI", "大电流注入"),
    "ESD": ("ESD", "静电放电"),
    "RE": ("RE", "辐射发射"),
    "CE": ("CE", "传导发射"),
    "RI": ("RI", "辐射抗扰"),
    "CI": ("CI", "传导抗扰"),
}


@dataclass(frozen=True)
class AnswerPlan:
    intent: str
    query: str
    test_item: str = ""


def build_answer_plan(query: str) -> AnswerPlan:
    text = (query or "").strip()
    if _SUBSTITUTION.search(text):
        return AnswerPlan("component_substitution", text)
    if _COMPARE.search(text) and _EMC.search(text):
        for code, aliases in _TEST_ITEM_ALIASES.items():
            if any(alias.lower() in text.lower() for alias in aliases):
                return AnswerPlan("emc_test_compare", text, code)
        return AnswerPlan("emc_full_compare", text)
    return AnswerPlan("qa", text)
