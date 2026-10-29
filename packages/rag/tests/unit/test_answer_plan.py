# Test values are synthetic fixtures for software behavior.
from core.answer_plan import build_answer_plan


def test_emc_test_item_comparison_plan_normalizes_alias():
    plan = build_answer_plan("对比两个标准的辐射发射 RE 限值")
    assert plan.intent == "emc_test_compare"
    assert plan.test_item == "RE"


def test_component_substitution_has_priority_over_generic_comparison():
    plan = build_answer_plan("TVS-A 可以替代 TVS-B 吗")
    assert plan.intent == "component_substitution"


def test_normal_question_is_qa_plan():
    assert build_answer_plan("这个器件的封装尺寸是多少").intent == "qa"
