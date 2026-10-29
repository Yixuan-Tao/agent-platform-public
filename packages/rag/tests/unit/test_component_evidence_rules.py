# Test values are synthetic fixtures for software behavior.
from tools.component_evidence_rules import compare_component_evidence


def _evidence(subject, field, value, *, kind="max", condition="", family="MOSFET"):
    return {"subject_id": subject, "component_family": family, "field_name": field,
            "value_raw": str(value), "value_normalized": value, "value_kind": kind,
            "unit": "V" if field == "VDS" else "mΩ" if field == "RDS(on)" else "",
            "test_condition": condition, "source_file": f"{subject}.pdf", "page": 3, "section": "Electrical"}


def test_mosfet_replacement_uses_directional_ratings_and_conditions():
    rules = [
        {"parameter": "VDS", "rule": "candidate_gte", "critical": True},
        {"parameter": "RDS(on)", "rule": "candidate_lte", "critical": True, "require_condition": True},
        {"parameter": "封装", "rule": "equal", "critical": True},
        {"parameter": "引脚映射", "rule": "equal", "critical": True},
    ]
    a = [_evidence("A", "VDS", 40), _evidence("A", "RDS(on)", 10, condition="VGS=10V;ID=10A"),
         _evidence("A", "封装", "DFN5x6", kind="enum"), _evidence("A", "引脚映射", "G-D-S", kind="enum")]
    b = [_evidence("B", "VDS", 60), _evidence("B", "RDS(on)", 8, condition="VGS=10V;ID=10A"),
         _evidence("B", "封装", "DFN5x6", kind="enum"), _evidence("B", "引脚映射", "G-D-S", kind="enum")]

    result = compare_component_evidence(a, b, "A", "B", rules)

    assert result["A_to_B"]["conclusion"] == "可替换"
    assert result["B_to_A"]["conclusion"] == "不可替换"


def test_missing_required_test_condition_is_conditional_not_replaceable():
    rules = [{"parameter": "RDS(on)", "rule": "candidate_lte", "critical": True, "require_condition": True}]
    result = compare_component_evidence(
        [_evidence("A", "RDS(on)", 10, condition="VGS=10V")],
        [_evidence("B", "RDS(on)", 8)], "A", "B", rules,
    )
    assert result["A_to_B"]["conclusion"] == "有条件替换"
