# Test values are synthetic fixtures for software behavior.
from tools.evidence_compare import compare_emc_evidence


def _item(subject, field, value, test_item="RE"):
    return {
        "subject_id": subject, "field_name": field, "value_raw": value,
        "test_item": test_item, "source_file": f"{subject}.pdf", "page": 10, "section": "6.2",
    }


def test_emc_test_item_compare_returns_fixed_matrix_and_citations():
    result = compare_emc_evidence(
        [_item("A", "频段", "30 MHz–1 GHz"), _item("A", "等级/限值", "Class 3")],
        [_item("B", "频段", "30 MHz–1 GHz"), _item("B", "等级/限值", "Class 4")],
        "A", "B", test_item="RE",
    )
    assert result["answer_status"] == "ok"
    assert result["coverage"]["required"] == 8
    assert next(row for row in result["comparison_matrix"] if row["field"] == "等级/限值")["status"] == "different"
    assert len(result["citations"]) == 4


def test_emc_test_item_missing_side_is_insufficient_evidence():
    result = compare_emc_evidence([_item("A", "频段", "30 MHz–1 GHz")], [], "A", "B", test_item="RE")
    assert result["answer_status"] == "insufficient_evidence"
    assert result["coverage"]["found"] == 1
