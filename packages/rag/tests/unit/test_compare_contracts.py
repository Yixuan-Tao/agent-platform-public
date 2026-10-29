# Test values are synthetic fixtures for software behavior.
from api.schemas import ComponentCompareRequest, EmcCompareRequest


def test_emc_request_accepts_test_item_selector():
    request = EmcCompareRequest(query="比较 BCI", doc_a="a", doc_b="b", comparison_mode="test_item", test_item="BCI", section_hint="5.3")
    assert request.test_item == "BCI"


def test_component_request_accepts_optional_rule_override():
    request = ComponentCompareRequest(doc_a="a", doc_b="b", params_template={}, criteria_override={"VDS": {"required": True}})
    assert request.criteria_override["VDS"]["required"] is True
