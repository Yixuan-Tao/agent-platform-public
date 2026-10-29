"""Regression tests using newly authored synthetic documents only."""
from core.standard_registry import alias_file_fragments, detect_families, family_member_files
from tools.param_template import list_available_types, parse_xlsx_template

def test_synthetic_aliases_and_document_families():
    assert alias_file_fragments('demo alpha') == ['demoalpha']
    assert detect_families('compare demo alpha with demo beta') == ['demo_alpha', 'demo_beta']
    assert family_member_files('demo_alpha', ['Demo-Alpha-part1.txt', 'unrelated.txt']) == ['Demo-Alpha-part1.txt']
    assert detect_families('unrecognized source') == []

def test_generic_parameter_resource_is_self_contained():
    assert {'MOSFET', 'TVS'} <= set(list_available_types())
    template = parse_xlsx_template(component_type='TVS')
    assert 'V_BR' in {name for group in template.values() for name in group}
