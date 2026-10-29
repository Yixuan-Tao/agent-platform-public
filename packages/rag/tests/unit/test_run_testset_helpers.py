# Test values are synthetic fixtures for software behavior.
"""评测运行器辅助函数测试：关键字变体命中与归一化。"""

from importlib import util
from pathlib import Path

_RUNNER = Path(__file__).resolve().parents[2] / "evaluation" / "run_testset.py"
spec = util.spec_from_file_location("run_testset_helpers", _RUNNER)
module = util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

normalize = module._normalize
expand_keywords = module._expand_keywords
keyword_hit = module._keyword_hit


def test_normalize_unifies_units_and_case():
    assert normalize("RDS(on) 8/20μs V_BR") == "rdson820usvbr"
    assert normalize("８／２０ｕｓ") == "820us"


def test_keyword_variants_expand_from_global_table():
    expanded = expand_keywords({"expected_keywords": ["BCI", "V_BR"]})
    assert "bci" in expanded
    assert "大电流注入" in expanded
    assert "bulkcurrentinjection" in expanded
    assert "vbr" in expanded
    assert "击穿电压" in expanded


def test_keyword_variants_expand_from_case_level():
    expanded = expand_keywords({
        "expected_keywords": ["PPK"],
        "expected_keyword_variants": ["峰值脉冲功率", "Peak Pulse Power"],
    })
    assert "峰值脉冲功率" in expanded
    assert "peakpulsepower" in expanded


def test_keyword_hit_accepts_equivalent_variants():
    expanded = expand_keywords({"expected_keywords": ["8/20us"]})
    assert keyword_hit("波形为 8/20μs 的标准浪涌", expanded) is True
    assert keyword_hit("波形采用 8 20 us", expanded) is True


def test_keyword_hit_accepts_chinese_english_mix():
    expanded = expand_keywords({"expected_keywords": ["BCI"]})
    assert keyword_hit("大电流注入（Bulk Current Injection）测试", expanded) is True


def test_resolve_doc_id_queries_real_metadata():
    """doc_id 必须从 Chroma metadata 取真实 hex 值，而非文件名前缀推导。"""
    class FakeColl:
        def get(self, **kwargs):
            return {
                "ids": ["c1"],
                "metadatas": [{"doc_id": "b1779587de5a", "file_name": "DEMO24-B[SOD-323]_fixture.pdf"}],
            }

    class FakeChroma:
        _doc_collection = FakeColl()

        def list_documents(self):
            return [{"file_name": "DEMO24-B[SOD-323]_fixture.pdf", "chunk_count": 20}]

    doc_id = module._resolve_doc_id(FakeChroma(), "DEMO24-B")
    assert doc_id == "b1779587de5a"


def test_resolve_doc_id_returns_empty_when_missing():
    class FakeColl:
        def get(self, **kwargs):
            return {"ids": [], "metadatas": []}

    class FakeChroma:
        _doc_collection = FakeColl()

        def list_documents(self):
            return [{"file_name": "other.pdf", "chunk_count": 1}]

    assert module._resolve_doc_id(FakeChroma(), "DEMO24-B") == ""
