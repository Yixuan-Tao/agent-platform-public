# Test values are synthetic fixtures for software behavior.
"""
元器件替换判断工具 ComponentCheckTool 的纯逻辑单元测试

覆盖 7 个纯静态/实例方法，不涉及 LLM / RAG 外部调用。
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from tools.component_check import ComponentCheckTool


class TestCriticalCategories:
    """_CRITICAL_CATEGORIES 类常量测试"""

    def test_critical_categories_content(self):
        """电气特性类别应被视为关键参数类别"""
        assert ComponentCheckTool._CRITICAL_CATEGORIES == {"electrical"}


class TestFlattenTemplate:
    """_flatten_template 静态方法测试"""

    def test_normal(self):
        """正常嵌套模板应正确扁平化并注入 category 字段"""
        template = {
            "electrical": {"VBR": {"unit": "V"}, "IR": {"unit": "uA"}},
            "package": {"type": {"value": "SOD-323"}},
        }
        result = ComponentCheckTool._flatten_template(template)

        # 验证返回的是 (name, spec) 元组列表
        assert isinstance(result, list)
        assert len(result) == 3

        # 验证每个 spec 都注入了 category 字段
        names = {item[0] for item in result}
        assert names == {"VBR", "IR", "type"}

        # 验证 category 注入
        for name, spec in result:
            if name in ("VBR", "IR"):
                assert spec["category"] == "electrical"
            else:
                assert spec["category"] == "package"

    def test_empty_dict(self):
        """空模板应返回空列表"""
        assert ComponentCheckTool._flatten_template({}) == []

    def test_empty_category(self):
        """某类别下没有参数时，该类别被跳过"""
        template = {"electrical": {}, "package": {"type": {"value": "SOD-323"}}}
        result = ComponentCheckTool._flatten_template(template)
        assert len(result) == 1
        assert result[0][0] == "type"

    def test_non_dict_params_skipped(self):
        """参数值不是 dict 时 → spec 降级为空 dict，仍参与扁平化"""
        template = {
            "electrical": {"VBR": "27V", "IR": {"unit": "uA"}},
        }
        result = ComponentCheckTool._flatten_template(template)
        # "VBR": "27V" 不是 dict → 源码中将其降级为 {}，而非跳过
        assert len(result) == 2
        names = {item[0] for item in result}
        assert names == {"VBR", "IR"}


class TestKeywordRankChunks:
    """_keyword_rank_chunks 静态方法测试"""

    @pytest.fixture
    def chunks(self):
        """构造一组模拟 chunk 文本"""
        return [
            {"idx": 0, "text": "VBR breakdown voltage is 27V VBR value"},
            {"idx": 1, "text": "Reverse standoff voltage VBR typical 27V"},
            {"idx": 2, "text": "Package type is SOD-323"},
            {"idx": 3, "text": "VBR VBR VBR maximum rating"},
            {"idx": 4, "text": "Some other content about current"},
        ]

    def test_rank_by_keyword_frequency(self, chunks):
        """关键词频率高的 chunk 应被选中，结果按原始输入顺序返回"""
        result = ComponentCheckTool._keyword_rank_chunks(chunks, "VBR", top_k=5)
        # 含 VBR 的 chunk: idx0(2次), idx1(2次), idx3(3次) → 共 3 个
        assert len(result) == 3
        # 结果按原始 chunk 顺序返回（而非按分数排序）
        assert [c["idx"] for c in result] == [0, 1, 3]

    def test_no_match_returns_empty(self, chunks):
        """无匹配关键词时应返回空列表"""
        result = ComponentCheckTool._keyword_rank_chunks(chunks, "ZENER", top_k=5)
        assert result == []

    def test_top_k_limiting(self, chunks):
        """top_k 参数应限制返回数量"""
        result = ComponentCheckTool._keyword_rank_chunks(chunks, "VBR", top_k=2)
        assert len(result) <= 2

    def test_empty_chunks_returns_empty(self):
        """空 chunks 列表应返回空列表"""
        result = ComponentCheckTool._keyword_rank_chunks([], "VBR", top_k=5)
        assert result == []

    def test_chunk_with_empty_text_skipped(self):
        """text 字段为空的 chunk 应被跳过"""
        chunks = [
            {"idx": 0, "text": "VBR is here"},
            {"idx": 1, "text": ""},
        ]
        result = ComponentCheckTool._keyword_rank_chunks(chunks, "VBR", top_k=5)
        assert len(result) == 1
        assert result[0]["idx"] == 0


class TestParseValueWithUnit:
    """_parse_value_with_unit 静态方法测试"""

    def test_basic_voltage(self):
        """'27V' 应解析为 (27.0, 'V')"""
        result = ComponentCheckTool._parse_value_with_unit("27V")
        assert result is not None
        val, unit = result
        assert math.isclose(val, 27.0)
        assert unit == "V"

    def test_microamp(self):
        """'1.0uA' 应解析为 (1.0, 'uA')"""
        result = ComponentCheckTool._parse_value_with_unit("1.0uA")
        assert result is not None
        val, unit = result
        assert math.isclose(val, 1.0)
        assert unit == "uA"

    def test_range_with_tilde(self):
        """'26~30V' 应解析为中间值 (28.0, 'V')"""
        result = ComponentCheckTool._parse_value_with_unit("26~30V")
        assert result is not None
        val, unit = result
        assert math.isclose(val, 28.0)
        assert unit == "V"

    def test_range_with_hyphen(self):
        """'26-30V' 应解析为中间值 (28.0, 'V')"""
        result = ComponentCheckTool._parse_value_with_unit("26-30V")
        assert result is not None
        val, unit = result
        assert math.isclose(val, 28.0)
        assert unit == "V"

    def test_percent_only_returns_value(self):
        """'±5%' 解析为 (5.0, '%')：± 被当作前缀符号，% 被当作单位"""
        result = ComponentCheckTool._parse_value_with_unit("±5%")
        assert result is not None
        val, unit = result
        assert val == 5.0
        assert unit == "%"

    def test_pure_number(self):
        """'3.3' 纯数值应解析为 (3.3, '')"""
        result = ComponentCheckTool._parse_value_with_unit("3.3")
        assert result is not None
        val, unit = result
        assert math.isclose(val, 3.3)
        assert unit == ""

    def test_empty_string(self):
        """空字符串应返回 None"""
        assert ComponentCheckTool._parse_value_with_unit("") is None

    def test_invalid_string(self):
        """无法解析的字符串应返回 None"""
        assert ComponentCheckTool._parse_value_with_unit("abc") is None

    def test_none_input(self):
        """None 输入应返回 None"""
        assert ComponentCheckTool._parse_value_with_unit(None) is None

    def test_signed_value(self):
        """带有 ± 符号的数值应解析为纯数值"""
        result = ComponentCheckTool._parse_value_with_unit("±1V")
        assert result is not None
        val, unit = result
        assert math.isclose(val, 1.0)
        assert unit == "V"


class TestCheckTolerance:
    """_check_tolerance 静态方法测试"""

    def test_percent_within_tolerance(self):
        """±5% 容差内：100 vs 103 偏差 3%，应在容差内"""
        assert ComponentCheckTool._check_tolerance(100, 103, "±5%") is True

    def test_percent_out_of_tolerance(self):
        """±5% 超容差：100 vs 110 偏差 10%，超出容差"""
        assert ComponentCheckTool._check_tolerance(100, 110, "±5%") is False

    def test_absolute_within_tolerance(self):
        """±1V 容差内：5 vs 5.5，偏差 0.5V，在容差内"""
        assert ComponentCheckTool._check_tolerance(5, 5.5, "±1V") is True

    def test_absolute_out_of_tolerance(self):
        """±1V 超容差：5 vs 7，偏差 2V，超出容差"""
        assert ComponentCheckTool._check_tolerance(5, 7, "±1V") is False

    def test_invalid_format_without_plusminus(self):
        """无 ± 前缀的容差格式应返回 False"""
        assert ComponentCheckTool._check_tolerance(100, 103, "5%") is False

    def test_tolerance_on_boundary(self):
        """恰好等于容差边界值应视为在容差内"""
        # ±5% of 100 = 5, val_b should be within [95, 105]
        assert ComponentCheckTool._check_tolerance(100, 105, "±5%") is True
        assert ComponentCheckTool._check_tolerance(100, 95, "±5%") is True

    def test_percent_negative_a(self):
        """负数值的百分比容差计算"""
        assert ComponentCheckTool._check_tolerance(-100, -103, "±5%") is True
        assert ComponentCheckTool._check_tolerance(-100, -110, "±5%") is False


class TestTryUnitConvert:
    """_try_unit_convert 静态方法测试"""

    def test_volt_to_millivolt(self):
        """V → mV: 1V = 1000mV"""
        result = ComponentCheckTool._try_unit_convert(1.0, "V", "mV")
        assert result is not None
        assert math.isclose(result, 1000.0)

    def test_amp_to_microamp(self):
        """A → uA: 0.001A = 1000uA"""
        result = ComponentCheckTool._try_unit_convert(0.001, "A", "uA")
        assert result is not None
        assert math.isclose(result, 1000.0)

    def test_incompatible_units_same_factor(self):
        """V → A：代码不校验量纲兼容性，V 和 A 都在 unit_map 中且 scale=1.0 → 返回 1.0"""
        # 注意：源码不校验量纲兼容性，只要两者都在 unit_map 中就做换算
        result = ComponentCheckTool._try_unit_convert(1.0, "V", "A")
        assert result is not None
        assert math.isclose(result, 1.0)

    def test_same_unit(self):
        """相同单位应返回原值"""
        result = ComponentCheckTool._try_unit_convert(5.0, "V", "V")
        assert result is not None
        assert math.isclose(result, 5.0)

    def test_ohm_to_kohm(self):
        """Ω → kΩ: 1000Ω = 1kΩ"""
        result = ComponentCheckTool._try_unit_convert(1000.0, "OHM", "KOHM")
        assert result is not None
        assert math.isclose(result, 1.0)

    def test_farad_to_nanofarad(self):
        """F → nF: 0.000001F = 1000nF"""
        result = ComponentCheckTool._try_unit_convert(0.000001, "F", "NF")
        assert result is not None
        assert math.isclose(result, 1000.0)


class TestCompareParams:
    """_compare_params 实例方法测试（self 不会被用到，传入 None 即可）"""

    def setup_tool(self):
        """创建一个 ComponentCheckTool 实例，传入 mock 的 rag_service 和 generator"""
        mock_rag = MagicMock()
        mock_gen = MagicMock()
        return ComponentCheckTool(mock_rag, mock_gen)

    def test_both_none(self):
        """两份文档都找不到参数 → match=False，note 提示两份文档均未找到"""
        tool = self.setup_tool()
        result = tool._compare_params(None, None, {}, strict_mode=True)
        assert result["match"] is False
        assert "两份文档均未找到" in result["note"]

    def test_only_doc_a(self):
        """仅文档 A 有值 → match=False，note 提示仅在文档 A 中找到"""
        tool = self.setup_tool()
        result = tool._compare_params("27V", None, {}, strict_mode=True)
        assert result["match"] is False
        assert "仅在文档 A 中找到" in result["note"]

    def test_only_doc_b(self):
        """仅文档 B 有值 → match=False，note 提示仅在文档 B 中找到"""
        tool = self.setup_tool()
        result = tool._compare_params(None, "27V", {}, strict_mode=True)
        assert result["match"] is False
        assert "仅在文档 B 中找到" in result["note"]

    def test_exact_numeric_match(self):
        """数值完全相等 → match=True, within_tolerance=True"""
        tool = self.setup_tool()
        result = tool._compare_params("27V", "27V", {}, strict_mode=True)
        assert result["match"] is True
        assert result["within_tolerance"] is True
        assert "完全匹配" in result["note"]

    def test_tolerance_match(self):
        """容差内匹配 → match=True, within_tolerance=True"""
        tool = self.setup_tool()
        spec = {"tolerance": "±5%"}
        result = tool._compare_params("100V", "103V", spec, strict_mode=True)
        assert result["match"] is True
        assert result["within_tolerance"] is True
        assert "容差内匹配" in result["note"]

    def test_beyond_tolerance_strict_mode(self):
        """严格模式下超容差 → match=False, within_tolerance=False"""
        tool = self.setup_tool()
        spec = {"tolerance": "±5%"}
        result = tool._compare_params("100V", "110V", spec, strict_mode=True)
        assert result["match"] is False
        assert result["within_tolerance"] is False
        assert "数值不匹配" in result["note"]

    def test_beyond_tolerance_non_strict_mode(self):
        """非严格模式下偏差 < 20% → match=True, within_tolerance=False"""
        tool = self.setup_tool()
        # 100 vs 110 偏差约 9.1% < 20%
        result = tool._compare_params("100V", "110V", {}, strict_mode=False)
        assert result["match"] is True
        assert result["within_tolerance"] is False
        assert "非严格模式" in result["note"]

    def test_beyond_tolerance_non_strict_mode_large_diff(self):
        """非严格模式下偏差 >= 20% → match=False"""
        tool = self.setup_tool()
        # 100 vs 130 偏差 23.1% > 20%
        result = tool._compare_params("100V", "130V", {}, strict_mode=False)
        assert result["match"] is False
        assert result["within_tolerance"] is False

    def test_string_value_match(self):
        """字符串值完全匹配（忽略大小写）→ match=True"""
        tool = self.setup_tool()
        result = tool._compare_params("SOD-323", "sod-323", {}, strict_mode=True)
        assert result["match"] is True
        assert result["within_tolerance"] is True
        assert "完全匹配" in result["note"]

    def test_string_value_mismatch(self):
        """字符串值不匹配 → match=False"""
        tool = self.setup_tool()
        result = tool._compare_params("SOT-23", "SOD-323", {}, strict_mode=True)
        assert result["match"] is False
        assert "字符串不匹配" in result["note"]

    def test_unit_conversion_before_compare(self):
        """单位不同但可换算时，应换算后对比"""
        tool = self.setup_tool()
        # 1V vs 1000mV → 相等
        result = tool._compare_params("1V", "1000mV", {}, strict_mode=True)
        assert result["match"] is True
        assert result["within_tolerance"] is True

    def test_unit_mismatch_same_scale(self):
        """单位不同但换算后数值相等 → match=True"""
        tool = self.setup_tool()
        # 注意：源码不校验量纲兼容性；V 和 A 在 unit_map 中 scale 均为 1.0 → 1V "换算" 为 1A → 相等
        result = tool._compare_params("1V", "1A", {}, strict_mode=True)
        assert result["match"] is True
        assert result["within_tolerance"] is True


class TestGenerateConclusion:
    """_generate_conclusion 实例方法测试"""

    def setup_tool(self):
        """创建一个 ComponentCheckTool 实例"""
        mock_rag = MagicMock()
        mock_gen = MagicMock()
        return ComponentCheckTool(mock_rag, mock_gen)

    def test_all_matched_within_tolerance(self):
        """所有参数完全匹配/容差内 → '可替换'"""
        tool = self.setup_tool()
        comparisons = [
            {"param": "VBR", "category": "electrical", "match": True, "within_tolerance": True,
             "doc_a_value": "27V", "doc_b_value": "27V"},
            {"param": "IR", "category": "electrical", "match": True, "within_tolerance": True,
             "doc_a_value": "1uA", "doc_b_value": "1uA"},
            {"param": "type", "category": "package", "match": True, "within_tolerance": True,
             "doc_a_value": "SOD-323", "doc_b_value": "SOD-323"},
        ]
        conclusion, detail = tool._generate_conclusion(comparisons, total_params=3)
        assert conclusion == "可替换"
        assert "所有参数完全匹配" in detail

    def test_critical_category_unmatched(self):
        """关键类别（electrical）有参数不匹配 → '不可替换'"""
        tool = self.setup_tool()
        comparisons = [
            {"param": "VBR", "category": "electrical", "match": False, "within_tolerance": False,
             "doc_a_value": "27V", "doc_b_value": "30V"},
            {"param": "type", "category": "package", "match": True, "within_tolerance": True,
             "doc_a_value": "SOD-323", "doc_b_value": "SOD-323"},
        ]
        conclusion, detail = tool._generate_conclusion(comparisons, total_params=2)
        assert conclusion == "不可替换"
        assert "关键参数" in detail
        assert "VBR" in detail

    def test_mixed_non_critical(self):
        """有参数不匹配但非关键类别 → '有条件替换'"""
        tool = self.setup_tool()
        comparisons = [
            {"param": "VBR", "category": "electrical", "match": True, "within_tolerance": True,
             "doc_a_value": "27V", "doc_b_value": "27V"},
            {"param": "type", "category": "package", "match": False, "within_tolerance": False,
             "doc_a_value": "SOT-23", "doc_b_value": "SOD-323"},
        ]
        conclusion, detail = tool._generate_conclusion(comparisons, total_params=2)
        assert conclusion == "有条件替换"
        assert "有条件替换" in detail or "有条件替换" in conclusion

    def test_empty_comparisons(self):
        """空对比列表 → '不可替换'"""
        tool = self.setup_tool()
        conclusion, detail = tool._generate_conclusion([], total_params=0)
        assert conclusion == "不可替换"
        assert "参数模板为空" in detail

    def test_some_params_missing_in_docs(self):
        """部分参数在文档中缺失（非关键类别）→ 有条件替换"""
        tool = self.setup_tool()
        comparisons = [
            {"param": "VBR", "category": "electrical", "match": True, "within_tolerance": True,
             "doc_a_value": "27V", "doc_b_value": "27V"},
            {"param": "type", "category": "package", "match": False, "within_tolerance": False,
             "doc_a_value": "SOT-23", "doc_b_value": None},
        ]
        conclusion, detail = tool._generate_conclusion(comparisons, total_params=2)
        assert conclusion == "有条件替换"
        assert "无法在两份文档中都找到" in detail
