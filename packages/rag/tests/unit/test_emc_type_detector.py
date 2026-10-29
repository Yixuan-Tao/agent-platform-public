# Test values are synthetic fixtures for software behavior.
"""EMC 表格类型检测器单元测试

覆盖范围：
    - detect_table_type：空/空白/None / 各类型关键词匹配 / 大小写不敏感 / 混合关键词 / 无关文本
    - detect_table_type_from_chunks：空列表 / str chunk / dict chunk / 加权投票
    - get_type_confidence：空列表 / 全部同类型 / 混合类型 / 置信度计算
"""
from __future__ import annotations

import pytest


# ============================================================
# detect_table_type 单测
# ============================================================

class TestDetectTableType:
    """detect_table_type 单测：单个 chunk 文本检测类型。"""

    # --- 边界输入 ---

    def test_empty_string(self):
        """空字符串应返回 Unknown"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type("") == "Unknown"

    def test_whitespace_string(self):
        """纯空白字符串应返回 Unknown"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type("   ") == "Unknown"
        assert detect_table_type("\t\n") == "Unknown"

    def test_none_input(self):
        """None 输入应返回 Unknown"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type(None) == "Unknown"

    # --- 各类型关键词检测 ---

    def test_type_a_emission_keyword(self):
        """包含辐射发射类关键词 → Type_A"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type("辐射发射限值线") == "Type_A"

    def test_type_a_english_keyword(self):
        """包含英文字段（Radiated Emission）→ Type_A"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type("Radiated Emission test result") == "Type_A"

    def test_type_a_re_keyword(self):
        """包含缩写 RE → Type_A"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type("RE test 30 MHz") == "Type_A"

    def test_type_b_bci_keyword(self):
        """包含 BCI 关键词 → Type_B"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type("BCI 大电流注入测试等级") == "Type_B"

    def test_type_b_immunity_keyword(self):
        """包含 Immunity 关键词 → Type_B"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type("Radiated Immunity test level") == "Type_B"

    def test_type_c_pulse_keyword(self):
        """包含脉冲类关键词 → Type_C"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type("Pulse 1 瞬态脉冲测试") == "Type_C"

    def test_type_c_transient_keyword(self):
        """包含 Transient Pulse → Type_C"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type("Transient Pulse tr/td") == "Type_C"

    def test_type_d_esd_keyword(self):
        """包含 ESD 关键词 → Type_D"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type("ESD 静电放电接触放电") == "Type_D"

    def test_type_d_discharge_keyword(self):
        """包含放电电压关键词 → Type_D"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type("接触/空气放电电压 8kV") == "Type_D"

    # --- 大小写不敏感 ---

    def test_case_insensitivity(self):
        """关键词匹配不区分大小写"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type("bci injection") == "Type_B"
        assert detect_table_type("radiated emission") == "Type_A"
        assert detect_table_type("esd test") == "Type_D"
        assert detect_table_type("pulse 1") == "Type_C"

    # --- 混合关键词：选择得分最高的类型 ---

    def test_mixed_keywords_picks_highest_score(self):
        """同时命中多个类型时，返回得分最高的类型"""
        from tools.emc_type_detector import detect_table_type

        # "Radiated Emission" 命中 Type_A，得分 3（子串+1，词边界+2）
        # "BCI" 命中 Type_B，得分 3（子串+1，词边界+2）
        # 平局时按 TYPE_KEYWORDS 定义顺序取第一个（Type_A）
        assert detect_table_type("Radiated Emission BCI") == "Type_A"

    def test_mixed_keywords_type_b_wins(self):
        """Type_B 关键词数量多于其他类型时返回 Type_B"""
        from tools.emc_type_detector import detect_table_type

        # "ESD" 命中 Type_D（子串+1，词边界+2）→ 3
        # "BCI" + "Immunity" 命中 Type_B（各子串+1，词边界+2）→ 6
        assert detect_table_type("ESD BCI Immunity") == "Type_B"

    # --- 无关文本 ---

    def test_irrelevant_text_returns_unknown(self):
        """不含任何 EMC 关键词的文本返回 Unknown"""
        from tools.emc_type_detector import detect_table_type

        assert detect_table_type("hello world this is a test") == "Unknown"
        assert detect_table_type("中文无关文本测试") == "Unknown"


# ============================================================
# detect_table_type_from_chunks 单测
# ============================================================

class TestDetectTableTypeFromChunks:
    """detect_table_type_from_chunks 单测：多 chunk 加权投票。"""

    def test_empty_list(self):
        """空列表应返回 Unknown"""
        from tools.emc_type_detector import detect_table_type_from_chunks

        assert detect_table_type_from_chunks([]) == "Unknown"

    def test_none_input(self):
        """None 输入应返回 Unknown"""
        from tools.emc_type_detector import detect_table_type_from_chunks

        assert detect_table_type_from_chunks(None) == "Unknown"

    def test_string_chunks(self):
        """chunks 为字符串列表时正常检测"""
        from tools.emc_type_detector import detect_table_type_from_chunks

        chunks = ["辐射发射测试", "RE 限值线"]
        assert detect_table_type_from_chunks(chunks) == "Type_A"

    def test_dict_chunks_content_key(self):
        """chunks 为字典列表（content 字段）时正常检测"""
        from tools.emc_type_detector import detect_table_type_from_chunks

        chunks = [{"content": "ESD 静电放电"}]
        assert detect_table_type_from_chunks(chunks) == "Type_D"

    def test_dict_chunks_text_key(self):
        """chunks 为字典列表（text 字段）时正常检测"""
        from tools.emc_type_detector import detect_table_type_from_chunks

        chunks = [{"text": "BCI 大电流注入"}]
        assert detect_table_type_from_chunks(chunks) == "Type_B"

    def test_dict_content_fallback_to_text(self):
        """字典缺少 content 时回退到 text 字段"""
        from tools.emc_type_detector import detect_table_type_from_chunks

        chunks = [{"content": "", "text": "Pulse 1 瞬态脉冲"}]
        assert detect_table_type_from_chunks(chunks) == "Type_C"

    def test_mixed_type_chunks_weighted_voting(self):
        """多个不同类型 chunk 按总分加权投票"""
        from tools.emc_type_detector import detect_table_type_from_chunks

        # Type_D（ESD）= 3 分，Type_B（BCI Immunity test level）= 9 分 → Type_B 胜出
        chunks = [
            {"content": "ESD"},                     # Type_D = 3
            {"content": "BCI Immunity test level"},  # Type_B = 9
        ]
        result = detect_table_type_from_chunks(chunks)
        assert result == "Type_B"

    def test_empty_content_skipped(self):
        """空内容的 chunk 应被跳过"""
        from tools.emc_type_detector import detect_table_type_from_chunks

        chunks = [
            {"content": ""},
            {"content": "   "},
            {"content": "辐射发射 RE"},
        ]
        assert detect_table_type_from_chunks(chunks) == "Type_A"

    def test_mixed_string_and_dict_chunks(self):
        """同时包含字符串和字典 chunk"""
        from tools.emc_type_detector import detect_table_type_from_chunks

        chunks = [
            "辐射发射 RE",
            {"content": "BCI 抗扰度"},
        ]
        # Type_A 得分更高 → Type_A
        assert detect_table_type_from_chunks(chunks) == "Type_A"


# ============================================================
# get_type_confidence 单测
# ============================================================

class TestGetTypeConfidence:
    """get_type_confidence 单测：置信度详情。"""

    def test_empty_list(self):
        """空列表返回默认字段"""
        from tools.emc_type_detector import get_type_confidence

        result = get_type_confidence([])
        assert result == {
            "type": "Unknown", "scores": {}, "total": 0,
            "confidence": 0.0, "chunk_count": 0,
        }

    def test_none_input(self):
        """None 输入返回默认字段"""
        from tools.emc_type_detector import get_type_confidence

        result = get_type_confidence(None)
        assert result == {
            "type": "Unknown", "scores": {}, "total": 0,
            "confidence": 0.0, "chunk_count": 0,
        }

    def test_all_same_type_confidence_one(self):
        """所有 chunk 均为同一类型 → confidence = 1.0"""
        from tools.emc_type_detector import get_type_confidence

        chunks = ["ESD 静电放电"]
        result = get_type_confidence(chunks)
        assert result["type"] == "Type_D"
        assert result["chunk_count"] == 1
        assert result["confidence"] == 1.0

    def test_mixed_types_confidence_less_than_one(self):
        """不同类型混合 → confidence < 1.0"""
        from tools.emc_type_detector import get_type_confidence

        chunks = ["ESD BCI Immunity"]
        result = get_type_confidence(chunks)
        assert result["type"] in ("Type_B", "Type_D")
        assert result["confidence"] < 1.0
        assert result["confidence"] > 0.0

    def test_irrelevant_text_empty_scores(self):
        """不含任何关键词 → scores 为空"""
        from tools.emc_type_detector import get_type_confidence

        chunks = ["hello world"]
        result = get_type_confidence(chunks)
        assert result["type"] == "Unknown"
        assert result["scores"] == {}
        assert result["total"] == 0
        assert result["confidence"] == 0.0
        assert result["chunk_count"] == 1

    def test_confidence_with_multiple_chunks(self):
        """多个 chunk 混合时置信度计算正确"""
        from tools.emc_type_detector import get_type_confidence

        # 2 个 Type_D + 1 个 Type_B → Type_D 应胜出
        chunks = [
            "ESD 静电放电",               # Type_D
            "接触放电 空气放电",           # Type_D
            "BCI Immunity test level",     # Type_B
        ]
        result = get_type_confidence(chunks)
        assert result["type"] == "Type_D"
        assert result["chunk_count"] == 3
        # 至少有一个其他类型的分数，confidence 应 < 1.0
        assert result["confidence"] < 1.0
        assert result["confidence"] > 0.0

    def test_scores_contain_all_types(self):
        """scores 包含所有被命中的类型"""
        from tools.emc_type_detector import get_type_confidence

        chunks = ["辐射发射 BCI ESD"]
        result = get_type_confidence(chunks)
        # 至少应包含 Type_A、Type_B、Type_D 三项
        assert "Type_A" in result["scores"]
        assert "Type_B" in result["scores"]
        assert "Type_D" in result["scores"]
        assert result["total"] > 0

    def test_dict_chunks_confidence(self):
        """字典 chunk 格式不影响置信度计算"""
        from tools.emc_type_detector import get_type_confidence

        chunks = [{"content": "辐射发射 RE"}]
        result = get_type_confidence(chunks)
        assert result["type"] == "Type_A"
        assert result["chunk_count"] == 1
        assert result["confidence"] == 1.0

    def test_empty_content_chunks_confidence(self):
        """空内容的 chunk 影响 chunk_count 但不贡献分数"""
        from tools.emc_type_detector import get_type_confidence

        chunks = [
            {"content": ""},
            {"content": "辐射发射"},
        ]
        result = get_type_confidence(chunks)
        assert result["type"] == "Type_A"
        assert result["chunk_count"] == 2
        assert result["total"] > 0
        assert result["confidence"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
