# Test values are synthetic fixtures for software behavior.
"""safe_formula 安全算术表达式求值器单元测试。"""

from __future__ import annotations

import math

import pytest

from tools.safe_formula import FormulaError, evaluate_formula, sample_formula


class TestEvaluateFormula:
    """evaluate_formula 核心功能测试。"""

    # ── 基本算术运算 ──────────────────────────────────────────

    def test_addition(self):
        """加法运算应正确计算。"""
        assert evaluate_formula("1 + 2", f=0) == 3.0

    def test_subtraction(self):
        """减法运算应正确计算。"""
        assert evaluate_formula("5 - 3", f=0) == 2.0

    def test_multiplication(self):
        """乘法运算应正确计算。"""
        assert evaluate_formula("4 * 1.5", f=0) == 6.0

    def test_division(self):
        """除法运算应正确计算。"""
        assert evaluate_formula("10 / 4", f=0) == 2.5

    def test_power(self):
        """幂运算（**）应正确计算。"""
        assert evaluate_formula("2 ** 3", f=0) == 8.0

    def test_power_with_caret(self):
        """脱字符（^）应被转换为幂运算。"""
        assert evaluate_formula("2 ^ 3", f=0) == 8.0

    def test_multiplication_with_times_sign(self):
        """乘号（×）应被识别为乘法。"""
        assert evaluate_formula("3 × 4", f=0) == 12.0

    def test_unary_minus(self):
        """一元负号应正确求值。"""
        assert evaluate_formula("-5", f=0) == -5.0

    def test_unary_plus(self):
        """一元正号应正确求值。"""
        assert evaluate_formula("+3", f=0) == 3.0

    def test_operator_precedence(self):
        """运算符优先级（先乘除后加减）应正确。"""
        assert evaluate_formula("1 + 2 * 3", f=0) == 7.0

    def test_parentheses(self):
        """括号应改变运算优先级。"""
        assert evaluate_formula("(1 + 2) * 3", f=0) == 9.0

    # ── 变量 f ───────────────────────────────────────────────

    def test_variable_f(self):
        """变量 f 应被替换为给定频率值。"""
        assert evaluate_formula("f", f=100) == 100.0

    def test_variable_f_in_expression(self):
        """表达式中的变量 f 应正确参与运算。"""
        assert evaluate_formula("f * 2", f=50) == 100.0

    def test_variable_f_in_division(self):
        """含变量 f 的除法应正确计算。"""
        assert evaluate_formula("1000 / f", f=10) == 100.0

    def test_f_is_negative(self):
        """f 为负值时应正确运算。"""
        assert evaluate_formula("f + 5", f=-3) == 2.0

    # ── 内置函数 ─────────────────────────────────────────────

    def test_abs_function(self):
        """abs 函数应返回绝对值。"""
        assert evaluate_formula("abs(-5)", f=0) == 5.0

    def test_cos_function(self):
        """cos 函数应返回余弦值。"""
        assert evaluate_formula("cos(0)", f=0) == pytest.approx(1.0)

    def test_exp_function(self):
        """exp 函数应返回 e 的指数值。"""
        assert evaluate_formula("exp(0)", f=0) == pytest.approx(1.0)

    def test_log_function(self):
        """log 函数应以 10 为底计算对数。"""
        assert evaluate_formula("log(100)", f=0) == pytest.approx(2.0)

    def test_log10_function(self):
        """log10 函数应以 10 为底计算对数（与 log 一致）。"""
        assert evaluate_formula("log10(1000)", f=0) == pytest.approx(3.0)

    def test_max_function(self):
        """max 函数应返回最大值。"""
        assert evaluate_formula("max(1, 5, 3)", f=0) == 5.0

    def test_min_function(self):
        """min 函数应返回最小值。"""
        assert evaluate_formula("min(1, 5, 3)", f=0) == 1.0

    def test_sin_function(self):
        """sin 函数应返回正弦值。"""
        assert evaluate_formula("sin(0)", f=0) == pytest.approx(0.0)

    def test_sqrt_function(self):
        """sqrt 函数应返回平方根。"""
        assert evaluate_formula("sqrt(9)", f=0) == 3.0

    def test_tan_function(self):
        """tan 函数应返回正切值。"""
        assert evaluate_formula("tan(0)", f=0) == pytest.approx(0.0)

    # ── 嵌套表达式 ───────────────────────────────────────────

    def test_nested_functions(self):
        """函数嵌套调用应正确求值。"""
        assert evaluate_formula("sqrt(abs(-16))", f=0) == 4.0

    def test_function_with_variable(self):
        """函数调用中传入变量 f 应正确求值。"""
        assert evaluate_formula("sqrt(f)", f=9) == 3.0

    def test_complex_nested_expression(self):
        """复杂嵌套表达式应正确求值。"""
        assert evaluate_formula("abs(f - 100) / 2 + 1", f=80) == 11.0

    def test_nested_power(self):
        """多层幂运算应正确求值。"""
        assert evaluate_formula("2 ** (3 ** 2)", f=0) == 512.0

    def test_negated_function_result(self):
        """函数结果取负应正确。"""
        assert evaluate_formula("-sqrt(9)", f=0) == -3.0

    # ── 安全/安全测试 ────────────────────────────────────────

    def test_rejects___import__(self):
        """应拒绝 __import__ 调用，防止动态导入。"""
        with pytest.raises(FormulaError):
            evaluate_formula("__import__('os')", f=0)

    def test_rejects_eval(self):
        """应拒绝 eval 调用。"""
        with pytest.raises(FormulaError):
            evaluate_formula("eval('1+1')", f=0)

    def test_rejects_attribute_access(self):
        """应拒绝属性访问（防止通过 getattr 绕过沙箱）。"""
        with pytest.raises(FormulaError):
            evaluate_formula("(1).__class__", f=0)

    def test_rejects_arbitrary_function(self):
        """应拒绝未在白名单中的函数调用。"""
        with pytest.raises(FormulaError):
            evaluate_formula("print('hi')", f=0)

    def test_rejects_unknown_function(self):
        """应拒绝未定义的函数名称。"""
        with pytest.raises(FormulaError):
            evaluate_formula("unknown_func(1)", f=0)

    def test_rejects_list_literal(self):
        """应拒绝列表字面量表达式。"""
        with pytest.raises(FormulaError):
            evaluate_formula("[1, 2, 3]", f=0)

    def test_rejects_dict_literal(self):
        """应拒绝字典字面量表达式。"""
        with pytest.raises(FormulaError):
            evaluate_formula("{'a': 1}", f=0)

    def test_rejects_lambda(self):
        """应拒绝 lambda 表达式。"""
        with pytest.raises(FormulaError):
            evaluate_formula("lambda x: x", f=0)

    # ── 边界 / 异常输入 ──────────────────────────────────────

    def test_empty_string(self):
        """空字符串应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            evaluate_formula("", f=1)

    def test_whitespace_only(self):
        """仅含空白字符的字符串应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            evaluate_formula("   ", f=1)

    def test_non_string_input(self):
        """非字符串输入应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            evaluate_formula(123, f=1)  # type: ignore[arg-type]

    def test_f_is_zero(self):
        """f=0 时表达式应正确求值（避免除零异常）。"""
        assert evaluate_formula("f + 1", f=0) == 1.0

    def test_division_by_zero(self):
        """除以零应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            evaluate_formula("1 / 0", f=0)

    def test_sqrt_of_negative(self):
        """对负数开平方应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            evaluate_formula("sqrt(-1)", f=0)

    def test_log_of_non_positive(self):
        """对非正数取对数应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            evaluate_formula("log(0)", f=0)

    def test_invalid_syntax(self):
        """无效语法应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            evaluate_formula("1 +", f=0)

    def test_unknown_variable(self):
        """使用未定义变量（非 f）应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            evaluate_formula("x + 1", f=0)

    def test_non_numeric_constant(self):
        """字符串常量应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            evaluate_formula("'hello'", f=0)

    def test_boolean_constant(self):
        """布尔常量应抛出 FormulaError（不允许非数值常量）。"""
        with pytest.raises(FormulaError):
            evaluate_formula("True", f=0)

    def test_unsupported_operator(self):
        """不支持的运算（如取模）应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            evaluate_formula("5 % 2", f=0)

    def test_infinite_result(self):
        """结果为无穷大时应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            evaluate_formula("1e999", f=0)

    def test_negative_f_in_division(self):
        """f 为负值时除法应正确（不应在 evaluate 层面报错）。"""
        assert evaluate_formula("100 / f", f=-5) == -20.0

    def test_f_negative_in_sqrt(self):
        """f 为负值传入 sqrt 应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            evaluate_formula("sqrt(f)", f=-4)


class TestSampleFormula:
    """sample_formula 采样功能测试。"""

    def test_valid_range(self):
        """有效频率范围应返回正确的采样点数和频率值。"""
        points = sample_formula("f", start=1, end=100, steps=3)
        assert len(points) == 4  # steps + 1
        # 对数采样：应为 [1, 10, ~46.4, 100] 近似值
        assert points[0][0] == pytest.approx(1.0)
        assert points[-1][0] == pytest.approx(100.0)

    def test_start_greater_than_end(self):
        """start > end 应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            sample_formula("f", start=10, end=1, steps=5)

    def test_start_equals_end(self):
        """start == end 应返回单一点。"""
        points = sample_formula("f", start=5, end=5, steps=3)
        assert len(points) == 4
        for freq, _ in points:
            assert freq == pytest.approx(5.0)

    def test_steps_zero(self):
        """steps < 1 应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            sample_formula("f", start=1, end=100, steps=0)

    def test_steps_negative(self):
        """steps 为负数时应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            sample_formula("f", start=1, end=100, steps=-5)

    def test_steps_one(self):
        """steps=1 应返回 2 个点（端点）。"""
        points = sample_formula("f", start=1, end=10, steps=1)
        assert len(points) == 2
        assert points[0][0] == pytest.approx(1.0)
        assert points[1][0] == pytest.approx(10.0)

    def test_steps_max_boundary(self):
        """steps=10000 应在边界内正常工作。"""
        points = sample_formula("f", start=1, end=2, steps=10000)
        assert len(points) == 10001
        assert points[0][0] == pytest.approx(1.0)
        assert points[-1][0] == pytest.approx(2.0)

    def test_steps_exceeds_max(self):
        """steps > 10000 应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            sample_formula("f", start=1, end=100, steps=10001)

    def test_log_spaced_output(self):
        """采样频率应呈对数均匀分布（等比数列）。"""
        points = sample_formula("f", start=1, end=1000, steps=3)
        # 对数均匀：相邻频率的比值应相等
        ratios = [points[i + 1][0] / points[i][0] for i in range(len(points) - 1)]
        for ratio in ratios:
            assert ratio == pytest.approx(ratios[0], rel=1e-12)

    def test_log_spaced_output_more_steps(self):
        """更多步数下采样频率应仍呈对数均匀分布。"""
        points = sample_formula("f", start=10, end=10000, steps=10)
        ratios = [points[i + 1][0] / points[i][0] for i in range(len(points) - 1)]
        for ratio in ratios:
            assert ratio == pytest.approx(ratios[0], rel=1e-12)

    def test_formula_evaluated_at_each_point(self):
        """每个采样点的公式值应正确计算。"""
        points = sample_formula("f * 2", start=1, end=10, steps=3)
        for freq, value in points:
            assert value == pytest.approx(freq * 2)

    def test_start_non_positive(self):
        """start <= 0 应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            sample_formula("f", start=0, end=100, steps=5)

    def test_start_negative(self):
        """start 为负值时应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            sample_formula("f", start=-1, end=100, steps=5)

    def test_end_non_positive(self):
        """end <= 0 应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            sample_formula("f", start=1, end=0, steps=5)

    def test_end_negative(self):
        """end 为负值时应抛出 FormulaError。"""
        with pytest.raises(FormulaError):
            sample_formula("f", start=1, end=-5, steps=5)

    def test_return_type(self):
        """返回结果应为 list[list[float]] 类型。"""
        points = sample_formula("f", start=1, end=10, steps=3)
        assert isinstance(points, list)
        for point in points:
            assert isinstance(point, list)
            assert len(point) == 2
            assert isinstance(point[0], float)
            assert isinstance(point[1], float)

    def test_samples_with_complex_formula(self):
        """复杂公式在采样时应正确求值。"""
        points = sample_formula("abs(20 * log10(f))", start=1, end=100, steps=3)
        # 第一个点：abs(20 * log10(1)) = abs(20 * 0) = 0
        assert points[0][1] == pytest.approx(0.0)
        # 最后一个点：abs(20 * log10(100)) = abs(20 * 2) = 40
        assert points[-1][1] == pytest.approx(40.0)

    def test_function_with_non_positive_f_in_sample(self):
        """采样过程中 f 为负值导致函数报错应向上传播。"""
        with pytest.raises(FormulaError):
            sample_formula("sqrt(f)", start=-10, end=10, steps=20)
