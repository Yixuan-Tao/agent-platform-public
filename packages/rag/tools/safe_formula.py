"""Restricted arithmetic evaluator for formulas extracted from EMC standards."""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Callable


class FormulaError(ValueError):
    """Raised when a formula contains unsupported or unsafe syntax."""


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "cos": math.cos,
    "exp": math.exp,
    "log": math.log10,
    "log10": math.log10,
    "max": max,
    "min": min,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}


def evaluate_formula(formula: str, *, f: float) -> float:
    """Evaluate a numeric EMC formula without exposing Python builtins."""
    if not isinstance(formula, str) or not formula.strip():
        raise FormulaError("formula must be a non-empty string")
    normalized = re.sub(r"(?<=\d)\s+(?=\d)", "", formula.strip()).replace("^", "**").replace("×", "*")
    try:
        root = ast.parse(normalized, mode="eval")
        result = float(_evaluate_node(root.body, float(f)))
    except FormulaError:
        raise
    except (ArithmeticError, SyntaxError, TypeError, ValueError, OverflowError) as exc:
        raise FormulaError(f"formula evaluation failed: {exc}") from exc
    if not math.isfinite(result):
        raise FormulaError("formula result must be finite")
    return result


def sample_formula(
    formula: str,
    *,
    start: float,
    end: float,
    steps: int = 200,
) -> list[list[float]]:
    """Sample a formula over a logarithmic frequency axis."""
    start = float(start)
    end = float(end)
    if start <= 0 or end <= 0 or end < start:
        raise FormulaError("frequency range must be positive and ordered")
    if steps < 1 or steps > 10_000:
        raise FormulaError("steps must be between 1 and 10000")
    points: list[list[float]] = []
    for index in range(steps + 1):
        frequency = start * (end / start) ** (index / steps)
        points.append([frequency, evaluate_formula(formula, f=frequency)])
    return points


def _evaluate_node(node: ast.AST, frequency: float) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise FormulaError("only numeric constants are allowed")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id != "f":
            raise FormulaError(f"unknown name: {node.id}")
        return frequency
    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise FormulaError("unsupported binary operator")
        return operation(
            _evaluate_node(node.left, frequency),
            _evaluate_node(node.right, frequency),
        )
    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise FormulaError("unsupported unary operator")
        return operation(_evaluate_node(node.operand, frequency))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise FormulaError("only direct calls to approved functions are allowed")
        function = _FUNCTIONS.get(node.func.id)
        if function is None:
            raise FormulaError(f"unsupported function: {node.func.id}")
        arguments = [_evaluate_node(argument, frequency) for argument in node.args]
        return float(function(*arguments))
    raise FormulaError(f"unsupported syntax: {type(node).__name__}")
