#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Type_B (射频/瞬态抗扰度参数表) 数据转换模块
支持 RI(辐射抗扰度)、BCI(大电流注入)、CIS(传导抗扰度)
"""

from ..emc_utils import _get_field
from ..safe_formula import FormulaError, sample_formula
from ..emc_chart_policy import normalize_type_b_group


def convert_to_dataset(data, source_name):
    """
    将 Type_B JSON 转换为 dataset 格式。

    Type_B 数据结构包含两部分：
    1. 测试方式：频率范围(fmin/fmax)、测试等级(lvl)、测试等级值(val)、单位(unit)、测试方法(mod)
    2. 功能状态：function_status(R1/R2/R3)，值为 A/B/C 或 null
    """
    items = data.get('data', [])

    test_methods = []
    regions = set()
    levels = set()
    all_units = set()

    for idx, item in enumerate(items):
        fmin = _get_field(item, 'fmin', 'freq_min')
        fmax = _get_field(item, 'fmax', 'freq_max')

        if fmin is None or fmax is None:
            continue

        fmin = float(fmin)
        fmax = float(fmax)

        test_level = _get_field(item, 'lvl', 'test_level', 'level') or 'Level Unknown'
        value = _get_field(item, 'val', 'value')
        formula = _get_field(item, 'form', 'formula')
        unit = _get_field(item, 'unit', 'units') or ''
        method = _get_field(item, 'meth', 'method') or ''
        modulation = _get_field(item, 'mod', 'modulation') or ''
        category = _get_field(item, 'cat', 'category') or ''
        label = _get_field(item, 'label', 'description') or ''

        fs = item.get('function_status', {}) or {}

        if isinstance(fs, dict):
            for r in ['R1', 'R2', 'R3', 'Region_I', 'Region_II', 'Region_III']:
                if r in fs:
                    regions.add(r)

        levels.add(test_level)
        all_units.add(unit)

        def format_freq(val):
            if val >= 1:
                return str(int(val)) if val == int(val) else f"{val:.6f}".rstrip('0').rstrip('.')
            elif val >= 0.001:
                return f"{val * 1000:.3f}".rstrip('0').rstrip('.') + 'kHz'
            elif val >= 1e-6:
                return f"{val * 1000000:.0f}" + 'Hz'
            else:
                return f"{val:.6e}".replace('e', '×10^')

        freq_str = f"{format_freq(fmin)}-{format_freq(fmax)}"

        note = _get_field(item, 'nt', 'note')
        method_family, physical_quantity, canonical_unit = normalize_type_b_group(
            category, method, unit
        )

        sampled_points = []
        if formula:
            try:
                sampled_points = sample_formula(
                    str(formula), start=fmin, end=fmax, steps=100
                )
            except FormulaError:
                sampled_points = []

        test_methods.append({
            'id': f"method_{idx}",
            'label': label,
            'category': category,
            'frequency': freq_str,
            'fmin': fmin,
            'fmax': fmax,
            'test_level': test_level,
            'value': float(value) if value is not None else None,
            'formula': formula,
            'points': sampled_points,
            'unit': unit,
            'method_family': method_family,
            'physical_quantity': physical_quantity,
            'canonical_unit': canonical_unit,
            'method': method,
            'mod': modulation,
            'function_status': fs,
            'note': note
        })

    regions = sorted(list(regions))
    levels = sorted(list(levels))
    primary_unit = next(iter(all_units), '')

    return {
        'name': source_name,
        'type': 'type_b',
        'data': {
            'test_item': _get_field(data, 'item', 'test_item') or source_name,
            'title': _get_field(data, 'title', 'table_title') or '',
            'test_methods': test_methods,
            'regions': regions,
            'levels': levels,
            'unit': primary_unit
        }
    }
