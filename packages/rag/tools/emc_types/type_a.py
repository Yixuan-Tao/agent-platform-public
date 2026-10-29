#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Type_A (辐射发射限值) 数据转换模块
"""

import re

from ..emc_utils import _get_field, normalize_class_name, extract_service_from_label, isfinite_safe, normalize_unit
from ..safe_formula import FormulaError, evaluate_formula, sample_formula


def _eval_formula_py(info, f):
    """与前端 evalFormula 等价的 Python 公式求值"""
    formula = info.get('formula') if isinstance(info, dict) else None
    value = info.get('value') if isinstance(info, dict) else None
    if formula and isinstance(formula, str) and formula.strip():
        s = formula.strip()
        if ';' in s:
            s = s.rsplit(';', 1)[-1].strip()
        if '~' in s and 'log' not in s.split('~')[0].lower():
            s = s.split('~', 1)[-1].strip()
        s = re.sub(r'(?i)(?<![a-zA-Z])log\b', 'log10', s)
        s = re.sub(r'(\d(?:\.\d+)?)\s*log10', r'\1*log10', s)
        try:
            return evaluate_formula(s, f=f)
        except FormulaError:
            return None
    if value is not None:
        try:
            return float(value)
        except Exception:
            return None
    return None


def _compute_chart_config(all_datasets):
    """为 Type_A 图表预计算 y 轴范围与单位（允许负数，按数据自适应）"""
    y_min, y_max = None, None
    y_unit = 'dBµV/m'
    for ds in all_datasets:
        data = ds.get('data') or ds.get('Data') or {}
        meta = data.get('metadata') or {}
        if meta.get('limit_unit'):
            normalized = normalize_unit(str(meta['limit_unit']))
            if normalized:
                y_unit = normalized
        for band in (data.get('bands') or []):
            limits = band.get('limits') or {}
            f1 = (band.get('frequency') or {}).get('start')
            f2 = (band.get('frequency') or {}).get('end')
            if not f1 or not f2 or f1 <= 0 or f2 <= 0:
                continue
            steps = 50
            for det_key, det_limits in limits.items():
                flat = det_limits
                if 'value' not in flat and 'formula' not in flat:
                    for v in det_limits.values():
                        flat = v
                        break
                has_formula = flat.get('formula')
                has_value = flat.get('value') is not None
                if not has_formula and not has_value:
                    continue
                info = {'value': flat.get('value'), 'formula': flat.get('formula')}
                for i in range(steps + 1):
                    freq = f1 * (f2 / f1) ** (i / steps)
                    y = _eval_formula_py(info, freq)
                    if y is not None and isinstance(y, (int, float)) and isfinite_safe(y):
                        if y_min is None or y < y_min:
                            y_min = y
                        if y_max is None or y > y_max:
                            y_max = y
    if y_min is None or y_max is None:
        return None
    if y_min == y_max:
        return {'y_min': int(y_min - 5), 'y_max': int(y_max + 5), 'y_unit': y_unit}
    span = y_max - y_min
    pad = max(5, span * 0.1)
    if y_min < 0 and y_min > -pad:
        pad = max(pad, abs(y_min) + 3)
    import math as _m
    y_min_r = int(_m.floor((y_min - pad) / 5) * 5)
    y_max_r = int(_m.ceil((y_max + pad) / 5) * 5)
    return {'y_min': y_min_r, 'y_max': y_max_r, 'y_unit': y_unit}


def convert_to_dataset(data, source_name):
    """
    将新版 Type_A schema 转换为 dataset 格式。
    相同 (test_no, freq_min, freq_max, category) 的条目合并为一个 band，
    当存在 limit_class 时以嵌套方式存储，否则保持扁平结构。

    字段映射 (key_mapping.json):
      category_name -> cat, unique_label -> label, test_no -> no,
      limit_class -> cls, freq_min_MHz -> fmin, freq_max_MHz -> fmax,
      detector -> det, unit -> unit, value -> val, formula -> form
    """
    items = data.get('data', [])
    has_class_hierarchy = any(_get_field(item, 'cls', 'limit_class') for item in items)

    limit_unit = 'dBµV/m'
    for item in items:
        for limit in item.get('limits', []):
            if _get_field(limit, 'unit'):
                limit_unit = _get_field(limit, 'unit')
                break
        if limit_unit != 'dBµV/m':
            break

    metadata = {
        'standard': _get_field(data, 'title', 'table_title') or source_name,
        'frequency_unit': 'MHz',
        'limit_unit': limit_unit,
        'has_formula': any(
            _get_field(limit, 'form', 'formula')
            for item in items
            for limit in item.get('limits', [])
        ),
        'has_class_hierarchy': has_class_hierarchy
    }

    bands_map = {}
    for item in items:
        freq_min = _get_field(item, 'fmin', 'freq_min_MHz')
        freq_max = _get_field(item, 'fmax', 'freq_max_MHz')
        test_no = _get_field(item, 'no', 'test_no')
        unique_label = _get_field(item, 'label', 'unique_label') or ''
        limit_class = _get_field(item, 'cls', 'limit_class')
        category = _get_field(item, 'cat', 'category_name') or 'General'
        bw = _get_field(item, 'bw')

        if freq_min is None or freq_max is None:
            continue

        key = (test_no, freq_min, freq_max, category, bw)

        if key not in bands_map:
            band_id = test_no if test_no is not None else unique_label
            bands_map[key] = {
                'id': band_id,
                'category': category,
                'service': extract_service_from_label(item),
                'raw_label': unique_label,  # 保留完整 label，供前端勾选栏位显示
                'frequency': {'start': freq_min, 'end': freq_max},
                'bw': bw,
                'limits': {}
            }

        band = bands_map[key]
        cls = normalize_class_name(limit_class) if limit_class else None

        for limit in item.get('limits', []):
            detector = (_get_field(limit, 'det', 'detector') or '').upper()
            if detector not in ('PK', 'QP', 'AV'):
                continue

            formula = _get_field(limit, 'form', 'formula')
            value = _get_field(limit, 'val', 'value')

            if isinstance(formula, str):
                formula = formula.strip()
                if not re.search(r'log|sin|cos|tan|exp|sqrt|abs|min|max|log10', formula, re.I):
                    m = re.match(r'^\s*([-+]?\d+(?:\.\d+)?)\s*(.*)$', formula)
                    if m:
                        num = float(m.group(1))
                        tail = m.group(2).strip()
                        if not tail or tail.startswith('('):
                            value = num if value is None else value
                            formula = None
                if isinstance(formula, str) and ';' in formula:
                    formula = formula.rsplit(';', 1)[-1].strip()
                if isinstance(formula, str) and '~' in formula and 'log' not in formula.split('~')[0].lower():
                    formula = formula.split('~', 1)[-1].strip()
                if isinstance(formula, str):
                    formula = re.sub(r'(?i)\b(lg|log)\b', 'log10', formula)
                    formula = re.sub(r'(\d(?:\.\d+)?)\s*log10', r'\1*log10', formula)

            if value is None and not formula:
                continue

            if detector not in band['limits']:
                band['limits'][detector] = {}

            info = {
                'value': value,
                'formula': formula,
                'unit': _get_field(limit, 'unit') or limit_unit,
            }
            if formula:
                try:
                    info['points'] = sample_formula(
                        formula,
                        start=float(freq_min),
                        end=float(freq_max),
                    )
                except (FormulaError, TypeError, ValueError):
                    info['points'] = []

            if has_class_hierarchy and cls:
                band['limits'][detector][cls] = info
            else:
                band['limits'][detector] = info

    return {
        'name': source_name,
        'type': 'TypeA',
        'data': {
            'metadata': metadata,
            'bands': list(bands_map.values())
        }
    }
