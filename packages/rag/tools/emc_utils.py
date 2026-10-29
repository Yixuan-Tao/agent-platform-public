#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMC 公共工具函数
供各类型转换模块和主模块复用
"""

import json
import re


def _parse_json(data):
    """解析输入 JSON，支持 BOM、前后缀垃圾文本"""
    if data is None:
        raise ValueError('json_data is empty')
    json_text = str(data).lstrip('\ufeff').strip()
    if json_text and json_text[0] not in '{[':
        fo = json_text.find('{')
        fa = json_text.find('[')
        cand = [i for i in (fo, fa) if i != -1]
        if cand:
            json_text = json_text[min(cand):].strip()
    return json.loads(json_text), json_text


def _get_field(data, *keys):
    """从 data 中按 keys 顺序获取第一个存在的值，支持新旧字段名映射"""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def normalize_dataset(ds, default_name):
    """规范化数据集字段名，确保使用小写 name"""
    return {
        'name': ds.get('name') or ds.get('Name') or default_name,
        'data': ds.get('data') or ds.get('Data') or {}
    }


def normalize_class_name(cls):
    """将 'Class 3' 等格式归一化为 'Class3'，用于颜色匹配"""
    if not cls:
        return cls
    s = str(cls).strip()
    m = re.match(r'Class\s+(\S+)', s, re.IGNORECASE)
    if m:
        return 'Class' + m.group(1)
    return s


def extract_service_from_label(item):
    """从 unique_label 中提取业务/服务名称，用于更友好的图例展示"""
    unique_label = item.get('unique_label', '')
    category = item.get('category_name', '')
    test_no = item.get('test_no')
    if not unique_label:
        return None
    s = str(unique_label).strip()
    s = re.sub(r'\s*[-–—]\s*Class\s+\S+', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'\s*\([^)]*\)\s*$', '', s).strip()
    if not s:
        return None
    if test_no is not None and s.lower() == str(test_no).lower():
        return None
    if s.lower() == str(category).strip().lower():
        return None
    return s


def isfinite_safe(x):
    """跨 Python/JS 一致的有限数判断"""
    try:
        import math
        return math.isfinite(x)
    except Exception:
        return x is not None and x == x and x not in (float('inf'), float('-inf'))


# ---------- 单位归一化 ----------

# 单位别名映射表：所有变体 → 统一标准形式
_UNIT_ALIASES = {
    'dbμv/m': 'dBµV/m', 'dbµv/m': 'dBµV/m', 'dbuv/m': 'dBµV/m',
    'dbμv':   'dBµV',   'dbµv':   'dBµV',   'dbuv':   'dBµV',
    'dbμa/m': 'dBµA/m', 'dbµa/m': 'dBµA/m', 'dbua/m': 'dBµA/m',
    'dbμa':   'dBµA',   'dbµa':   'dBµA',   'dbua':   'dBµA',
    'dbm':    'dBm',    'db':     'dB',
    'v/m':    'V/m',    'a/m':    'A/m',    'ma':     'mA',
}


def normalize_unit(raw: str) -> str:
    """将各种单位变体归一化为标准形式。

    处理以下情况：
    1. 括号: dB(µV/m) → dBµV/m
    2. Unicode μ: μ (U+03BC) → µ (U+00B5)
    3. 别名: dBuV/m → dBµV/m, dbµa → dBµA 等
    """
    if not raw:
        return ''
    # 1. 去括号
    text = raw.strip().replace('(', '').replace(')', '')
    # 2. 统一 Unicode μ (U+03BC) → µ (U+00B5)
    text = text.replace('\u03bc', '\u00b5').replace('\u039c', '\u00b5')
    # 3. 别名映射（大小写不敏感）
    key = text.lower()
    return _UNIT_ALIASES.get(key, text)
