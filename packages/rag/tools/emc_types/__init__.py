#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMC 类型转换模块注册表

类型名与转换函数的映射，供 build_emc_data 按类型名分发调用
"""

from .type_a import convert_to_dataset as convert_type_a, _compute_chart_config as compute_type_a_config
from .type_b import convert_to_dataset as convert_type_b
from .type_c import convert_to_dataset as convert_type_c
from .type_d import convert_to_dataset as convert_type_d

CONVERTERS = {
    'Type_A': convert_type_a,
    'Type_B': convert_type_b,
    'Type_C': convert_type_c,
    'Type_D': convert_type_d,
}

TYPE_A_CONFIG_COMPUTER = compute_type_a_config
