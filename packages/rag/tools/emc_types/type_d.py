#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Type_D (静电放电抗扰度参数表) 数据转换模块
"""

from ..emc_utils import _get_field


def convert_to_dataset(data, source_name):
    """
    将新版 Type_D schema 转换为 dataset 格式。
    Type_D 为静电放电抗扰度参数表 (ESD)。

    字段映射 (key_mapping.json):
      discharge_type -> dtype, test_level_kV -> lvlkv,
      human_body_model -> hbm, discharge_points -> dpts,
      no_of_discharges -> ndis, recovery_time -> rtime,
      function_status.Region_I/II/III -> R1/R2/R3
    """
    items = data.get('data', [])
    discharges = []
    for item in items:
        discharges.append({
            'discharge_type': _get_field(item, 'dtype', 'discharge_type'),
            'test_level_kV': _get_field(item, 'lvlkv', 'test_level_kV'),
            'human_body_model': _get_field(item, 'hbm', 'human_body_model'),
            'discharge_points': _get_field(item, 'dpts', 'discharge_points'),
            'no_of_discharges': _get_field(item, 'ndis', 'no_of_discharges'),
            'recovery_time': _get_field(item, 'rtime', 'recovery_time'),
            'function_status': item.get('function_status')
        })

    return {
        'name': source_name,
        'type': 'type_d',
        'data': {
            'test_item': _get_field(data, 'item', 'test_item') or source_name,
            'discharges': discharges
        }
    }
