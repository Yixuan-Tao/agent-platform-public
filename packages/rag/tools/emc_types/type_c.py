#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..emc_utils import _get_field


def convert_to_dataset(data, source_name):
    items = data.get('data', [])
    pulses = []
    for item in items:
        params = item.get('params', {}) or item.get('parameters', {})
        test_req = item.get('test_requirement', {})
        pulses.append({
            'pulse_id': _get_field(item, 'pid', 'pulse_id'),
            'voltage_V': _get_field(params, 'volt', 'voltage_V'),
            'duration': _get_field(params, 'dur', 'duration'),
            'duration_unit': _get_field(params, 'dunit', 'duration_unit'),
            'internal_resistance_ohm': _get_field(params, 'res', 'internal_resistance_ohm'),
            'test_level': _get_field(item, 'lvl', 'test_level'),
            'function_status': item.get('function_status'),
            'test_req_raw': _get_field(test_req, 'raw'),
            'test_req_npulses': _get_field(test_req, 'npulses'),
            'test_req_interval': _get_field(test_req, 'interval'),
            'test_req_duration': _get_field(test_req, 'duration'),
            'note': _get_field(item, 'nt', 'note')
        })

    return {
        'name': source_name,
        'type': 'type_c',
        'data': {
            'test_item': _get_field(data, 'item', 'test_item') or source_name,
            'pulses': pulses
        }
    }
