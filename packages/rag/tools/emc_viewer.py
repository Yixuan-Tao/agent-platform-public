#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMC测试标准表格可视化工具
参考 example.py / example_out.html 实现
支持新版 Type_A schema 和多文件对比

类型转换逻辑已拆分到 emc_types/ 包，公共工具已拆分到 emc_utils.py
"""

import json
import base64
import argparse
from pathlib import Path

from .emc_utils import _parse_json, _get_field, normalize_dataset
from .emc_types import CONVERTERS, TYPE_A_CONFIG_COMPUTER
from .emc_chart_policy import frequency_extent


# 频段分组默认值：4 组，按 MHz 单位划分（与 band.bandStart/bandEnd 单位一致）
# 边界说明：[min, max)；min=None 表示无下限；max=None 表示无上限
# 通过 chart_config.frequency_groups 可在前端 / Python 层覆盖
DEFAULT_FREQUENCY_GROUPS = [
    {'id': 'low',    'name': '低频段 (≤30 MHz)',     'min': None,  'max': 30},
    {'id': 'mid',    'name': '中频段 (30-300 MHz)',   'min': 30,    'max': 300},
    {'id': 'high',   'name': '高频段 (300-1000 MHz)', 'min': 300,   'max': 1000},
    {'id': 'ultra',  'name': '超高频 (>1 GHz)',       'min': 1000,  'max': None},
]


def build_emc_data(json_data_list, titles=None):
    """把一条或多条 JSON 文本归一化成 emc_data {'datasets': [...]} 结构。"""
    if not json_data_list:
        raise ValueError('json_data is empty')

    all_datasets = []
    for i, data in enumerate(json_data_list):
        parsed, json_text = _parse_json(data)

        title = None
        if titles and i < len(titles) and titles[i]:
            title = titles[i]
        elif _get_field(parsed, 'title', 'table_title'):
            title = _get_field(parsed, 'title', 'table_title')
        elif _get_field(parsed, 'metadata', 'standard') and (
            isinstance(parsed.get('metadata'), dict) and parsed['metadata'].get('standard')
        ):
            title = parsed['metadata']['standard']
        else:
            title = f"Source {i + 1}"

        if 'datasets' in parsed and isinstance(parsed['datasets'], list):
            for ds in parsed['datasets']:
                all_datasets.append(normalize_dataset(ds, title))
        else:
            data_type = _get_field(parsed, 'type', 'table_type')
            if data_type in CONVERTERS and 'data' in parsed:
                converter = CONVERTERS[data_type]
                all_datasets.append(converter(parsed, title))
            elif 'bands' in parsed:
                all_datasets.append({
                    'name': title,
                    'data': {
                        'metadata': parsed.get('metadata', {'standard': title}),
                        'bands': parsed['bands']
                    }
                })
            else:
                raise ValueError(f'Unsupported JSON structure in input {i + 1}')

    return {'datasets': all_datasets}


def main(json_data_list, titles=None):
    """主函数：生成统一的可视化HTML（保留原接口）"""
    emc_data = build_emc_data(json_data_list, titles)
    return generate_html(emc_data)


# 模板目录路径
_TEMPLATE_DIR = Path(__file__).parent / "emc_viewer_templates"

# 从外部文件读取 CSS 样式，便于前端独立维护
_STYLE_CSS = (_TEMPLATE_DIR / "style.css").read_text(encoding="utf-8")

# 从外部文件读取 JS 脚本（含占位符 ___JSON_DATA___ / ___DEFAULT_FREQUENCY_GROUPS___）
_SCRIPT_JS = (_TEMPLATE_DIR / "script.js").read_text(encoding="utf-8")

# _HTML_TEMPLATE 仅包含 HTML 壳 + CSS + JS 占位符
# JS 在 generate_html 中通过 _SCRIPT_JS 动态注入
_HTML_TEMPLATE = (
    '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
    '<title>EMC 标准数据可视化</title>\n<style>\n'
    + _STYLE_CSS +
    '\n</style>\n</head>\n<body>\n'
    '<div class="toolbar" id="toolbar">\n'
    '    <button onclick="downloadPNG()">📥 导出 PNG</button>\n'
    '    <button class="primary" onclick="resetZoom()">⟲ 重置缩放</button>\n'
    '</div>\n'
    '<div id="chart-container"><div id="chart"></div></div>\n'
    '<div id="type-a-zoom-toolbar" class="zoom-toolbar"></div>\n'
    '<div id="type-a-readout" class="type-a-readout" hidden></div>\n'
    '<div id="type-a-diff" class="type-a-diff" hidden></div>\n'
    '<div id="control-panel"></div>\n'
    '<div id="type-b-panel"></div>\n'
    '<div id="type-c-panel"></div>\n'
    '<div id="type-d-panel"></div>\n'
    '\n<script>\n'
    '___JS_PLACEHOLDER___'
    '\n</script>\n</body>\n</html>'
)


def to_inline_bundle(emc_data):
    """生成内联 bundle 供 SPA 内嵌使用"""
    _BODY_MARKUP = """<div class="toolbar" id="toolbar">
    <button id="btn-download-png">Export PNG</button>
    <button class="primary" id="btn-reset-zoom">Reset Zoom</button>
</div>
<div id="chart" style="height:84vh"></div>
<div id="type-a-zoom-toolbar" class="zoom-toolbar"></div>
<div id="type-a-readout" class="type-a-readout" hidden></div>
<div id="type-a-diff" class="type-a-diff" hidden></div>
<div id="type-a-bandwidth" class="type-a-bandwidth" hidden></div>
<div id="control-panel"></div>
<div id="type-b-panel"></div>
<div id="type-c-panel"></div>
<div id="type-d-panel"></div>"""

    _INIT_SCRIPT_TEMPLATE = _SCRIPT_JS

    type_a = [d for d in emc_data.get('datasets', []) if (d.get('type') or 'TypeA') == 'TypeA']
    if type_a:
        # 保留调用方已写入的 chart_config（如自定义 frequency_groups / 自定义 y 范围等）
        cfg = emc_data.get('chart_config') if isinstance(emc_data.get('chart_config'), dict) else None
        cfg = dict(cfg) if cfg else {}
        cfg.update(TYPE_A_CONFIG_COMPUTER(type_a) or {})
        ranges = frequency_extent([
            {
                'ranges': [
                    ((band.get('frequency') or {}).get('start'), (band.get('frequency') or {}).get('end'))
                    for band in ((dataset.get('data') or {}).get('bands') or [])
                ]
            }
            for dataset in type_a
        ])
        cfg['x_union'] = ranges['union']
        cfg['x_intersection'] = ranges['intersection']
        # 频段分组默认配置：调用方未指定时填入默认 4 组
        if not cfg.get('frequency_groups'):
            cfg['frequency_groups'] = DEFAULT_FREQUENCY_GROUPS
        emc_data['chart_config'] = cfg

    json_data = json.dumps(emc_data, ensure_ascii=False)

    init_script = _INIT_SCRIPT_TEMPLATE.replace('___JSON_DATA___', json_data)
    # 注入前端默认常量（用于 buildControlPanel 兜底分支）
    init_script = init_script.replace(
        '___DEFAULT_FREQUENCY_GROUPS___',
        json.dumps(DEFAULT_FREQUENCY_GROUPS, ensure_ascii=False)
    )

    return {
        'css': _STYLE_CSS,
        'echarts_script': '',
        'html': _BODY_MARKUP,
        'init_script': init_script
    }


def generate_html(emc_data):
    """生成HTML - 基于 example.py 的界面方案（保留原有 CLI / 独立 HTML 能力）"""
    type_a = [d for d in emc_data.get('datasets', []) if (d.get('type') or 'TypeA') == 'TypeA']
    if type_a:
        # 保留调用方已写入的 chart_config（如自定义 frequency_groups / 自定义 y 范围等）
        cfg = emc_data.get('chart_config') if isinstance(emc_data.get('chart_config'), dict) else None
        cfg = dict(cfg) if cfg else {}
        cfg.update(TYPE_A_CONFIG_COMPUTER(type_a) or {})
        ranges = frequency_extent([
            {
                'ranges': [
                    ((band.get('frequency') or {}).get('start'), ((band.get('frequency') or {}).get('end')))
                    for band in ((dataset.get('data') or {}).get('bands') or [])
                ]
            }
            for dataset in type_a
        ])
        cfg['x_union'] = ranges['union']
        cfg['x_intersection'] = ranges['intersection']
        if not cfg.get('frequency_groups'):
            cfg['frequency_groups'] = DEFAULT_FREQUENCY_GROUPS
        emc_data['chart_config'] = cfg
    json_data = json.dumps(emc_data, ensure_ascii=False)

    # 注入数据到 JS 脚本，再嵌入 HTML 模板
    js_with_data = _SCRIPT_JS.replace('___JSON_DATA___', json_data)
    js_with_data = js_with_data.replace(
        '___DEFAULT_FREQUENCY_GROUPS___',
        json.dumps(DEFAULT_FREQUENCY_GROUPS, ensure_ascii=False)
    )
    html = _HTML_TEMPLATE.replace('___JS_PLACEHOLDER___', js_with_data)
    html_bytes = html.encode('utf-8')
    html_b64 = base64.b64encode(html_bytes).decode('utf-8')

    return {
        'result': f'data:text/html;base64,{html_b64}',
        'html_content': html
    }


def main_cli():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="EMC Standards Visualization Tool")
    parser.add_argument("inputs", nargs='+', help="Input JSON file(s)")
    parser.add_argument("-o", "--output", help="Output HTML file")
    parser.add_argument("--titles", nargs='*', help="Titles for each input file")

    args = parser.parse_args()

    json_data_list = []
    for input_path in args.inputs:
        path = Path(input_path)
        if path.exists() and path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                json_data_list.append(f.read())
        else:
            json_data_list.append(input_path)

    result = main(json_data_list, args.titles)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result['html_content'])
        print(f"[OK] Generated: {args.output}")
    else:
        print(result['html_content'])


if __name__ == '__main__':
    main_cli()
