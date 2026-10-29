"""
元器件参数模板解析工具 - 从 XLSX 模板读取参数结构并转换为对比工具可用的 JSON 格式。

功能：
    1. 使用 openpyxl 读取 XLSX 参数模板
    2. 自动解析章节结构，按元器件类型提取参数表
    3. 输出与 `compare_params_buk6d120.json` 格式兼容的 dict
    4. 支持通过元器件类型/关键字查询对应模板
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

from utils.logger import log

# 默认模板路径（相对于项目根目录）——公开演示 CSV 参数表
_DEFAULT_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources",
    "component-parameters.csv",
)

# 元器件类型关键词到章节标题的映射
_COMPONENT_TYPE_MAP: dict[str, str] = {
    "mosfet": "MOSFET",
    "tvs": "TVS",
    "esd": "TVS",
    "zener": "Zener",
    "schottky": "Schottky",
    "bjt": "Bi-polar",
    "transistor": "Bi-polar",
    "mlcc": "MLCC",
    "capacitor": "MLCC",
    "电阻": "Thick Film",
    "resistor": "Thick Film",
    "电感": "Inductors",
    "inductor": "Inductors",
    "connector": "连接器",
    "ic": "集成电路",
}

# 元器件类型检测规则（按优先级顺序匹配）：(类型名, 关键词列表)
# 关键词大小写不敏感；命中任一即判定为该类型
_TYPE_DETECT_RULES: list[tuple[str, list[str]]] = [
    ("MOSFET", ["mosfet", "场效应管", "n-ch", "p-ch", "n沟道", "p沟道",
                "vdss", "rds(on)", "rds_on", "buk", "sqa", "power mos"]),
    ("TVS", ["tvs", "esd", "瞬态抑制", "瞬态电压", "钳位", "clamping",
             "v_br", "vbr", "静电", "浪涌保护", "sod-", "sot-",
             "usb保护", "保护二极管"]),
    ("Zener", ["zener", "齐纳", "稳压二极管"]),
    ("Schottky", ["schottky", "肖特基"]),
    ("Bi-polar", ["bjt", "双极", "晶体管", "npn", "pnp", "transistor"]),
    ("MLCC", ["mlcc", "陶瓷电容", "capacitor", "电容"]),
    ("Thick Film", ["厚膜", "thick film", "电阻", "resistor"]),
    ("Thermistors", ["thermistor", "热敏", "ntc", "ptc"]),
    ("Inductors", ["电感", "inductor"]),
    ("Common Mode Chokes", ["共模", "choke"]),
    ("LED", ["led", "发光二极管"]),
    ("Crystal", ["crystal", "晶振", "晶体"]),
    ("Ceramic Resonator", ["陶瓷谐振", "resonator"]),
    ("Connectors", ["连接器", "connector"]),
    ("Regulator", ["regulator", "ldo", "稳压器"]),
    ("Op Amp", ["运算放大器", "运放", "op amp"]),
    ("Driver", ["driver", "驱动器"]),
    ("Sensor", ["sensor", "传感器"]),
    ("Memory", ["memory", "存储器", "flash", "eeprom"]),
    ("Processor", ["processor", "mcu", "处理器", "单片机", "soc"]),
    ("AD_DA", ["adc", "dac", "模数", "数模"]),
    ("Interface", ["interface", "接口", "transceiver"]),
    ("Timer", ["timer", "定时器"]),
    ("Logic Miscellaneous", ["逻辑", "logic"]),
    ("Comp", ["比较器", "comparator"]),
    ("Audio Amp", ["音频功放", "audio amp"]),
]

# 风险等级到 severity 映射
_RISK_TO_SEVERITY = {
    "高": "critical",
    "中": "normal",
    "低": "low",
}


def _detect_component_type(doc_name: str, model: str = "", query: str = "") -> str:
    """从文档名、型号和用户查询中检测元器件类型（如 MOSFET/TVS/MLCC）。

    Args:
        doc_name: 文档文件名（大写处理前）。
        model: 产品型号字符串（可选）。
        query: 用户查询文本（可选），用于元器件问答场景。

    Returns:
        检测到的元器件类型英文名，如 ``"MOSFET"``。未知类型返回 ``"集成电路"``。
    """
    combined = f"{doc_name} {model} {query}".lower()
    # 按优先级遍历规则表：先命中更具体的类型（如 MOSFET 优先于通用封装词）
    for type_name, keywords in _TYPE_DETECT_RULES:
        for kw in keywords:
            if kw in combined:
                return type_name
    return "集成电路"


def _read_rows(path: str) -> list[tuple]:
    """按扩展名读取参数表为行列表（统一数据源，xlsx/csv 均可）。

    - ``.csv``：用标准 csv 模块解析（UTF-8，兼容 BOM）
    - 其他（.xlsx 等）：用 openpyxl 读取首个 sheet
    """
    if not os.path.exists(path):
        return []
    if path.lower().endswith(".csv"):
        import csv as _csv

        rows: list[tuple] = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in _csv.reader(f):
                rows.append(tuple(r))
        return rows

    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return list(wb.active.iter_rows(values_only=True))
    finally:
        wb.close()


def _get_section_name_to_row_map(rows: list[tuple]) -> dict[str, int]:
    """解析参数表行列表，返回 {元器件类型名: 起始行索引(0-based)} 映射表。

    排除项：
        - 空行
        - Markdown 粗体/修订说明行（如 ``1. **全局合并**...``，以 ``**`` 开头）
    """
    sections: dict[str, int] = {}
    for i, row in enumerate(rows):
        v = row[0] if row else None
        if not v or not isinstance(v, str):
            continue
        v = v.strip()
        # 排除修订说明（Markdown 粗体包裹）与非章节行
        if not v or v.startswith("**") or "**" in v:
            continue
        # 匹配 "13. MOSFET（功率MOSFET）" 或 "8. TVS（瞬态电压抑制二极管）"
        m = re.match(r"\d+\.\s*(.+?)[（(]", v)
        if m:
            sections[m.group(1).strip()] = i
            continue
        # 兜底：匹配 "1. MLCC" 等无括号章节标题
        m2 = re.match(r"\d+\.\s*(.+)$", v)
        if m2:
            sec_name = m2.group(1).strip()
            if sec_name:
                sections[sec_name] = i
    return sections


def parse_xlsx_template(
    xlsx_path: str = _DEFAULT_TEMPLATE,
    component_type: str = "",
    doc_name: str = "",
    model: str = "",
    query: str = "",
) -> dict[str, Any]:
    """解析参数表（XLSX 或真 CSV），为指定元器件类型生成对比参数模板。

    Args:
        xlsx_path: 参数表文件路径（.xlsx / .csv 均可）。
        component_type: 元器件类型（如 ``"MOSFET"``, ``"TVS"``）。为空时自动从 doc_name/model/query 检测。
        doc_name: 文档文件名（用于自动类型检测）。
        model: 产品型号（用于自动类型检测）。
        query: 用户查询文本（用于自动类型检测，可选）。

    Returns:
        与 ``compare_params_buk6d120.json`` 格式兼容的参数模板 dict：
        ``{category: {param_name: {unit, tolerance, severity, description}}}``。
    """
    if not os.path.exists(xlsx_path):
        log.warning("参数表文件不存在: %s，回退到默认 JSON 模板", xlsx_path)
        return _fallback_json_template()

    # 确定元器件类型
    if not component_type:
        component_type = _detect_component_type(doc_name, model, query)
    log.info("参数模板检测到元器件类型: %s", component_type)

    rows = _read_rows(xlsx_path)
    if not rows:
        log.warning("参数表读取为空: %s，回退到默认 JSON 模板", xlsx_path)
        return _fallback_json_template()

    sections = _get_section_name_to_row_map(rows)

    # 查找目标章节
    target_idx = None
    for sec_name, row_idx in sections.items():
        if component_type.lower() in sec_name.lower():
            target_idx = row_idx
            break

    if target_idx is None:
        log.warning("未找到类型 %s 的参数表 (可用类型: %s)", component_type, list(sections.keys()))
        return _fallback_json_template()

    # 读取参数表（从表头行+数据行）
    return _extract_params_table(rows, target_idx, component_type)


def _extract_params_table(rows: list[tuple], start_idx: int, component_type: str) -> dict[str, Any]:
    """从指定起始行索引开始，读取参数表并转换为模板格式。

    表结构: [序号, 关键参数, 替代评估要点, 风险等级]
    """
    # 找到表头行（包含"关键参数"的行）
    header_idx = None
    for offset in range(5):
        row_idx = start_idx + offset
        if row_idx >= len(rows):
            break
        row = rows[row_idx]
        if any(cell and "关键参数" in str(cell) for cell in row):
            header_idx = row_idx
            break

    if header_idx is None:
        log.warning("在行 %d 附近未找到参数表头", start_idx)
        return _fallback_json_template()

    # 读取数据行（表头行+1 开始，直到遇到空行或下一个章节）
    result: dict[str, dict[str, Any]] = {}
    for row in rows[header_idx + 1:]:
        seq = row[0] if len(row) > 0 else None
        param_name = row[1] if len(row) > 1 else None
        eval_points = row[2] if len(row) > 2 else None
        risk = row[3] if len(row) > 3 else None
        # 结束条件：空序号，或下一个章节标题
        if seq is None or (isinstance(seq, str) and re.match(r"\d+\.\s*", seq)):
            break
        if not param_name:
            continue

        param_str = str(param_name).strip()
        # 去除序号前缀（如 "1. "）
        param_str = re.sub(r"^\d+[\.\、]\s*", "", param_str)
        # 提取主要参数名（括号前的内容作为 key）
        param_key = re.split(r"[（(]", param_str)[0].strip()
        if not param_key:
            continue

        # 确定参数分类
        category = _categorize_param(param_str, component_type, eval_points)

        # 确定 severity（从风险等级或参数名推测）
        risk_str = str(risk or "").strip()
        severity = _RISK_TO_SEVERITY.get(risk_str, "normal")
        # 含「P1」标记的降一级
        if "【P1】" in param_str and severity == "critical":
            severity = "normal"

        # 确定单位（从参数名和评估要点中提取）
        unit = _extract_unit(param_str, str(eval_points or ""))

        # 确定容差（默认空，交由 LLM 提取时判断）
        tolerance = ""

        # 描述
        description = param_str
        if eval_points:
            desc_detail = str(eval_points).strip()
            # 取前 40 字作为描述
            description = desc_detail[:60] if len(desc_detail) > 30 else desc_detail

        if category not in result:
            result[category] = {}
        result[category][param_key] = {
            "unit": unit,
            "tolerance": tolerance,
            "severity": severity,
            "description": description,
        }

    return result if result else _fallback_json_template()


def _categorize_param(
    param_name: str, component_type: str, eval_points: Any = None,
) -> str:
    """根据参数名和元器件类型确定参数分类（electrical/package/thermal/...）。"""
    p = param_name.lower()
    # 封装类
    if any(kw in p for kw in ["封装", "引脚", "package", "pin", "smt", "焊接", "卷盘", "带宽"]):
        return "package"
    # 热类
    if any(kw in p for kw in ["热阻", "温度", "thermal", "结温", "功率耗散", "pd", "tj"]):
        return "thermal"
    # 认证/合规类
    if any(kw in p for kw in ["认证", "合规", "rohs", "reach", "aec", "msl"]):
        return "compliance"
    # 安全/可靠性类
    if any(kw in p for kw in ["失效", "安全", "雪崩", "soa", "tddb", "esd", "hbm", "cdm"]):
        return "reliability"
    # 默认：电气类
    return "electrical"


def _extract_unit(param_name: str, eval_points: str) -> str:
    """从参数名和评估要点中提取单位。"""
    combined = param_name + " " + eval_points
    # 常见单位正则
    unit_patterns = [
        (r"p[Ff]", "pF"),
        (r"n[Ff]", "nF"),
        (r"µ[Ff]|u[Ff]", "µF"),
        (r"m[Ff]", "mF"),
        (r"[Vv]", "V"),
        (r"[Aa]", "A"),
        (r"m[Ωω]|mohm", "mΩ"),
        (r"[Ωω]|ohm", "Ω"),
        (r"[Kk][Ww]", "K/W"),
        (r"[Ww]", "W"),
        (r"[nN][Cc]", "nC"),
        (r"[Kk][Vv]", "kV"),
        (r"n[sS]", "ns"),
        (r"[Mm][HHz]", "MHz"),
        (r"[Kk][HHz]", "kHz"),
        (r"[HHz]", "Hz"),
        (r"[mM][Vv]", "mV"),
        (r"µ[Aa]|u[Aa]", "µA"),
        (r"m[Aa]", "mA"),
        (r"[nN][Aa]", "nA"),
    ]
    for pattern, unit in unit_patterns:
        if re.search(pattern, combined, re.IGNORECASE):
            return unit
    return ""


def _fallback_json_template() -> dict[str, Any]:
    """返回手写 JSON 模板作为回退（当 XLSX 不可用时）。"""
    import json

    # 查找项目中的默认模板
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback_path = os.path.join(base_dir, "tests", "integration", "compare_params_buk6d120.json")
    if os.path.exists(fallback_path):
        try:
            with open(fallback_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def list_available_types(xlsx_path: str = _DEFAULT_TEMPLATE) -> list[str]:
    """列出参数表中可用的元器件类型。"""
    if not os.path.exists(xlsx_path):
        return []
    rows = _read_rows(xlsx_path)
    if not rows:
        return []
    sections = _get_section_name_to_row_map(rows)
    return list(sections.keys())
