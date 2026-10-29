"""管理员维护的证据 Excel 模板校验与导入。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from utils.evidence_store import EvidenceItem, EvidenceStore


_SHEETS: dict[str, dict[str, Any]] = {
    "EMC证据": {
        "domain": "emc",
        "required": {"subject_id", "field_name", "value_raw", "value_kind", "source_file", "page", "section", "test_item"},
    },
    "器件证据": {
        "domain": "component",
        "required": {"subject_id", "component_family", "field_name", "value_raw", "value_kind", "source_file", "page", "section"},
    },
    "替代规则": {
        "domain": "component",
        "required": {"parameter", "rule", "critical"},
        "record_type": "rule",
    },
}


def validate_workbook(path: Path) -> list[dict]:
    """校验受支持 sheet 的列与必填单元格，返回标准化行。"""
    path = Path(path)
    if path.suffix.lower() != ".xlsx" or not path.is_file():
        raise ValueError("仅支持存在的 .xlsx 证据模板")
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows: list[dict] = []
        found = False
        for name, spec in _SHEETS.items():
            if name not in workbook.sheetnames:
                continue
            found = True
            sheet = workbook[name]
            headers = [str(value).strip() if value is not None else "" for value in next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))]
            missing = spec["required"] - set(headers)
            if missing:
                raise ValueError(f"{name} 缺少必填列: {', '.join(sorted(missing))}")
            for index, values in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
                if not any(value is not None and str(value).strip() for value in values):
                    continue
                item = dict(zip(headers, values))
                for field in spec["required"]:
                    if item.get(field) is None or not str(item[field]).strip():
                        raise ValueError(f"{name} 第 {index} 行缺少 {field}")
                item["domain"] = spec["domain"]
                item["record_type"] = spec.get("record_type", "evidence")
                rows.append(item)
        if not found:
            raise ValueError("模板必须包含 EMC证据 或 器件证据 sheet")
        if not rows:
            raise ValueError("模板不包含有效证据行")
        return rows
    finally:
        # read_only 模式会持有文件句柄；必须关闭，否则 Windows 上临时目录清理会失败
        workbook.close()


def import_workbook(path: Path, store: EvidenceStore, name: str) -> dict:
    rows = validate_workbook(path)
    domains = {row["domain"] for row in rows}
    if len(domains) != 1:
        raise ValueError("一次导入只能包含一个领域的证据")
    domain = domains.pop()
    version = store.create_version(name, domain)
    items = [EvidenceItem(
        subject_id=str(row["subject_id"]).strip(), domain=domain,
        field_name=str(row["field_name"]).strip(), value_raw=str(row["value_raw"]).strip(),
        value_normalized=row.get("value_normalized"), unit=str(row.get("unit") or ""),
        value_kind=str(row["value_kind"]).strip(), test_condition=str(row.get("test_condition") or ""),
        source_file=str(row["source_file"]).strip(), page=int(row["page"]),
        section=str(row["section"]).strip(), evidence_status=str(row.get("evidence_status") or "confirmed"),
        test_item=str(row.get("test_item") or ""), component_family=str(row.get("component_family") or ""),
        risk_level=str(row.get("risk_level") or ""), evaluation_note=str(row.get("evaluation_note") or ""),
    ) for row in rows if row["record_type"] == "evidence"]
    if items:
        store.add_items(version["id"], items)
    rules = [{
        "parameter": str(row["parameter"]).strip(), "rule": str(row["rule"]).strip(),
        "critical": str(row["critical"]).strip().lower() in {"true", "1", "是", "y"},
        "derating_factor": float(row.get("derating_factor") or 1.0),
        "require_condition": str(row.get("require_condition") or "").strip().lower() in {"true", "1", "是", "y"},
    } for row in rows if row["record_type"] == "rule"]
    if rules:
        store.add_rules(version["id"], rules)
    return version
