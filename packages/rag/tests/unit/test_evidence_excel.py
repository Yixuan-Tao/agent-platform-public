# Test values are synthetic fixtures for software behavior.
from openpyxl import Workbook
import pytest

from utils.evidence_excel import import_workbook, validate_workbook
from utils.evidence_store import EvidenceStore


def _workbook(path, *, include_page=True):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "EMC证据"
    headers = ["subject_id", "field_name", "value_raw", "value_kind", "source_file", "page", "section", "test_item"]
    if not include_page:
        headers.remove("page")
    sheet.append(headers)
    row = {
        "subject_id": "CISPR25-2021", "field_name": "频段", "value_raw": "150 kHz–30 MHz",
        "value_kind": "range", "source_file": "cispr.pdf", "page": 12, "section": "5.1", "test_item": "CE",
    }
    sheet.append([row[key] for key in headers])
    workbook.save(path)


def test_excel_validation_requires_all_required_columns(tmp_path):
    path = tmp_path / "invalid.xlsx"
    _workbook(path, include_page=False)

    with pytest.raises(ValueError, match="page"):
        validate_workbook(path)


def test_excel_import_creates_a_draft_evidence_version(tmp_path):
    path = tmp_path / "emc.xlsx"
    _workbook(path)
    store = EvidenceStore(tmp_path / "evidence.db")

    version = import_workbook(path, store, "EMC 2026-01")

    assert version["status"] == "draft"
    assert version["domain"] == "emc"
    assert store.list_items(version["id"])[0]["test_item"] == "CE"


def test_component_workbook_imports_replacement_rules(tmp_path):
    path = tmp_path / "rules.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "替代规则"
    sheet.append(["parameter", "rule", "critical", "derating_factor", "require_condition"])
    sheet.append(["VDS", "candidate_gte", "是", 0.8, "否"])
    workbook.save(path)
    store = EvidenceStore(tmp_path / "evidence.db")

    version = import_workbook(path, store, "MOSFET 规则")

    assert store.list_rules(version["id"])[0]["derating_factor"] == 0.8
