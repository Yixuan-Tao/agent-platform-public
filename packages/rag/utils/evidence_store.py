"""人工确认的 EMC 与器件证据版本库。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4
from typing import Any


@dataclass
class EvidenceItem:
    subject_id: str
    domain: str
    field_name: str
    value_raw: str
    value_normalized: Any = None
    unit: str = ""
    value_kind: str = "enum"
    test_condition: str = ""
    source_file: str = ""
    page: int = 0
    section: str = ""
    evidence_status: str = "confirmed"
    test_item: str = ""
    component_family: str = ""
    risk_level: str = ""
    evaluation_note: str = ""


class EvidenceStore:
    """SQLite 持久化的草稿/发布证据版本；运行时只读取 published。"""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS evidence_versions ("
                "id TEXT PRIMARY KEY, name TEXT NOT NULL, domain TEXT NOT NULL, "
                "status TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS evidence_items ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, version_id TEXT NOT NULL, "
                "subject_id TEXT NOT NULL, domain TEXT NOT NULL, field_name TEXT NOT NULL, "
                "value_raw TEXT NOT NULL, value_normalized TEXT, unit TEXT, value_kind TEXT, "
                "test_condition TEXT, source_file TEXT, page INTEGER, section TEXT, "
                "evidence_status TEXT, test_item TEXT, component_family TEXT, "
                "risk_level TEXT, evaluation_note TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS evidence_rules ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, version_id TEXT NOT NULL, rule_json TEXT NOT NULL)"
            )
            self._ensure_optional_columns(db)

    def _ensure_optional_columns(self, db: sqlite3.Connection) -> None:
        """旧库迁移：为 evidence_items 补齐可选列（向后兼容）。"""
        columns = {row[1] for row in db.execute("PRAGMA table_info(evidence_items)").fetchall()}
        if "risk_level" not in columns:
            db.execute("ALTER TABLE evidence_items ADD COLUMN risk_level TEXT DEFAULT ''")
        if "evaluation_note" not in columns:
            db.execute("ALTER TABLE evidence_items ADD COLUMN evaluation_note TEXT DEFAULT ''")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def create_version(self, name: str, domain: str) -> dict:
        if domain not in {"emc", "component"}:
            raise ValueError("domain 必须为 emc 或 component")
        if not name.strip():
            raise ValueError("版本名称不能为空")
        version = {"id": uuid4().hex, "name": name.strip(), "domain": domain, "status": "draft"}
        with self._connect() as db:
            db.execute(
                "INSERT INTO evidence_versions VALUES (?, ?, ?, ?, ?)",
                (version["id"], version["name"], domain, "draft", time.time()),
            )
        return version

    def get_version(self, version_id: str) -> dict:
        with self._connect() as db:
            row = db.execute(
                "SELECT id,name,domain,status,created_at FROM evidence_versions WHERE id=?", (version_id,),
            ).fetchone()
        if not row:
            raise KeyError("证据版本不存在")
        return dict(zip(("id", "name", "domain", "status", "created_at"), row))

    def list_versions(self, domain: str | None = None) -> list[dict]:
        query = "SELECT id,name,domain,status,created_at FROM evidence_versions"
        args: tuple = ()
        if domain:
            query += " WHERE domain=?"
            args = (domain,)
        query += " ORDER BY created_at DESC"
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [dict(zip(("id", "name", "domain", "status", "created_at"), row)) for row in rows]

    def add_items(self, version_id: str, items: list[EvidenceItem]) -> None:
        version = self.get_version(version_id)
        if version["status"] != "draft":
            raise ValueError("仅草稿版本可以写入证据")
        rows = []
        for item in items:
            if item.domain != version["domain"] or not item.subject_id or not item.field_name or not item.value_raw:
                raise ValueError("证据的领域、对象、字段名和原始值不能为空")
            raw = asdict(item)
            rows.append((
                version_id, raw["subject_id"], raw["domain"], raw["field_name"], raw["value_raw"],
                json.dumps(raw["value_normalized"], ensure_ascii=False), raw["unit"], raw["value_kind"],
                raw["test_condition"], raw["source_file"], int(raw["page"] or 0), raw["section"],
                raw["evidence_status"], raw["test_item"], raw["component_family"],
                raw["risk_level"], raw["evaluation_note"],
            ))
        with self._connect() as db:
            db.executemany(
                "INSERT INTO evidence_items (version_id,subject_id,domain,field_name,value_raw,value_normalized,unit,"
                "value_kind,test_condition,source_file,page,section,evidence_status,test_item,component_family,"
                "risk_level,evaluation_note) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows,
            )

    def list_items(self, version_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT subject_id,domain,field_name,value_raw,value_normalized,unit,value_kind,test_condition,"
                "source_file,page,section,evidence_status,test_item,component_family,risk_level,evaluation_note "
                "FROM evidence_items WHERE version_id=?",
                (version_id,),
            ).fetchall()
        names = ("subject_id", "domain", "field_name", "value_raw", "value_normalized", "unit", "value_kind",
                 "test_condition", "source_file", "page", "section", "evidence_status", "test_item",
                 "component_family", "risk_level", "evaluation_note")
        result = []
        for row in rows:
            item = dict(zip(names, row))
            item["value_normalized"] = json.loads(item["value_normalized"]) if item["value_normalized"] else None
            item["citation"] = {"file_name": item["source_file"], "page": item["page"], "section": item["section"]}
            result.append(item)
        return result

    def publish(self, version_id: str) -> None:
        version = self.get_version(version_id)
        if version["status"] != "draft":
            raise ValueError("只能发布草稿版本")
        with self._connect() as db:
            db.execute("UPDATE evidence_versions SET status='archived' WHERE domain=? AND status='published'", (version["domain"],))
            db.execute("UPDATE evidence_versions SET status='published' WHERE id=?", (version_id,))

    def activate(self, version_id: str) -> None:
        """将已归档版本回退为当前领域唯一的发布版本。"""
        version = self.get_version(version_id)
        if version["status"] not in {"archived", "published"}:
            raise ValueError("仅已归档或已发布版本可以启用")
        with self._connect() as db:
            db.execute("UPDATE evidence_versions SET status='archived' WHERE domain=? AND status='published'", (version["domain"],))
            db.execute("UPDATE evidence_versions SET status='published' WHERE id=?", (version_id,))

    def add_rules(self, version_id: str, rules: list[dict]) -> None:
        version = self.get_version(version_id)
        if version["domain"] != "component" or version["status"] != "draft":
            raise ValueError("仅器件草稿版本可以写入替代规则")
        for rule in rules:
            if not rule.get("parameter") or rule.get("rule", "equal") not in {"equal", "candidate_gte", "candidate_lte"}:
                raise ValueError("替代规则缺少 parameter 或 rule 无效")
        with self._connect() as db:
            db.executemany("INSERT INTO evidence_rules (version_id,rule_json) VALUES (?,?)", [
                (version_id, json.dumps(rule, ensure_ascii=False)) for rule in rules
            ])

    def list_rules(self, version_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT rule_json FROM evidence_rules WHERE version_id=? ORDER BY id", (version_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def published_version(self, domain: str) -> dict | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT id,name,domain,status,created_at FROM evidence_versions WHERE domain=? AND status='published'",
                (domain,),
            ).fetchone()
        return dict(zip(("id", "name", "domain", "status", "created_at"), row)) if row else None
