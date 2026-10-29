"""Consistent SQLite backup and non-content manifest; refuses overwriting backups."""
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3


def digest_rows(db, table, columns):
    quoted = '"' + table.replace('"', '""') + '"'
    fields = ",".join('"' + column.replace('"', '""') + '"' for column in columns)
    rows = db.execute(f"SELECT {fields} FROM {quoted} ORDER BY rowid").fetchall()
    return {"count": len(rows), "rows_hash": hashlib.sha256(json.dumps(rows, ensure_ascii=False, default=str).encode()).hexdigest(), "columns": columns}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    source, target = args.source.resolve(strict=True), args.target.resolve()
    manifest_path = target.with_suffix(".json")
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as db:
        if args.verify:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert all(digest_rows(db, table, entry["columns"]) == entry for table, entry in manifest.items()), "Historical records changed"
            print("Verified: all pre-migration columns and records are unchanged")
            return
        if target == source or target.exists() or manifest_path.exists():
            raise ValueError("Backup destination already exists or equals the source")
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(target) as saved:
            db.backup(saved)
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        manifest = {}
        for table in tables:
            escaped = table.replace('"', '""')
            columns = [row[1] for row in db.execute(f'PRAGMA table_info("{escaped}")')]
            manifest[table] = digest_rows(db, table, columns)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Consistent backup completed; table hashes saved without row content")


if __name__ == "__main__":
    main()
