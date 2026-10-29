"""Logical corpus/index/evidence hashes, independent of physical SQLite page layout."""
from __future__ import annotations

import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import sqlite3

from config import config


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str,
                                     separators=(",", ":")).encode()).hexdigest()


def executable_hash():
    root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ("api", "core", "db", "pipeline", "rag", "tools", "utils")
               for p in sorted((root / folder).rglob("*.py"))}
    for name in ("config.py", "requirements.txt"):
        sources[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return digest(sources)


LOADED_CODE_HASH = executable_hash()


def collection_hash(collection, page_size=100):
    count = collection.count()
    rows = []
    for offset in range(0, count, page_size):
        page = collection.get(limit=page_size, offset=offset, include=["documents", "metadatas", "embeddings"])
        for i, chunk_id in enumerate(page["ids"]):
            row = {"id": chunk_id}
            for key in ("documents", "metadatas", "embeddings"):
                values = page.get(key)
                value = values[i] if values is not None else None
                row[key] = value.tolist() if hasattr(value, "tolist") else value
            rows.append((chunk_id, digest(row)))
    if len(rows) != count or len({r[0] for r in rows}) != count or collection.count() != count:
        raise RuntimeError("corpus changed while computing snapshot")
    return {"count": count, "hash": digest(sorted(rows)), "metadata_hash": digest(collection.metadata or {})}


def evidence_hash(store):
    if store is None:
        return {"available": False}
    path = Path(store.path).resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        tables = {}
        # Only published business evidence and rules; no account, audit or draft records.
        for table in ("evidence_versions", "evidence_items", "evidence_rules"):
            query = ("SELECT * FROM evidence_versions WHERE status='published' ORDER BY id" if table == "evidence_versions" else
                     f"SELECT * FROM {table} WHERE version_id IN (SELECT id FROM evidence_versions WHERE status='published') ORDER BY id")
            cursor = db.execute(query)
            tables[table] = {"columns": [c[0] for c in cursor.description], "rows": cursor.fetchall()}
    return {"available": True, "hash": digest(tables)}


def storage_marker(chroma, evidence_store):
    # Read-only change detector across the actual persistent DB, WAL, HNSW and image files.
    root = Path(chroma._db_path).resolve()
    paths = [p for p in root.rglob("*") if p.is_file()]
    if evidence_store is not None:
        paths += [p for p in Path(evidence_store.path).resolve().parent.glob(Path(evidence_store.path).name + "*") if p.is_file()]
    return [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(set(paths))]


def image_hashes(chroma):
    root = Path(chroma._db_path).resolve() / "images"
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            if not path.resolve().is_relative_to(root.resolve()):
                raise RuntimeError("image reference escapes corpus root")
            with path.open("rb") as stream:
                hashes[str(path.relative_to(root))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"count": len(hashes), "hash": digest(hashes)}


def runtime_snapshot(chroma, evidence_store=None):
    if chroma is None or not chroma.is_connected:
        raise RuntimeError("RAG index is unavailable; cannot freeze corpus")
    before = storage_marker(chroma, evidence_store)
    # Hash chunks, source/version metadata AND vectors, including image chunks.
    collections = {name: collection_hash(getattr(chroma, attr)) for name, attr in
                   (("documents", "_doc_collection"), ("images", "_image_collection"))}
    excluded = ("API_KEY", "PASSWORD", "API_TOKEN", "URL", "HOST", "PATH", "DIR", "FILE", "USERNAME")
    settings = {key: getattr(config, key) for key in dir(type(config)) if key.isupper() and not any(part in key for part in excluded)}
    settings["endpoint_hash"] = digest({key: getattr(config, key) for key in ("ONEAPI_BASE_URL", "CHAT_BASE_URL", "RERANK_BASE_URL")})
    code_hash = executable_hash()
    if code_hash != LOADED_CODE_HASH:
        raise RuntimeError("RAG source changed after process startup; restart before freezing a new baseline")
    dependencies = {}
    for name in ("chromadb", "openai", "httpx", "numpy", "rank-bm25", "jieba"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = "unavailable"
    content = {"collections": collections, "images": image_hashes(chroma), "evidence": evidence_hash(evidence_store),
               "configuration": settings, "dependencies": dependencies, "code_hash": code_hash, "cache_policy": "run_and_corpus_scoped"}
    if before != storage_marker(chroma, evidence_store):
        raise RuntimeError("corpus changed while computing snapshot")
    return {"version": 1, "corpus_hash": digest(content), **content}
