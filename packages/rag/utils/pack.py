"""离线资料包清单校验。"""
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pack(directory: Path) -> dict:
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("缺少 manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("version"), str) or not manifest["version"]:
        raise ValueError("manifest 缺少 version")
    for item in manifest.get("files", []):
        relative = Path(item.get("path", ""))
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"资料文件不存在: {relative}")
        if _sha256(path) != item.get("sha256"):
            raise ValueError(f"SHA-256 不匹配: {relative}")
    return manifest
