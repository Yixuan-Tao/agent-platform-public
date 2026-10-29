"""本地离线资料包工具。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.pack import validate_pack


def import_pack(directory: Path, *, dry_run: bool = False) -> dict:
    """校验后顺序摄入资料包，失败文件不会被伪装为成功。"""
    manifest = validate_pack(directory)
    files = manifest.get("files", [])
    if dry_run:
        return {"version": manifest["version"], "status": "validated", "files": len(files)}

    # 延迟导入，确保 ``pack validate`` 不需要加载 ChromaDB 或模型客户端。
    from pipeline.ingest import IngestPipeline

    pipeline = IngestPipeline()
    results = []
    for item in files:
        path = Path(directory) / item["path"]
        result = pipeline.ingest_file(str(path))
        results.append({"path": item["path"], "status": result.get("status", "unknown")})
        if result.get("status") != "ok":
            return {"version": manifest["version"], "status": "failed", "results": results}
    return {"version": manifest["version"], "status": "ok", "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(prog="pack")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("directory", type=Path)
    import_command = commands.add_parser("import", help="校验后摄入资料包")
    import_command.add_argument("directory", type=Path)
    import_command.add_argument("--dry-run", action="store_true", help="仅校验，不写入索引")
    args = parser.parse_args()
    if args.command == "validate":
        try:
            print(json.dumps(validate_pack(args.directory), ensure_ascii=False))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.command == "import":
        try:
            result = import_pack(args.directory, dry_run=args.dry_run)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["status"] in {"ok", "validated"} else 1
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
