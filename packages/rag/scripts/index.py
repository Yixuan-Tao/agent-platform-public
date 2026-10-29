"""本机索引维护命令。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.chroma_client import ChromaClientSingleton
from scripts.pack import import_pack
from utils.pack import validate_pack


def main() -> int:
    parser = argparse.ArgumentParser(prog="index")
    commands = parser.add_subparsers(dest="command", required=True)
    rebuild = commands.add_parser("rebuild", help="清空本机索引后从资料包重建")
    rebuild.add_argument("directory", type=Path)
    rebuild.add_argument("--confirm", action="store_true", help="确认清空当前本机索引")
    args = parser.parse_args()

    if args.command == "rebuild":
        if not args.confirm:
            parser.error("索引重建会清空当前本机索引；请追加 --confirm")
        try:
            validate_pack(args.directory)  # 先校验，校验失败绝不清空旧索引
            removed = ChromaClientSingleton.get_instance().clear_all()
            result = import_pack(args.directory)
            print(json.dumps({"removed_chunks": removed, **result}, ensure_ascii=False))
            return 0 if result["status"] == "ok" else 1
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
