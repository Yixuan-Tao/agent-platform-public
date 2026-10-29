"""为局域网部署生成自签名 TLS 证书。"""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.tls import ensure_self_signed_certificate


def main() -> int:
    parser = argparse.ArgumentParser(description="生成局域网 HTTPS 自签名证书")
    parser.add_argument("--cert", type=Path, default=Path("runtime_data/tls/server.crt"))
    parser.add_argument("--key", type=Path, default=Path("runtime_data/tls/server.key"))
    parser.add_argument("--common-name", default="intern-rag.local")
    args = parser.parse_args()
    cert, key = ensure_self_signed_certificate(args.cert, args.key, args.common_name)
    print(f"证书: {cert}\n私钥: {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
