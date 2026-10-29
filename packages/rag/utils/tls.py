"""局域网 HTTPS 启动参数校验。"""
import hashlib
from pathlib import Path
import subprocess

from config import config


def tls_options() -> dict:
    if not config.LAN_ENABLED:
        return {}
    cert, key = Path(config.TLS_CERT_FILE), Path(config.TLS_KEY_FILE)
    if not config.TLS_CERT_FILE or not config.TLS_KEY_FILE or not cert.is_file() or not key.is_file():
        raise RuntimeError("局域网模式必须配置存在的 TLS_CERT_FILE 和 TLS_KEY_FILE")
    return {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}


def ensure_self_signed_certificate(cert: Path, key: Path, common_name: str = "intern-rag.local") -> tuple[Path, Path]:
    """首次部署生成 365 天自签名证书；已有配对文件保持不变。"""
    cert, key = Path(cert), Path(key)
    if cert.is_file() and key.is_file():
        return cert, key
    cert.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "365",
        "-keyout", str(key), "-out", str(cert), "-subj", f"/CN={common_name}",
    ], check=True, capture_output=True)
    return cert, key


def certificate_fingerprint(cert: Path) -> str:
    """返回证书的 SHA-256 指纹，供团队成员在首次信任前核验。"""
    digest = hashlib.sha256(Path(cert).read_bytes()).hexdigest().upper()
    return ":".join(digest[index:index + 2] for index in range(0, len(digest), 2))
