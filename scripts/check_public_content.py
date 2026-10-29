"""Reject common sensitive artifacts in the files selected for publication."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = {'.venv', 'node_modules', 'chroma_data', 'mock_chroma_data', 'runtime_data', 'logs', 'cache', 'uploads', 'metrics'}
BLOCKED_EXTENSIONS = {'.db', '.sqlite', '.sqlite3', '.pdf', '.docx', '.xlsx', '.pptx', '.pem', '.key', '.pfx', '.p12', '.log', '.zip'}
RULES = {
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'provider token': re.compile(r'\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})\b'),
    'personal home path': re.compile(r'(?i:[A-Za-z]:[\\/]Users[\\/][^\s/\\]+)|[/]Users/[^\s/]+'),
}

def main():
    result = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, check=False)
    if result.returncode != 0:
        print('Initialize Git and stage the intended public files before running this check.')
        return 2
    names = [p for p in result.stdout.decode('utf-8').split('\0') if p]
    if not names:
        print('No tracked files to check.')
        return 2
    errors = []
    for name in names:
        relative = Path(name)
        if set(relative.parts) & BLOCKED_DIRS or relative.suffix.lower() in BLOCKED_EXTENSIONS:
            errors.append((name, 'operational or binary data'))
        if relative.name == '.env' or relative.name.startswith('.env.') and not relative.name.startswith('.env.example'):
            errors.append((name, 'actual environment file'))
        path = ROOT / relative
        if path.is_symlink():
            errors.append((name, 'symlink requires manual review'))
            continue
        try:
            content = path.read_text(encoding='utf-8-sig')
        except (UnicodeError, OSError):
            errors.append((name, 'unreadable or non-text file'))
            continue
        for label, pattern in RULES.items():
            if pattern.search(content):
                errors.append((name, label))
    for name, label in errors:
        print(f'{name}: {label}')
    print(f'Checked {len(names)} tracked files; {len(errors)} findings.')
    return bool(errors)

if __name__ == '__main__':
    sys.exit(main())
