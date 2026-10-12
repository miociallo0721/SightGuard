"""Check repository-local Markdown links without network or model dependencies."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def check_documents(include_archive: bool = False) -> tuple[int, list[str]]:
    """Return document count and missing/out-of-tree local link descriptions."""
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).splitlines()
    failures: list[str] = []
    checked = 0
    for name in sorted(set(names)):
        path = ROOT / name
        if path.suffix != ".md" or not path.is_file():
            continue
        if not include_archive and "/archive/" in name:
            continue
        checked += 1
        content = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8-sig"), flags=re.S)
        links = re.findall(r"\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)", content)
        links += re.findall(r'<(?:img|a)\b[^>]*(?:src|href)="([^"]+)"', content)
        for link in links:
            parsed = urlsplit(link.strip("<>"))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (path.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():
                failures.append(f"{name}: {link}")
    return checked, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Include archived historical reports")
    args = parser.parse_args()
    checked, failures = check_documents(args.all)
    for failure in failures:
        print(f"BROKEN {failure}")
    print(f"Checked {checked} Markdown files; {len(failures)} broken local links")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
