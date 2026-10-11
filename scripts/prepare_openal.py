"""Explicitly download the official Windows x64 OpenAL Soft runtime and license."""

import hashlib
import io
import json
import platform
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

VERSION = "1.25.2"
URL = (
    f"https://github.com/kcat/openal-soft/releases/download/{VERSION}/openal-soft-{VERSION}-bin.zip"
)


def main() -> None:
    if platform.system() != "Windows" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError(
            "This helper prepares Windows x64 only; use system OpenAL Soft elsewhere"
        )
    target = Path(__file__).resolve().parents[1] / "models" / "openal-soft"
    target.mkdir(parents=True, exist_ok=True)
    runtime = target / "soft_oal.dll"
    if runtime.exists():
        print(f"Existing runtime retained: {runtime}")
        return
    with urlopen(URL, timeout=60) as response:
        archive = response.read()
    prefix = f"openal-soft-{VERSION}-bin/"
    with ZipFile(io.BytesIO(archive)) as package:
        dll = package.read(prefix + "bin/Win64/soft_oal.dll")
        license_text = package.read(prefix + "COPYING")
    (target / "COPYING").write_bytes(license_text)
    runtime.write_bytes(dll)
    metadata = {
        "version": VERSION,
        "source_url": URL,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "dll_sha256": hashlib.sha256(dll).hexdigest(),
        "license": "LGPL-2.0-or-later",
    }
    (target / "runtime_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {runtime}; original COPYING and runtime_metadata.json retained")


if __name__ == "__main__":
    main()
