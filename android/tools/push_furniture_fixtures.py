"""Provision only public test fixtures to the debuggable app through authorized ADB."""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="D:/Android/Sdk/platform-tools/adb.exe")
    parser.add_argument(
        "--fixtures", type=Path, default=Path("datasets/imported/ade20k-furniture-v06")
    )
    parser.add_argument("--extra", type=Path, action="append", default=[])
    args = parser.parse_args()
    manifest = json.loads((args.fixtures / "manifest.json").read_text(encoding="utf-8"))
    archive = args.fixtures / "adb-fixtures.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(args.fixtures / "manifest.json", arcname="manifest.json")
        for extra in args.extra:
            if extra.name == "manifest.json":
                raise ValueError("Extra files cannot replace the dataset manifest")
            tar.add(extra, arcname=extra.name)
        for case in manifest["cases"]:
            for key in ("image", "mask"):
                name = case[key]
                if Path(name).name != name:
                    raise ValueError("Fixture names must be simple filenames")
                tar.add(args.fixtures / name, arcname=name)
    command = [args.adb, "-s", args.serial]
    remote = "/data/local/tmp/sightguard-furniture-v06.tar"
    subprocess.run([*command, "push", str(archive), remote], check=True)
    subprocess.run(
        [*command, "shell", "run-as", "org.sightguard.app", "mkdir", "-p", "files/furniture"],
        check=True,
    )
    # run-as is the app's standard debug identity, not root or a permission bypass.
    subprocess.run(
        [
            *command,
            "shell",
            f"cat {remote} | run-as org.sightguard.app tar -xf - -C files/furniture",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
