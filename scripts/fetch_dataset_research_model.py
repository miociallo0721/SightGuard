"""Fetch pinned official Metric3Dv2 Small weights/source with bounded resources."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from sightguard.research.resources import check_resources, lower_own_priority

REVISION = "80d2d1410afb4b23cd9d18c6be9144483d4b70b6"
SOURCE_SHA = "eb5b6fac0dc155e4e52f576e304fbf11655ff339"
WEIGHT_SHA = "b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6"
WEIGHT_BYTES = 150120967
WEIGHT_NAME = "metric_depth_vit_small_800k.pth"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--stop-utc", required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    lower_own_priority()
    check_resources(stop_utc=args.stop_utc, output_dir=args.cache)
    weight = args.cache / WEIGHT_NAME
    url = f"https://huggingface.co/JUGGHM/Metric3D/resolve/{REVISION}/{WEIGHT_NAME}"
    if not weight.exists():
        part = weight.with_suffix(".pth.part")
        if part.exists():
            raise FileExistsError("preserve partial download; choose a new cache or inspect it")
        subprocess.run(
            [
                "curl.exe",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "15",
                "--max-time",
                "120",
                "--limit-rate",
                "8M",
                "--max-filesize",
                str(WEIGHT_BYTES),
                "--output",
                str(part),
                url,
            ],
            check=True,
            timeout=130,
        )
        if part.stat().st_size != WEIGHT_BYTES or digest(part) != WEIGHT_SHA:
            raise ValueError("official LFS size or SHA256 mismatch; partial retained")
        part.rename(weight)
    if weight.stat().st_size != WEIGHT_BYTES or digest(weight) != WEIGHT_SHA:
        raise ValueError("cached model checksum mismatch")
    source = args.cache / "Metric3D"
    if not source.exists():
        source.mkdir()
        commands = [
            ["git", "init"],
            ["git", "remote", "add", "origin", "https://github.com/YvanYin/Metric3D.git"],
            ["git", "sparse-checkout", "init", "--cone"],
            ["git", "sparse-checkout", "set", "mono"],
            ["git", "fetch", "--depth", "1", "origin", SOURCE_SHA],
            ["git", "checkout", "--detach", "FETCH_HEAD"],
        ]
        for command in commands:
            check_resources(stop_utc=args.stop_utc, output_dir=args.cache)
            subprocess.run(command, cwd=source, check=True, timeout=60)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != SOURCE_SHA:
        raise ValueError("source revision differs; do not overwrite existing checkout")
    record = {
        "source": "https://github.com/YvanYin/Metric3D",
        "source_sha": actual,
        "source_code_license": "BSD-2-Clause",
        "weights_url": url,
        "weights_sha256": WEIGHT_SHA,
        "bytes": WEIGHT_BYTES,
        "publisher_integrity": "HF LFS sha256 verified",
        "weight_redistribution_license": "not_independently_verified_no_APK_distribution",
    }
    (args.cache / "metric3d-source-manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
