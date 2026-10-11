"""Download pinned official models and a modest TUM RGB-D subset, serially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from sightguard.research.resources import check_resources, lower_own_priority
from sightguard.research.settings import research_cache_root

ROOT = research_cache_root()
SOURCES = [
    (
        "hypersim",
        "3bc65d4e14a6786a61acec16453c50e12bf5f338",
        "b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545",
    ),
    (
        "vkitti",
        "c725b8589bdf6ab04072cab74c0467830db80d6d",
        "9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb",
    ),
]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch(url: str, name: str, expected: str | None = None) -> dict:
    check_resources(output_dir=ROOT)
    path = ROOT / name
    if not path.exists() or (expected and digest(path) != expected):
        partial = path.with_suffix(path.suffix + ".partial")
        remaining = (
            datetime.fromisoformat(
                os.getenv("SIGHTGUARD_RESEARCH_FREEZE_UTC", "2026-10-11T22:00:00+00:00")
            )
            - datetime.now(UTC)
        ).total_seconds()
        if remaining < 30:
            raise RuntimeError("feature_freeze_prevents_new_download")
        if name.endswith(".tgz"):
            ranged_download(url, partial)
        else:
            subprocess.run(
                [
                    "curl.exe",
                    "-fL",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "2",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    str(int(min(900, remaining - 5))),
                    "--limit-rate",
                    "8M",
                    url,
                    "-o",
                    str(partial),
                ],
                check=True,
                timeout=min(910, remaining),
            )
        actual = digest(partial)
        if expected and actual != expected:
            raise ValueError(f"SHA256 mismatch: {name}: {actual}")
        partial.replace(path)
    record = {
        "url": url,
        "path": str(path),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "retrieved_utc": datetime.now(UTC).isoformat(),
    }
    print(json.dumps(record), flush=True)
    return record


def ranged_download(url: str, partial: Path) -> None:
    """Bound each request and disk copy; validate Content-Range before appending."""
    head = subprocess.run(
        ["curl.exe", "--http1.1", "-fILsS", "--max-time", "25", url],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    lengths = re.findall(r"(?im)^Content-Length:\s*(\d+)", head)
    if not lengths:
        raise ValueError("archive_length_missing")
    total = int(lengths[-1])
    chunk = partial.with_suffix(".chunk")
    headers = partial.with_suffix(".headers")
    # This task owns the partial file; each appended range already passed header/length checks.
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total or (offset != total and offset % (8 * 1024**2)):
        raise ValueError("partial_archive_not_on_verified_range_boundary")
    with partial.open("ab") as output:
        for start in range(offset, total, 8 * 1024**2):
            check_resources(output_dir=ROOT)
            end = min(total - 1, start + 8 * 1024**2 - 1)
            subprocess.run(
                [
                    "curl.exe",
                    "--http1.1",
                    "-fLsS",
                    "--retry",
                    "2",
                    "--retry-all-errors",
                    "--max-time",
                    "90",
                    "--connect-timeout",
                    "15",
                    "--limit-rate",
                    "8M",
                    "--range",
                    f"{start}-{end}",
                    "--max-filesize",
                    str(end - start + 1),
                    "-D",
                    str(headers),
                    url,
                    "-o",
                    str(chunk),
                ],
                check=True,
                timeout=290,
            )
            wanted = f"content-range: bytes {start}-{end}/{total}"
            if wanted not in headers.read_text().lower() or chunk.stat().st_size != end - start + 1:
                raise ValueError("archive_content_range_mismatch")
            with chunk.open("rb") as source:
                while block := source.read(1024**2):
                    output.write(block)
            output.flush()
            print(f"archive_progress {partial.name} {end + 1}/{total}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["freiburg1_xyz", "freiburg1_rpy", "freiburg3_nostructure_notexture_far"],
    )
    args = parser.parse_args()
    lower_own_priority()
    ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for domain, revision, sha in SOURCES:
        model_id = f"depth-anything/Depth-Anything-V2-Metric-{domain.title()}-Small"
        if domain == "vkitti":
            model_id = "depth-anything/Depth-Anything-V2-Metric-VKITTI-Small"
        base = f"https://huggingface.co/{model_id}/resolve/{revision}"
        name = f"depth_anything_v2_metric_{domain}_vits.pth"
        records.append(fetch(f"{base}/{name}", name, sha))
        records.append(fetch(f"{base}/README.md", f"{domain}-model-card.md"))
    for selection in args.sequences:
        group, sequence = selection.split("_", 1)
        if group not in ("freiburg1", "freiburg2", "freiburg3"):
            raise ValueError("invalid TUM sequence group")
        name = f"rgbd_dataset_{group}_{sequence}.tgz"
        records.append(fetch(f"https://cvg.cit.tum.de/rgbd/dataset/{group}/{name}", name))
    for name, url in [
        ("tum-file-formats.html", "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"),
        ("tum-license.html", "https://cvg.cit.tum.de/data/datasets/rgbd-dataset"),
    ]:
        records.append(fetch(url, name))
    provenance = ROOT / "download-provenance.json"
    existing = json.loads(provenance.read_text()) if provenance.exists() else []
    combined = {item["url"]: item for item in existing + records}
    provenance.write_text(json.dumps(list(combined.values()), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
