"""Fetch a bounded, ground-truth-selected ADE20K validation subset, never model-selected.

Requires Pillow from the existing project environment. HTTP Range avoids downloading
the 967 MB training+validation archive. Zip CRC and per-file SHA256 are verified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

URL = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
GROUPS = {
    "TABLE": {15, 33, 64},
    "CHAIR": {19, 30, 31, 69, 75, 110},
    "CABINET": {10, 35},
    "SOFA": {23},
    "BED": {7},
    "SHELF": {24},
    "COUNTER": {45, 70},
    "APPLIANCE": {50},
}


def open_retry(request: urllib.request.Request, timeout: int = 40):
    """Retry transient transport errors without relaxing TLS verification."""
    for attempt in range(4):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


class RangeArchive(io.RawIOBase):
    """Small read/seek adapter, refusing unexpected full-archive responses."""

    def __init__(self) -> None:
        super().__init__()
        with open_retry(urllib.request.Request(URL, method="HEAD")) as response:
            self.length = int(response.headers["Content-Length"])
            self.etag = response.headers["ETag"]
        self.position = 0
        self.downloaded = 0
        self.blocks: dict[int, bytes] = {}

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset + (
            self.position if whence == 1 else self.length if whence == 2 else 0
        )
        return self.position

    def tell(self) -> int:
        return self.position

    def seekable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        size = min(self.length - self.position, size if size >= 0 else self.length)
        if size <= 0:
            return b""
        if size > 8_000_000:
            raise ValueError("Refusing unexpectedly large ZIP read")
        start = self.position
        block_size = 262144
        block_start = start // block_size * block_size
        use_block = size <= block_size and start + size <= block_start + block_size
        if use_block and block_start in self.blocks:
            self.position += size
            return self.blocks[block_start][start - block_start : start - block_start + size]
        request_start = block_start if use_block else start
        request_size = min(block_size, self.length - block_start) if use_block else size
        request = urllib.request.Request(
            URL,
            headers={
                "Range": f"bytes={request_start}-{request_start + request_size - 1}",
                "If-Range": self.etag,
            },
        )
        with open_retry(request) as response:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
                f"bytes {request_start}-"
            ):
                raise ValueError("Server did not honor byte range; refusing bulk download")
            data = response.read(request_size + 1)
        if len(data) != request_size:
            raise ValueError("Incomplete archive range")
        self.position += size
        self.downloaded += request_size
        if use_block:
            if len(self.blocks) >= 8:
                del self.blocks[next(iter(self.blocks))]
            self.blocks[block_start] = data
            return data[start - block_start : start - block_start + size]
        return data


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--scan-count", type=int, default=160)
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--negative-count", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=root / "datasets/imported/ade20k-furniture-v06"
    )
    args = parser.parse_args()
    if not (
        0 <= args.start_index < 2000
        and 1 <= args.scan_count <= 1000
        and 1 <= args.per_group <= 30
        and 1 <= args.negative_count <= 50
    ):
        parser.error("Choose a bounded validation slice and sample count")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    archive = RangeArchive()
    selected: list[dict] = []
    counts: Counter[str] = Counter()
    scanned: list[dict] = []
    with zipfile.ZipFile(archive) as zf:
        masks = sorted(
            n for n in zf.namelist() if "/annotations/validation/" in n and n.endswith(".png")
        )[args.start_index : args.start_index + args.scan_count]
        for index, name in enumerate(masks):
            mask_path = output / Path(name).name
            data = mask_path.read_bytes() if mask_path.exists() else zf.read(name)
            mask_path.write_bytes(data)
            with Image.open(io.BytesIO(data)) as mask:
                histogram = mask.histogram()
                pixels = mask.width * mask.height
                fractions = {
                    group: sum(histogram[i + 1] for i in ids) / pixels
                    for group, ids in GROUPS.items()
                }
            present = [g for g, value in fractions.items() if value >= 0.015]
            requested = [g for g in present if counts[g] < args.per_group]
            negative = not present and counts["NEGATIVE"] < args.negative_count
            scanned.append(
                {
                    "mask": Path(name).name,
                    "fractions": fractions,
                    "selected": bool(requested or negative),
                }
            )
            if requested or negative:
                image_name = name.replace("/annotations/", "/images/").replace(".png", ".jpg")
                image_path = output / Path(image_name).name
                image_data = image_path.read_bytes() if image_path.exists() else zf.read(image_name)
                image_path.write_bytes(image_data)
                groups = requested or ["NEGATIVE"]
                for group in groups:
                    counts[group] += 1
                selected.append(
                    {
                        "image": image_path.name,
                        "mask": mask_path.name,
                        "groups": groups,
                        "gt_fractions": fractions,
                        "image_sha256": hashlib.sha256(image_data).hexdigest(),
                        "mask_sha256": hashlib.sha256(data).hexdigest(),
                        "split": "validation",
                    }
                )
                print(f"selected {image_path.name} {groups}", flush=True)
            if index % 10 == 0:
                print(
                    f"scanned {index + 1} downloaded={archive.downloaded} counts={dict(counts)}",
                    flush=True,
                )
            if (
                all(counts[g] >= args.per_group for g in GROUPS)
                and counts["NEGATIVE"] >= args.negative_count
            ):
                break
    manifest = {
        "schema": "sightguard.furniture-eval.v1",
        "source": URL,
        "archive_etag": archive.etag,
        "archive_bytes": archive.length,
        "downloaded_bytes": archive.downloaded,
        "license_reference": "https://github.com/CSAILVision/sceneparsing/blob/master/LICENSE",
        "license": "BSD-3-Clause per benchmark repository and dataset card; images kept local",
        "annotation_contract": "0=ignore; 1..150 maps to logits 0..149 by subtracting one",
        "selection_parameters": {
            "start_index": args.start_index,
            "scan_count": args.scan_count,
            "per_group": args.per_group,
            "negative_count": args.negative_count,
        },
        "selection": (
            "Lexically ordered bounded validation slice, first N per group >=1.5% GT area, "
            "first N images without furniture above that area. Selected using GT only before "
            "inference. Not a full benchmark or independent external-domain validation."
        ),
        "counts": dict(counts),
        "cases": selected,
        "scanned": scanned,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"complete {len(selected)} cases; {archive.downloaded} bytes", flush=True)


if __name__ == "__main__":
    main()
