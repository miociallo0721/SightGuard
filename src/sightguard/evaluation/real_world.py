"""Bounded acquisition and video provenance; evaluation stays in Phase 3."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from sightguard.evaluation.dataset import sha256_file

OBJECT_LIMIT = 500 * 1024**2


def write_record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def download_object(
    url: str,
    target: Path,
    *,
    expected_size: int,
    expected_md5: str,
    approved_large: bool = False,
) -> dict:
    """Known size and publisher checksum required; never publish a partial download."""
    if expected_size <= 0 or (expected_size > OBJECT_LIMIT and not approved_large):
        raise ValueError("Unknown/oversize object needs explicit approval")
    if not url.startswith("https://") or len(expected_md5) != 32:
        raise ValueError("HTTPS and publisher MD5 required")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    status = target.with_suffix(target.suffix + ".download.json")
    record = {"url": url, "expected_size": expected_size, "expected_md5": expected_md5}
    if target.exists():
        raise FileExistsError("Existing raw object preserved; verify separately")
    if partial.exists():
        raise FileExistsError("Existing partial object preserved; no implicit overwrite")
    try:
        md5 = hashlib.md5(usedforsecurity=False)
        count = 0
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("xb") as stream:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected_size:
                raise ValueError("Publisher length differs from expected object size")
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > expected_size:
                    raise ValueError("Download exceeded approved size")
                stream.write(chunk)
                md5.update(chunk)
        if count != expected_size or md5.hexdigest() != expected_md5:
            raise ValueError("Incomplete download or publisher checksum mismatch")
        partial.rename(target)
        record.update(status="verified", bytes=count, sha256=sha256_file(target))
        return record
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_record(status, record)


def extract_member(archive: Path, member: str, target: Path) -> None:
    """Extract only an explicitly selected safe member; never unpack the entire dataset."""
    relative = PurePosixPath(member)
    if relative.is_absolute() or ".." in relative.parts or "\\" in member or ":" in member:
        raise ValueError("Unsafe archive member")
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(archive) as container:
        info = container.getinfo(member)
        if info.is_dir():
            raise ValueError("A regular file is required")
        target.parent.mkdir(parents=True, exist_ok=True)
        with container.open(info) as source, target.open("xb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)


def inspect_video(path: Path) -> dict:
    """Fully decode, preserve presentation timestamps, and surface quality limitations."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Missing valid encoded video FPS")
        timestamps, hashes, motion = [], [], []
        previous = None
        width = height = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)))
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 48))
            if previous is not None:
                motion.append(float(np.abs(small.astype(float) - previous).mean()))
            previous = small.astype(float)
        if not timestamps or (expected > 0 and len(timestamps) != expected):
            raise ValueError(f"Decode count mismatch: {len(timestamps)} / {expected}")
        if len(timestamps) > 1 and any(
            b <= a for a, b in zip(timestamps, timestamps[1:], strict=False)
        ):
            raise ValueError("Presentation timestamps are not strictly increasing")
        return {
            "frames": len(timestamps),
            "width": width,
            "height": height,
            "fps": fps,
            "duration_ms": timestamps[-1] - timestamps[0] + 1000 / fps,
            "timestamps_ms": timestamps,
            "sha256": sha256_file(path),
            "decoded_pixels_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
            "exact_duplicate_frame_fraction": 1 - len(set(hashes)) / len(hashes),
            "mean_small_frame_difference": sum(motion) / len(motion) if motion else None,
            "manual_review": "pending; motion/cut/privacy suitability requires visual review",
        }
    finally:
        capture.release()


def clip_video(
    source: Path, target: Path, start_s: float, end_s: float, *, source_info: dict | None = None
) -> dict:
    """Keep video timing via FFmpeg passthrough; remove audio and verify the full result."""
    import imageio_ffmpeg

    if not (math.isfinite(start_s) and math.isfinite(end_s) and 0 <= start_s < end_s):
        raise ValueError("Invalid clip interval")
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-n",
        "-ss",
        str(start_s),
        "-i",
        str(source),
        "-t",
        str(end_s - start_s),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-fps_mode",
        "passthrough",
        str(target),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode:
        raise RuntimeError(process.stderr[-4000:])
    result = inspect_video(target)
    info = source_info if source_info is not None else inspect_video(source)
    selected = [t for t in info["timestamps_ms"] if start_s * 1000 <= t < end_s * 1000]
    if len(selected) != result["frames"]:
        raise ValueError("Source/output frame count differs; cannot claim an exact time mapping")
    offset = selected[0] - result["timestamps_ms"][0]
    if any(
        abs(src - dst - offset) > 1
        for src, dst in zip(selected, result["timestamps_ms"], strict=True)
    ):
        raise ValueError("Source/output PTS differ beyond 1 ms; review conversion")
    result.update(
        selected_start_time=start_s,
        selected_end_time=end_s,
        time_mapping={
            "basis": "source MP4 presentation time, not sensor host clock",
            "source_ms": selected,
            "requested_offset_ms": start_s * 1000,
            "actual_offset_ms": offset,
            "precision_note": "Source PTS matched one-to-one; relative delta tolerance 1 ms",
        },
        processing={"command": command, "audio_removed": True, "fps_resampling": False},
    )
    return result


def group_split(rows: list[dict], *, seed: int, test_fraction: float = 0.0) -> dict[str, str]:
    """Split by parent recording; first batch defaults entirely to development."""
    if not 0 <= test_fraction < 1:
        raise ValueError("Invalid test fraction")
    groups = sorted({row["parent_source_id"] for row in rows})
    random.Random(seed).shuffle(groups)
    count = int(len(groups) * test_fraction)
    test = set(groups[:count])
    return {
        row["video_id"]: "test" if row["parent_source_id"] in test else "development"
        for row in rows
    }


def validate_manifest(rows: list[dict[str, Any]]) -> None:
    """Accepted requires real local evidence; annotations remain independently reviewed."""
    seen, splits, hashes = set(), {}, set()
    for row in rows:
        if row.get("schema_version") != "sightguard.real-world.v0.1":
            raise ValueError("Unsupported provenance schema")
        if row["video_id"] in seen:
            raise ValueError("Duplicate video_id")
        seen.add(row["video_id"])
        fingerprint = row.get("processed_sha256")
        if fingerprint:
            if fingerprint in hashes:
                raise ValueError("Duplicate processed video content")
            hashes.add(fingerprint)
        if row["annotation_status"] not in {"pending", "draft", "reviewed"}:
            raise ValueError("Invalid annotation status")
        parent = row["parent_source_id"]
        if parent in splits and splits[parent] != row["split"]:
            raise ValueError("Parent recording split leakage")
        splits[parent] = row["split"]
        if row["eligibility_status"] == "accepted":
            path = Path(row["filepath"])
            if row["acquisition_status"] != "verified" or not path.is_file():
                raise ValueError("Accepted video must have verified local acquisition")
            if sha256_file(path) != row["processed_sha256"]:
                raise ValueError("Processed SHA256 mismatch")
            if row["review_status"] != "visual_reviewed":
                raise ValueError("Accepted video needs visual quality review")
