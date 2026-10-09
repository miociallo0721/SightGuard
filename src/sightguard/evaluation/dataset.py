"""Dataset provenance, split leakage, temporal coverage, and independent label validation."""

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from sightguard.data import FrameRecord, replay_records
from sightguard.evaluation.schema import VideoAnnotations, load_annotations

MANIFEST_FIELDS = (
    "video_id",
    "split",
    "source_group_id",
    "scene",
    "data_kind",
    "duration_ms",
    "fps",
    "video_path",
    "tracks_path",
    "annotations_path",
    "video_sha256",
    "tracks_sha256",
    "annotations_sha256",
)


class DatasetError(ValueError):
    """Missing or invalid dataset evidence must never become an empty negative sample."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VideoEntry:
    video_id: str
    split: str
    source_group_id: str
    scene: str
    data_kind: str
    duration_ms: float
    fps: float
    video_path: Path | None
    tracks_path: Path
    annotations_path: Path
    video_sha256: str
    tracks_sha256: str
    annotations_sha256: str


@dataclass(frozen=True)
class DatasetVideo:
    entry: VideoEntry
    labels: VideoAnnotations
    records: tuple[FrameRecord, ...]


def read_manifest(path: Path) -> tuple[VideoEntry, ...]:
    if not path.is_file():
        raise DatasetError(f"Dataset manifest missing: {path}")
    entries = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != len(MANIFEST_FIELDS)
            or set(reader.fieldnames) != set(MANIFEST_FIELDS)
        ):
            raise DatasetError(f"Manifest columns must be {MANIFEST_FIELDS}")
        for line_number, row in enumerate(reader, 2):
            try:
                if any(value is None for value in row.values()) or None in row:
                    raise ValueError("Malformed CSV row")
                if not re.fullmatch(r"[A-Za-z0-9_-]+", row["video_id"]):
                    raise ValueError("video_id must be a safe ASCII identifier")
                for key in ("source_group_id", "scene"):
                    if not row[key].strip():
                        raise ValueError(f"Missing {key}")
                if row["split"] not in ("development", "test") or row["data_kind"] not in (
                    "real",
                    "synthetic",
                ):
                    raise ValueError("Invalid split/data_kind")
                duration, fps = float(row["duration_ms"]), float(row["fps"])
                if not all(math.isfinite(v) and v > 0 for v in (duration, fps)):
                    raise ValueError("Positive finite duration/FPS required")

                def resolve(key: str, values: dict[str, str] = row) -> Path:
                    if not values[key].strip():
                        raise ValueError(f"Missing {key}")
                    return (path.parent / values[key]).resolve()

                video = resolve("video_path") if row["video_path"] else None
                if row["data_kind"] == "real" and video is None:
                    raise ValueError("Real datasets require the original video")
                entries.append(
                    VideoEntry(
                        row["video_id"],
                        row["split"],
                        row["source_group_id"],
                        row["scene"],
                        row["data_kind"],
                        duration,
                        fps,
                        video,
                        resolve("tracks_path"),
                        resolve("annotations_path"),
                        row["video_sha256"],
                        row["tracks_sha256"],
                        row["annotations_sha256"],
                    )
                )
            except (ValueError, TypeError) as exc:
                raise DatasetError(f"Manifest row {line_number}: {exc}") from exc
    if not entries:
        raise DatasetError("Dataset manifest is empty")
    if len({entry.video_id for entry in entries}) != len(entries):
        raise DatasetError("Duplicate video_id")
    seen: dict[tuple[str, str], str] = {}
    for entry in entries:
        for kind, key in (
            ("source group", entry.source_group_id),
            ("video hash", entry.video_sha256),
            ("perception hash", entry.tracks_sha256),
        ):
            if not key:
                continue
            previous_split = seen.setdefault((kind, key), entry.split)
            if previous_split != entry.split:
                raise DatasetError(f"Development/test leakage: {kind} {key}")
    return tuple(sorted(entries, key=lambda entry: entry.video_id))


def _check_hash(path: Path, expected: str, kind: str) -> None:
    if not path.is_file():
        raise DatasetError(f"{kind} file missing: {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(path) != expected:
        raise DatasetError(f"{kind} hash mismatch: {path}")


def probe_video(path: Path) -> tuple[float, float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise DatasetError(f"Video cannot be opened: {path}")
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not all(math.isfinite(v) and v > 0 for v in (fps, count)):
            raise DatasetError("Video FPS/duration unavailable; prepare a verifiable clip first")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise DatasetError("Video first frame cannot be decoded")
        return count * 1000 / fps, fps
    finally:
        capture.release()


def _validate_overlaps(labels: VideoAnnotations) -> None:
    confirmed = [event for event in labels.events if event.annotation_status == "confirmed"]
    for i, first in enumerate(confirmed):
        for second in confirmed[i + 1 :]:
            if (
                max(first.start_ms, second.start_ms) < min(first.end_ms, second.end_ms)
                and set(first.class_ids) & set(second.class_ids)
                and (not first.target_regions or not second.target_regions)
            ):
                raise DatasetError(
                    "Overlapping same-class events require independent target regions"
                )


def load_dataset(
    path: Path, *, require_reviewed: bool = True, check_videos: bool = False
) -> tuple[DatasetVideo, ...]:
    loaded = []
    for entry in read_manifest(path):
        _check_hash(entry.tracks_path, entry.tracks_sha256, "Perception")
        _check_hash(entry.annotations_path, entry.annotations_sha256, "Annotation")
        if entry.video_path is not None:
            _check_hash(entry.video_path, entry.video_sha256, "Video")
        try:
            labels = load_annotations(entry.annotations_path, require_reviewed=require_reviewed)
            if (
                labels.video_id != entry.video_id
                or labels.duration_ms != entry.duration_ms
                or labels.data_kind != entry.data_kind
            ):
                raise DatasetError("Annotation identity, duration, or data kind mismatch")
            if not re.fullmatch(r"[0-9a-f]{64}", entry.video_sha256):
                raise DatasetError("Invalid video/scenario hash")
            if labels.source_video_sha256 != entry.video_sha256:
                raise DatasetError("Annotations do not refer to the original video hash")
            _validate_overlaps(labels)
            records = tuple(replay_records(entry.tracks_path))
            if not records:
                raise DatasetError("Perception file has no frame records")
            interval = 1000 / entry.fps
            if (
                records[0].timestamp_ms > interval + 1e-6
                or records[-1].timestamp_ms >= entry.duration_ms
                or records[-1].timestamp_ms + interval < entry.duration_ms - 1e-6
            ):
                raise DatasetError("Perception timestamps do not cover the declared video duration")
            if entry.data_kind == "real":
                source_path = entry.tracks_path.parent / "run_metadata.json"
                if not source_path.is_file():
                    raise DatasetError("Real perception requires Phase 1 run_metadata.json")
                source = json.loads(source_path.read_text(encoding="utf-8"))
                if source.get("status") != "completed":
                    raise DatasetError(
                        "Source perception run failed/incomplete; not an empty safe video"
                    )
                if source.get("input", {}).get("sha256") != entry.video_sha256:
                    raise DatasetError("Perception provenance refers to another video")
                if source.get("frames_processed") != len(records):
                    raise DatasetError("Perception frame count differs from source metadata")
            if check_videos and entry.video_path is not None:
                duration, fps = probe_video(entry.video_path)
                if (
                    abs(duration - entry.duration_ms) > max(1, 1000 / fps)
                    or abs(fps - entry.fps) > 0.01
                ):
                    raise DatasetError("Actual video duration/FPS differs from manifest and labels")
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise DatasetError(f"{entry.video_id}: {exc}") from exc
        loaded.append(DatasetVideo(entry, labels, records))
    return tuple(loaded)
