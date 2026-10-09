"""Reference local footage, create draft labels, and seal reviewed annotation hashes."""

import csv
import re
import tempfile
from pathlib import Path

from sightguard.evaluation.dataset import (
    MANIFEST_FIELDS,
    DatasetError,
    load_dataset,
    probe_video,
    read_manifest,
    sha256_file,
)
from sightguard.evaluation.schema import VideoAnnotations, load_annotations


def import_video(
    video: Path,
    tracks: Path,
    dataset_dir: Path,
    *,
    video_id: str,
    source_group_id: str,
    scene: str,
    split: str = "development",
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise ValueError("video_id must be a safe ASCII identifier")
    if split not in ("development", "test") or not source_group_id.strip() or not scene.strip():
        raise ValueError("Valid split/source group/scene required")
    video, tracks, dataset_dir = video.resolve(), tracks.resolve(), dataset_dir.resolve()
    if not video.is_file() or not tracks.is_file():
        raise FileNotFoundError("Original video and existing perception JSONL are required")
    duration, fps = probe_video(video)
    manifest = dataset_dir / "manifest.csv"
    rows = []
    if manifest.is_file():
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                raise DatasetError("Invalid import manifest columns")
            rows = list(reader)
        if any(row["video_id"] == video_id for row in rows):
            raise DatasetError("video_id already imported")
        for row in rows:
            if row["source_group_id"] == source_group_id and row["split"] != split:
                raise DatasetError("Source-group leakage across development/test")
    annotation = dataset_dir / "annotations" / f"{video_id}.json"
    if annotation.exists():
        raise FileExistsError(annotation)
    label = VideoAnnotations(
        video_id,
        duration,
        "real",
        "draft",
        "UNASSIGNED",
        "pending_human_review",
        sha256_file(video),
        "Draft template only. Watch raw video independently; do not copy algorithm alerts.",
        (),
    )
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text(label.to_json() + "\n", encoding="utf-8")
    new_row = dict(
        zip(
            MANIFEST_FIELDS,
            (
                video_id,
                split,
                source_group_id,
                scene,
                "real",
                str(duration),
                str(fps),
                str(video),
                str(tracks),
                str(annotation),
                sha256_file(video),
                sha256_file(tracks),
                sha256_file(annotation),
            ),
            strict=True,
        )
    )
    rows.append(new_row)
    temporary = None
    try:
        # Validate the proposed manifest before replacing the prior import list.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=dataset_dir, suffix=".csv", encoding="utf-8", newline="", delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        load_dataset(temporary, require_reviewed=False, check_videos=True)
        temporary.replace(manifest)
    except Exception:
        annotation.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return annotation


def seal_labels(manifest: Path) -> None:
    """Bind manually reviewed files before freezing a protocol; never changes labels."""
    read_manifest(manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = (manifest.parent / row["annotations_path"]).resolve()
        labels = load_annotations(path)
        if labels.video_id != row["video_id"]:
            raise ValueError("Annotation video mismatch")
        row["annotations_sha256"] = sha256_file(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=manifest.parent, suffix=".csv", encoding="utf-8", newline="", delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        load_dataset(temporary)
        temporary.replace(manifest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
