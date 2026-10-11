"""Prepare the approved UrbanEgo archive; no network or generated ground truth."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

from sightguard.evaluation.real_world import (
    clip_video,
    extract_member,
    group_split,
    inspect_video,
    validate_manifest,
    write_record,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument(
        "--starts", type=float, nargs="+", help="Reviewed source start times in seconds"
    )
    args = parser.parse_args()
    if not 8 <= args.count <= 12 or not 10 <= args.seconds <= 30:
        parser.error("First batch requires 8-12 clips, 10-30 seconds each")
    if args.starts is not None and (
        len(args.starts) != args.count
        or any(not math.isfinite(start) or start < 0 for start in args.starts)
        or any(b < a + args.seconds for a, b in zip(args.starts, args.starts[1:], strict=False))
    ):
        parser.error("Start times must match count, be nonnegative and non-overlapping")
    data = ROOT / "datasets/real_world"
    for name in ("raw", "processed", "manifests", "annotations", "splits", "reports"):
        (data / name).mkdir(parents=True, exist_ok=True)
    manifest_path = data / "manifests/dataset.json"
    if manifest_path.exists():
        raise FileExistsError("Existing first batch preserved; use a separate batch")
    download = json.loads(args.archive.with_suffix(".zip.download.json").read_text())
    if download["status"] != "verified":
        raise ValueError("A publisher-checksummed archive is required")
    from sightguard.evaluation.dataset import sha256_file

    if sha256_file(args.archive) != download["sha256"]:
        raise ValueError("Archive SHA256 mismatch")
    with zipfile.ZipFile(args.archive) as archive:
        videos = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".mp4")
            and "__MACOSX" not in Path(name).parts
            and not Path(name).name.startswith("._")
        ]
        write_record(data / "reports/archive_members.json", {"members": archive.namelist()})
    if len(videos) != 1:
        raise ValueError(f"Expected one RGB MP4; choose the correct member explicitly: {videos}")
    video = data / "raw/urbanego" / Path(videos[0]).name
    extract_member(args.archive, videos[0], video)
    info = inspect_video(video)
    write_record(data / "reports/source_video_quality.json", info)
    source = next(
        row
        for row in json.loads((data / "manifests/source_catalog.json").read_text(encoding="utf-8"))[
            "sources"
        ]
        if row["dataset"] == "UrbanEgo"
    )
    duration_s = info["duration_ms"] / 1000
    if duration_s < args.count * args.seconds:
        raise ValueError("Insufficient source duration for non-overlapping diagnostic clips")
    rows = []
    for index in range(args.count):
        start = (
            args.starts[index]
            if args.starts is not None
            else int((duration_s - args.seconds) * index / args.count)
        )
        if start + args.seconds > duration_s:
            raise ValueError("Requested interval exceeds original duration")
        video_id = f"urbanego_run8_{index + 1:02}"
        target = data / "processed" / f"{video_id}.mp4"
        print(f"Preparing {video_id} at {start}s", flush=True)
        clip = clip_video(video, target, start, start + args.seconds, source_info=info)
        write_record(data / "reports" / f"{video_id}_quality.json", clip)
        rows.append(
            {
                "schema_version": "sightguard.real-world.v0.1",
                "video_id": video_id,
                "parent_source_id": "urbanego_run8_20260714_174138",
                "scenario_category": "unassigned_pending_visual_review",
                "camera_perspective": "head-mounted pedestrian first-person; publisher description",
                "acquisition_status": "verified",
                "annotation_status": "pending",
                "eligibility_status": "pending_visual_review",
                "split": "development",
                "filepath": str(target.resolve()),
                "duration": clip["duration_ms"] / 1000,
                "width": clip["width"],
                "height": clip["height"],
                "fps": clip["fps"],
                "source_sha256": info["sha256"],
                "processed_sha256": clip["sha256"],
                "source_dataset": "UrbanEgo",
                "source_url": source["data_url"],
                "source_record_id": source["source_record_id"],
                "license": source["license"],
                "license_url": source["license_url"],
                "access_date": "2026-10-11",
                "original_filename": videos[0],
                "original_sha256": info["sha256"],
                "archive_sha256": download["sha256"],
                "original_duration": duration_s,
                "original_fps": info["fps"],
                "original_resolution": [info["width"], info["height"]],
                "selected_start_time": start,
                "selected_end_time": start + args.seconds,
                "selection_reason": (
                    "Deterministic spaced diagnostic crops; no presumed risk quotas"
                ),
                "permitted_usage": source["permitted_usage"],
                "attribution": source["attribution"],
                "review_status": "pending",
                "sensor_capture_time_mapping": "unavailable; not inferred from MP4 FPS",
                "processing_record": str((data / "reports" / f"{video_id}_quality.json").resolve()),
            }
        )
    if len({row["processed_sha256"] for row in rows}) != len(rows):
        raise ValueError("Exact duplicate processed clips")
    validate_manifest(rows)
    write_record(manifest_path, {"schema_version": "sightguard.real-world.v0.1", "videos": rows})
    write_record(
        data / "splits/development.json",
        {
            "seed": 20261011,
            "rule": "entire first parent in development; no test set",
            "assignments": group_split(rows, seed=20261011),
        },
    )


if __name__ == "__main__":
    main()
