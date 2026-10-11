"""Run existing pipelines and create Phase 3 draft labels, never automatic GT."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sightguard.config import load_config
from sightguard.evaluation.dataset import load_dataset, read_manifest, sha256_file
from sightguard.evaluation.importing import import_video
from sightguard.evaluation.real_world import validate_manifest, write_record
from sightguard.pipeline.assess import assess_tracks
from sightguard.pipeline.video import analyze_video
from sightguard.risk.config import load_risk_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "datasets/real_world/manifests/dataset.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/real_world/v01")
    parser.add_argument("--resume", action="store_true", help="Verify completed runs before reuse")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))["videos"]
    validate_manifest(rows)
    if any(row["eligibility_status"] != "accepted" for row in rows):
        raise ValueError("Review video quality before importing the batch")
    dataset = ROOT / "datasets/real_world/annotations/phase3"
    if not args.resume and ((dataset / "manifest.csv").exists() or args.output_dir.exists()):
        raise FileExistsError("Existing batch outputs preserved; select a fresh batch")
    perception = load_config(ROOT / "configs/real-world-perception.yaml")
    risk = load_risk_config(ROOT / "configs/risk.yaml")
    imported = (
        {entry.video_id: entry for entry in read_manifest(dataset / "manifest.csv")}
        if (dataset / "manifest.csv").exists()
        else {}
    )
    results = []
    for row in rows:
        video_id = row["video_id"]
        output = args.output_dir / video_id
        metadata = output / "perception/run_metadata.json"
        if args.resume and metadata.exists():
            detected = json.loads(metadata.read_text(encoding="utf-8"))
            if (
                detected["status"] != "completed"
                or detected["input"]["sha256"] != sha256_file(Path(row["filepath"]))
                or detected["config"] != json.loads(json.dumps(perception.to_dict()))
            ):
                raise ValueError("Existing perception cannot be safely resumed")
        else:
            detected = analyze_video(Path(row["filepath"]), output / "perception", perception)
        assessment_metadata = output / "assessment/run_metadata.json"
        if args.resume and assessment_metadata.exists():
            assessed = json.loads(assessment_metadata.read_text(encoding="utf-8"))
            if (
                assessed["status"] != "completed"
                or assessed["input"]["sha256"] != sha256_file(output / "perception/tracks.jsonl")
                or assessed["config"] != json.loads(json.dumps(risk.to_dict()))
            ):
                raise ValueError("Existing assessment is incomplete")
        else:
            assessed = assess_tracks(
                output / "perception/tracks.jsonl", output / "assessment", risk, stream_id=video_id
            )
        # Research Console requires tracks under datasets; reference a local copy.
        tracks = dataset / "tracks" / video_id / "tracks.jsonl"
        tracks.parent.mkdir(parents=True, exist_ok=True)
        for source, target in (
            (output / "perception/tracks.jsonl", tracks),
            (metadata, tracks.parent / "run_metadata.json"),
        ):
            if target.exists():
                if sha256_file(target) != sha256_file(source):
                    raise ValueError("Dataset copy differs; refusing overwrite")
            else:
                with target.open("xb") as stream:
                    stream.write(source.read_bytes())
        if video_id in imported:
            entry = imported[video_id]
            if (
                entry.video_sha256 != row["processed_sha256"]
                or entry.tracks_path != tracks.resolve()
            ):
                raise ValueError("Previously imported video differs")
            label = entry.annotations_path
        else:
            label = import_video(
                Path(row["filepath"]),
                tracks,
                dataset,
                video_id=video_id,
                source_group_id=row["parent_source_id"],
                scene=row["scenario_category"],
                split=row["split"],
            )
        results.append(
            {
                "video_id": video_id,
                "frames": detected["frames_processed"],
                "detections": detected["detections"],
                "alerts": assessed["alerts"],
                "event_records": assessed["event_records"],
                "annotation_path": str(label),
                "annotation_status": "draft",
                "human_reviewed": False,
            }
        )
        write_record(
            args.output_dir / "batch.json", {"videos": results, "supervised_metrics": None}
        )
    # Existing Phase 3 gate must refuse draft labels, not interpret them as negatives.
    load_dataset(dataset / "manifest.csv", require_reviewed=False, check_videos=True)
    try:
        load_dataset(dataset / "manifest.csv", require_reviewed=True)
    except ValueError as exc:
        write_record(
            args.output_dir / "evaluation_gate.json",
            {"status": "pending_human_labels", "reason": str(exc), "metrics": None},
        )
    else:
        raise RuntimeError("Draft batch unexpectedly passed the supervised gate")
    tasks = {
        "schema_version": "sightguard.visual-tasks.v0.1",
        "status": "pending_human_annotation",
        "instructions": (
            "Watch raw video independently of model outputs; null means unlabelled, never negative."
        ),
        "attributes": [
            "object_class",
            "visible",
            "position_left_center_right",
            "overlap_visible_passage",
            "occlusion",
            "frame_motion",
            "evidence_sufficient",
        ],
        "definitions": {
            "visible": "Target directly visible: true/false; uncertain: null",
            "position_left_center_right": (
                "Normalized target center x: <1/3 left; <=2/3 center; otherwise right"
            ),
            "overlap_visible_passage": (
                "Human-visible candidate walking area; unknown: null; no travel path prediction"
            ),
            "occlusion": "Visible partial blockage of target; unknown: null",
            "frame_motion": "Visible change in neighboring frames; no physical speed inference",
            "evidence_sufficient": "Enough image evidence for requested attribute; unknown: null",
        },
        "videos": [
            {
                "video_id": row["video_id"],
                "image_timestamp_ms": 7500,
                "image_path": None,
                "image_sha256": None,
                "human_attributes": None,
                "annotator_id": None,
                "model_suggestions": None,
            }
            for row in rows
        ],
    }
    task_path = ROOT / "datasets/real_world/annotations/visual_tasks.json"
    if not task_path.exists():
        import cv2

        for row, task in zip(rows, tasks["videos"], strict=True):
            capture = cv2.VideoCapture(row["filepath"])
            try:
                capture.set(cv2.CAP_PROP_POS_MSEC, task["image_timestamp_ms"])
                ok, image = capture.read()
                if not ok:
                    raise ValueError("Cannot extract visual annotation frame")
                task["image_timestamp_ms"] = capture.get(cv2.CAP_PROP_POS_MSEC)
                image_path = task_path.parent / "visual_frames" / f"{row['video_id']}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if image_path.exists():
                    raise FileExistsError(image_path)
                if not cv2.imwrite(str(image_path), image):
                    raise ValueError("Cannot save visual annotation frame")
                task.update(
                    image_path=str(image_path.resolve()), image_sha256=sha256_file(image_path)
                )
            finally:
                capture.release()
        write_record(task_path, tasks)


if __name__ == "__main__":
    main()
