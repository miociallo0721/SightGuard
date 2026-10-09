"""Create small, deterministic synthetic Phase 3 benchmark inputs.

The annotations describe the scripted synthetic scenarios directly. This script
does not invoke a perception model, strategy, or evaluator, and does not claim to
represent real video or real-world safety performance.
"""

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from sightguard.data import Detection, FrameRecord
from sightguard.evaluation.dataset import MANIFEST_FIELDS, sha256_file
from sightguard.evaluation.schema import (
    AlertOpportunity,
    LabeledEvent,
    TargetRegion,
    VideoAnnotations,
)
from sightguard.risk.schema import RiskLevel

FPS = 4.0
FRAME_INTERVAL_MS = 1000 / FPS
ANNOTATOR_ID = "synthetic-fixture-author"
ESCALATION_SIZES = (
    (20, 40),
    (20, 40),
    (30, 60),
    (45, 90),
    (68, 136),
    (102, 204),
    (102, 204),
    (102, 204),
)


@dataclass(frozen=True)
class Scenario:
    video_id: str
    split: str
    scene: str
    duration_ms: int
    records: tuple[FrameRecord, ...]
    events: tuple[LabeledEvent, ...]
    notes: str


def _detection(
    class_id: int, bbox: tuple[float, float, float, float], track_id: int = 1
) -> Detection:
    class_names = {0: "person", 1: "bicycle", 3: "motorcycle"}
    return Detection(track_id, class_id, class_names[class_id], 0.95, bbox)


def _frame(index: int, detections: tuple[Detection, ...] = ()) -> FrameRecord:
    return FrameRecord(
        frame_index=index,
        timestamp_ms=index * FRAME_INTERVAL_MS,
        frame_width=640,
        frame_height=480,
        detections=detections,
        timestamp_source="video_pts",
    )


def _records(
    frame_count: int, detections: tuple[Detection, ...] | None = None
) -> tuple[FrameRecord, ...]:
    per_frame = detections or ()
    return tuple(_frame(index, per_frame) for index in range(frame_count))


def _event(
    video_id: str,
    event_id: str,
    *,
    event_type: str,
    severity: RiskLevel,
    duration_ms: int,
    class_id: int,
    opportunities: tuple[AlertOpportunity, ...],
    regions: tuple[TargetRegion, ...],
    notes: str,
) -> LabeledEvent:
    return LabeledEvent(
        video_id=video_id,
        event_id=event_id,
        event_type=event_type,
        start_ms=0,
        end_ms=duration_ms,
        severity=severity,
        annotation_status="confirmed",
        alert_opportunities=opportunities,
        annotator_id=ANNOTATOR_ID,
        notes=notes,
        class_ids=(class_id,),
        target_regions=regions,
    )


def _initial_opportunity(deadline_ms: int = 750) -> AlertOpportunity:
    return AlertOpportunity("initial", "initial", 0, deadline_ms, RiskLevel.MEDIUM)


def _normalized_box(
    bbox: tuple[float, float, float, float], padding: float = 0.005
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, x1 / 640 - padding),
        max(0.0, y1 / 480 - padding),
        min(1.0, x2 / 640 + padding),
        min(1.0, y2 / 480 + padding),
    )


def _square_bbox(width: int, height: int) -> tuple[float, float, float, float]:
    return (320 - width / 2, 336 - height, 320 + width / 2, 336)


def _scenario_specs() -> tuple[Scenario, ...]:
    dev_id = "dev_stationary"
    dev_box = (280, 240, 360, 400)
    dev_duration = 2000
    dev_event = _event(
        dev_id,
        "gt_dev_bicycle",
        event_type="stationary_bicycle",
        severity=RiskLevel.HIGH,
        duration_ms=dev_duration,
        class_id=1,
        opportunities=(_initial_opportunity(),),
        regions=(TargetRegion(0, dev_duration, _normalized_box(dev_box)),),
        notes="Development-only scripted bicycle scenario; label authored from scenario rules.",
    )

    stationary_id = "test_stationary"
    stationary_duration = 4000
    stationary_box = (224, 96, 416, 456)  # normalized xyxy=(0.35, 0.20, 0.65, 0.95)
    stationary_event = _event(
        stationary_id,
        "gt_stationary_person",
        event_type="stationary_obstacle",
        severity=RiskLevel.HIGH,
        duration_ms=stationary_duration,
        class_id=0,
        opportunities=(_initial_opportunity(),),
        regions=(TargetRegion(0, stationary_duration, _normalized_box(stationary_box)),),
        notes="Large stationary person in the scripted forward image region.",
    )

    escalation_id = "test_escalation"
    escalation_duration = 2000
    escalation_times = tuple(index * FRAME_INTERVAL_MS for index in range(len(ESCALATION_SIZES)))
    escalation_boxes = tuple(_square_bbox(width, height) for width, height in ESCALATION_SIZES)
    escalation_event = _event(
        escalation_id,
        "gt_growing_person",
        event_type="growing_scale_obstacle",
        severity=RiskLevel.HIGH,
        duration_ms=escalation_duration,
        class_id=0,
        opportunities=(
            _initial_opportunity(),
            AlertOpportunity("escalation", "escalation", 1000, 1500, RiskLevel.HIGH),
        ),
        regions=tuple(
            TargetRegion(
                timestamp,
                min(escalation_duration, timestamp + FRAME_INTERVAL_MS),
                _normalized_box(box),
            )
            for timestamp, box in zip(escalation_times, escalation_boxes, strict=True)
        ),
        notes=(
            "Scripted target grows in image scale; "
            "escalation opportunity is independently declared."
        ),
    )

    missed_id = "test_missed"
    missed_duration = 4000
    missed_event = _event(
        missed_id,
        "gt_unperceived_person",
        event_type="important_unperceived_obstacle",
        severity=RiskLevel.HIGH,
        duration_ms=missed_duration,
        class_id=0,
        opportunities=(_initial_opportunity(),),
        regions=(TargetRegion(0, missed_duration, (0.4, 0.25, 0.6, 0.95)),),
        notes=(
            "Scenario ground truth contains an important obstacle, "
            "while the tracks intentionally omit it."
        ),
    )

    nonrisk_id = "test_nonrisk"
    small_box = _square_bbox(20, 40)

    return (
        Scenario(
            dev_id,
            "development",
            "synthetic_stationary_bicycle",
            dev_duration,
            _records(8, (_detection(1, dev_box),)),
            (dev_event,),
            "Synthetic development fixture; no real video is included.",
        ),
        Scenario(
            stationary_id,
            "test",
            "synthetic_stationary_person",
            stationary_duration,
            _records(16, (_detection(0, stationary_box),)),
            (stationary_event,),
            (
                "Synthetic test fixture; fixed-size target occupies normalized bbox "
                "(0.35, 0.20, 0.65, 0.95)."
            ),
        ),
        Scenario(
            escalation_id,
            "test",
            "synthetic_scale_growth",
            escalation_duration,
            tuple(
                _frame(index, (_detection(0, box),)) for index, box in enumerate(escalation_boxes)
            ),
            (escalation_event,),
            (
                "Synthetic test fixture; target boxes and independent alert opportunities "
                "are scripted."
            ),
        ),
        Scenario(
            missed_id,
            "test",
            "synthetic_perception_miss",
            missed_duration,
            _records(16),
            (missed_event,),
            (
                "Synthetic test fixture; ground-truth obstacle is deliberately absent "
                "from perception tracks."
            ),
        ),
        Scenario(
            nonrisk_id,
            "test",
            "synthetic_nonrisk_passage",
            3000,
            _records(12, (_detection(0, small_box),)),
            (),
            (
                "Synthetic test fixture; small target passage is labelled as no "
                "alert-worthy risk event."
            ),
        ),
    )


def generate_fixtures(output_dir: Path) -> Path:
    """Write labels, replayable tracks, and a split-aware manifest."""
    output_dir = output_dir.resolve()
    tracks_dir = output_dir / "tracks"
    annotations_dir = output_dir / "annotations"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    rows = []
    for scenario in _scenario_specs():
        video_sha256 = hashlib.sha256(
            f"sightguard-synthetic-scenario-v1:{scenario.video_id}".encode()
        ).hexdigest()
        tracks_path = tracks_dir / f"{scenario.video_id}.jsonl"
        annotation_path = annotations_dir / f"{scenario.video_id}.json"
        tracks_path.write_text(
            "".join(f"{record.to_json()}\n" for record in scenario.records),
            encoding="utf-8",
            newline="",
        )
        annotations = VideoAnnotations(
            video_id=scenario.video_id,
            duration_ms=scenario.duration_ms,
            data_kind="synthetic",
            annotation_status="reviewed",
            annotator_id=ANNOTATOR_ID,
            label_source="synthetic_scenario",
            source_video_sha256=video_sha256,
            notes=scenario.notes,
            events=scenario.events,
        )
        annotation_path.write_text(
            json.dumps(
                json.loads(annotations.to_json()),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="",
        )
        rows.append(
            {
                "video_id": scenario.video_id,
                "split": scenario.split,
                "source_group_id": f"synthetic-{scenario.split}-{scenario.video_id}",
                "scene": scenario.scene,
                "data_kind": "synthetic",
                "duration_ms": scenario.duration_ms,
                "fps": f"{FPS:g}",
                "video_path": "",
                "tracks_path": tracks_path.relative_to(output_dir).as_posix(),
                "annotations_path": annotation_path.relative_to(output_dir).as_posix(),
                "video_sha256": video_sha256,
                "tracks_sha256": sha256_file(tracks_path),
                "annotations_sha256": sha256_file(annotation_path),
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "datasets" / "synthetic",
        help="Output directory (defaults to datasets/synthetic)",
    )
    args = parser.parse_args()
    manifest = generate_fixtures(args.output_dir)
    print(f"Generated deterministic synthetic Phase 3 fixtures: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
