"""Generate a versioned PC AssessmentEngine reference for Android replay parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sightguard.data import Detection, FrameRecord  # noqa: E402
from sightguard.risk.config import RiskConfig  # noqa: E402
from sightguard.risk.engine import AssessmentEngine  # noqa: E402

CAMERA_TRACE_PATH = REPOSITORY_ROOT / "android/app/src/test/resources/camera_detection_trace.jsonl"
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "android/app/src/test/resources/parity_reference.json"
CLASS_NAMES = {0: "person", 5: "bus"}


def _camel_case(name: str) -> str:
    return re.sub(r"_([a-z])", lambda match: match.group(1).upper(), name)


def _camelize(value: Any) -> Any:
    if is_dataclass(value):
        return _camelize(asdict(value))
    if isinstance(value, dict):
        return {_camel_case(key): _camelize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_camelize(item) for item in value]
    return value


def _p_detection(
    class_id: int,
    confidence: float,
    box: list[float],
    track_id: int | None,
) -> Detection:
    return Detection(
        track_id=track_id,
        class_id=class_id,
        class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
        confidence=confidence,
        bbox_xyxy=tuple(box),
    )


def _frame_record(raw: dict[str, Any], frame_index: int, *, synthetic: bool) -> FrameRecord:
    width, height = int(raw["width"]), int(raw["height"])
    detections = []
    for row in raw.get("detections", []):
        if synthetic:
            class_id, confidence, x1, y1, x2, y2, *rest = row
            track_id = rest[0] if rest else None
            box = [float(x1), float(y1), float(x2), float(y2)]
        else:
            class_id, confidence, left, top, right, bottom = row
            track_id = None
            # The saved CameraX trace uses normalized detector coordinates. Convert back to
            # FrameRecord's pixel-space bbox without changing the stored source trace.
            box = [left * width, top * height, right * width, bottom * height]
        detections.append(_p_detection(int(class_id), float(confidence), box, track_id))

    status = raw.get("status", "ok")
    return FrameRecord(
        frame_index=frame_index,
        timestamp_ms=float(raw["timestamp_ms"]),
        frame_width=width,
        frame_height=height,
        detections=tuple(detections),
        status=status,
        error=raw.get("error"),
        timestamp_source=raw.get("timestamp_source", "monotonic"),
    )


def _record_json(frame: FrameRecord) -> dict[str, Any]:
    return {
        "frameIndex": frame.frame_index,
        "timestampMs": frame.timestamp_ms,
        "frameWidth": frame.frame_width,
        "frameHeight": frame.frame_height,
        "detections": [
            {
                "trackId": item.track_id,
                "classId": item.class_id,
                "className": item.class_name,
                "confidence": item.confidence,
                "bboxXyxy": list(item.bbox_xyxy),
            }
            for item in frame.detections
        ],
        "status": frame.status,
        "error": frame.error,
        "timestampSource": frame.timestamp_source,
        "schemaVersion": frame.schema_version,
    }


def _reference_run(
    name: str,
    frames: list[FrameRecord],
    config: RiskConfig,
) -> dict[str, Any]:
    engine = AssessmentEngine(config, stream_id=name)
    outputs = []
    for frame in frames:
        result = engine.step(frame)
        risk_frame = result.frame
        events = []
        for event in risk_frame.events:
            assessment = event.assessment
            events.append(
                {
                    "eventId": event.event_id,
                    "state": event.state,
                    "classId": event.class_id,
                    "className": event.class_name,
                    "firstSeenMs": event.first_seen_ms,
                    "lastSeenMs": event.last_seen_ms,
                    "validObservations": event.valid_observations,
                    "confirmed": event.confirmed,
                    "sourceTrackIds": event.source_track_ids,
                    "lastTrackId": event.last_track_id,
                    "observedNow": event.observed_now,
                    "assessment": None
                    if assessment is None
                    else {
                        "score": assessment.score,
                        "rawScore": assessment.raw_score,
                        "level": assessment.level,
                        "motionScore": assessment.motion_score,
                        "qualityScore": assessment.quality.score,
                    },
                }
            )
        changes = []
        for change in result.changes:
            compact = _camelize(change)
            compact.pop("snapshot", None)
            changes.append(compact)
        outputs.append(
            {
                "frame": {
                    "streamId": risk_frame.stream_id,
                    "frameIndex": risk_frame.frame_index,
                    "timestampMs": risk_frame.timestamp_ms,
                    "timestampSource": risk_frame.timestamp_source,
                    "inputStatus": risk_frame.input_status,
                    "perception": _camelize(risk_frame.perception),
                    "observations": _camelize(risk_frame.observations),
                    "events": events,
                    "warnings": risk_frame.warnings,
                    "schemaVersion": risk_frame.schema_version,
                },
                "changes": changes,
                "alerts": _camelize(result.alerts),
            }
        )
    return {"steps": outputs, "finishChanges": _camelize(engine.finish())}


def _synthetic_scenarios(config: RiskConfig) -> list[dict[str, Any]]:
    """Small labeled state fixtures; these are hand-authored inputs, never model output."""
    width, height = 640, 480

    def detection(box: tuple[float, float, float, float], track_id: int | None = 11,
                  confidence: float = 0.9, class_id: int = 0) -> list[float | int]:
        return [class_id, confidence, *box, track_id]

    def raw_frame(timestamp_ms: float, detections: list[list[float | int]] | None = None,
                  *, status: str = "ok", error: str | None = None) -> dict[str, Any]:
        return {
            "timestamp_ms": timestamp_ms,
            "width": width,
            "height": height,
            "detections": detections or [],
            "status": status,
            "error": error,
            "timestamp_source": "monotonic",
        }

    definitions: list[tuple[str, str, list[dict[str, Any]]]] = [
        (
            "track_id_switch",
            "same visible target switches tracker ID while geometry remains consistent",
            [
                raw_frame(0, [detection((270, 260, 350, 430), 11)]),
                raw_frame(250, [detection((272, 262, 352, 432), 12)]),
                raw_frame(500, [detection((274, 264, 354, 434), 12)]),
                raw_frame(750, [detection((276, 266, 356, 436), 12)]),
            ],
        ),
        (
            "lost_recovery",
            "confirmed target disappears, recovers inside its window, then expires",
            [
                raw_frame(0, [detection((270, 260, 350, 430))]),
                raw_frame(250, [detection((271, 261, 351, 431))]),
                raw_frame(500),
                raw_frame(700, [detection((273, 263, 353, 433))]),
                raw_frame(1000),
                raw_frame(1501),
            ],
        ),
        (
            "health_episode",
            "processing errors form a persistent perception-limited episode with an alert",
            [
                raw_frame(0, status="processing_error", error="synthetic camera read failure"),
                raw_frame(250, status="processing_error", error="synthetic camera read failure"),
                raw_frame(500, status="processing_error", error="synthetic camera read failure"),
                raw_frame(750),
            ],
        ),
        (
            "risk_upgrade",
            "increasing image scale and lower-frame position can raise an active risk level",
            [
                raw_frame(0, [detection((305, 265, 335, 345), 21)]),
                raw_frame(250, [detection((295, 250, 345, 390), 21)]),
                raw_frame(750, [detection((270, 225, 370, 440), 21)]),
                raw_frame(1500, [detection((230, 190, 410, 455), 21)]),
            ],
        ),
        (
            "hysteresis",
            "risk downgrades after its hold time; a later upgrade is immediate",
            [
                raw_frame(0, [detection((250, 200, 390, 455), 31)]),
                raw_frame(250, [detection((250, 200, 390, 455), 31)]),
                raw_frame(350, [detection((20, 20, 55, 110), 31)]),
                raw_frame(749, [detection((20, 20, 55, 110), 31)]),
                raw_frame(750, [detection((20, 20, 55, 110), 31)]),
                raw_frame(1000, [detection((250, 200, 390, 455), 31)]),
            ],
        ),
        (
            "nonuniform_sampling",
            "motion intervals use capture timestamps across short and irregular sampling gaps",
            [
                raw_frame(0, [detection((300, 300, 320, 340), 41)]),
                raw_frame(5, [detection((299, 299, 321, 341), 41)]),
                raw_frame(10, [detection((298, 298, 322, 342), 41)]),
                raw_frame(260, [detection((290, 290, 330, 370), 41)]),
                raw_frame(1000, [detection((270, 250, 350, 410), 41)]),
            ],
        ),
        (
            "empty_and_fallback_time",
            "empty valid frames and fallback timestamps remain explicit in feature quality",
            [
                raw_frame(0),
                {
                    **raw_frame(100, [detection((280, 300, 360, 450), None)]),
                    "timestamp_source": "fps_fallback",
                },
                {
                    **raw_frame(300, [detection((275, 295, 365, 452), None)]),
                    "timestamp_source": "fps_fallback",
                },
            ],
        ),
    ]

    result = []
    for name, description, raw_frames in definitions:
        records = [
            _frame_record(raw, index, synthetic=True)
            for index, raw in enumerate(raw_frames)
        ]
        result.append(
            {
                "name": name,
                "fixtureType": "synthetic_state_scenario",
                "synthetic": True,
                "description": description,
                "frames": [_record_json(frame) for frame in records],
                "reference": _reference_run(f"synthetic_{name}", records, config),
            }
        )
    return result


def generate_reference(output_path: Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    trace_bytes = CAMERA_TRACE_PATH.read_bytes()
    trace_lines = trace_bytes.decode("utf-8").splitlines()
    if not trace_lines:
        raise ValueError(f"CameraX trace is empty: {CAMERA_TRACE_PATH}")
    metadata = json.loads(trace_lines[0])
    if metadata.get("source") != "real CameraX ImageProxy":
        raise ValueError("Input trace does not identify real CameraX ImageProxy detections")
    raw_detector_frames = [json.loads(line) for line in trace_lines[1:] if line.strip()]
    if len(raw_detector_frames) != 200:
        raise ValueError(f"Expected 200 saved CameraX frames; got {len(raw_detector_frames)}")

    records = [
        _frame_record(raw, index, synthetic=False)
        for index, raw in enumerate(raw_detector_frames)
    ]
    config = RiskConfig()
    fixture = {
        "schemaVersion": "1.0",
        "riskSchemaVersion": "2.0",
        "generator": "android/tools/generate_parity_reference.py",
        "generatedAtUtc": datetime.now(UTC).isoformat(),
        "riskConfig": _camelize(config),
        "realCameraTrace": {
            "fixtureType": "real_camera_trace",
            "synthetic": False,
            "provenance": {
                "source": metadata["source"],
                "path": str(CAMERA_TRACE_PATH.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
                "traceSha256": hashlib.sha256(trace_bytes).hexdigest(),
                "modelSha256": metadata.get("model_sha256"),
                "savedFrameCount": len(raw_detector_frames),
                "coordinatesInSource": "normalized_xyxy",
                "coordinatesInFrames": "pixel_xyxy",
                "timestampSource": "monotonic",
                "trackIdsInSource": False,
                "note": "Stored CameraX detector output; no inference was run by this generator.",
            },
            "rawDetectorFrames": raw_detector_frames,
            "frames": [_record_json(frame) for frame in records],
            "reference": _reference_run("camera_trace", records, config),
        },
        "syntheticScenarios": _synthetic_scenarios(config),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(fixture, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    fixture = generate_reference(args.output.resolve())
    print(
        f"Wrote {args.output} with {len(fixture['realCameraTrace']['frames'])} real CameraX frames "
        f"and {len(fixture['syntheticScenarios'])} labeled synthetic scenarios."
    )


if __name__ == "__main__":
    main()
