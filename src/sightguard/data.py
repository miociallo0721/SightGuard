"""Versioned plain-Python interface for independent perception replay."""

import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Detection:
    track_id: int | None
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if self.track_id is not None and (type(self.track_id) is not int or self.track_id < 0):
            raise ValueError("track_id must be non-negative or null")
        if (
            type(self.class_id) is not int
            or self.class_id < 0
            or not isinstance(self.class_name, str)
            or not self.class_name
        ):
            raise ValueError("Invalid class")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid confidence")
        if len(self.bbox_xyxy) != 4 or not all(math.isfinite(v) for v in self.bbox_xyxy):
            raise ValueError("bbox_xyxy must contain four finite coordinates")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 < x1 or y2 < y1:
            raise ValueError("Inverted bounding box")
        object.__setattr__(self, "bbox_xyxy", tuple(self.bbox_xyxy))


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    timestamp_ms: float
    frame_width: int
    frame_height: int
    detections: tuple[Detection, ...] = ()
    status: Literal["ok", "processing_error"] = "ok"
    error: str | None = None
    timestamp_source: Literal["video_pts", "fps_fallback", "monotonic"] = "fps_fallback"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported perception schema_version")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("Invalid frame_index")
        if not math.isfinite(self.timestamp_ms) or self.timestamp_ms < 0:
            raise ValueError("Invalid timestamp_ms")
        if any(type(v) is not int or v <= 0 for v in (self.frame_width, self.frame_height)):
            raise ValueError("Invalid frame dimensions")
        if self.timestamp_source not in ("video_pts", "fps_fallback", "monotonic"):
            raise ValueError("Invalid timestamp_source")
        if self.status not in ("ok", "processing_error"):
            raise ValueError("Invalid frame status")
        if self.status == "processing_error" and (not self.error or self.detections):
            raise ValueError("Failed frames require an error and no detections")
        if self.status == "ok" and self.error is not None:
            raise ValueError("Successful frames cannot contain an error")
        if any(not isinstance(item, Detection) for item in self.detections):
            raise ValueError("detections must contain Detection records")
        object.__setattr__(self, "detections", tuple(self.detections))

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, line: str) -> "FrameRecord":
        raw = json.loads(line)
        if "schema_version" not in raw:
            raise ValueError("Missing schema_version")
        raw["detections"] = tuple(Detection(**item) for item in raw["detections"])
        return cls(**raw)


def replay_records(path: Path) -> Iterator[FrameRecord]:
    """Stream saved frames without loading YOLO, torch, or video files."""
    previous_index = -1
    previous_time = -1.0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = FrameRecord.from_json(line)
                if record.frame_index != previous_index + 1 or record.timestamp_ms < previous_time:
                    raise ValueError("Frames must be contiguous with nondecreasing video time")
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid JSONL line {line_number}: {exc}") from exc
            previous_index, previous_time = record.frame_index, record.timestamp_ms
            yield record
