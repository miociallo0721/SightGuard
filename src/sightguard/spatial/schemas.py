"""Versioned native-portable cue contract; time is relative to one session."""

import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum


class CueSource(StrEnum):
    RISK = "risk"
    NAVIGATION = "navigation"
    SYSTEM = "system"


class DirectionReference(StrEnum):
    CAMERA = "CAMERA"
    LISTENER = "LISTENER"


CUE_TYPES = {"risk_initial", "risk_escalation", "perception_limited", "navigation", "system_test"}


@dataclass(frozen=True)
class SpatialCue:
    cue_id: str
    source: CueSource
    event_id: str | None
    cue_type: str
    timestamp_ms: float
    azimuth_deg: float
    elevation_deg: float
    direction_reference: DirectionReference
    direction_confidence: float | None
    priority: int
    expires_at_ms: float
    direction_method: str = "unavailable"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", CueSource(self.source))
        object.__setattr__(
            self, "direction_reference", DirectionReference(self.direction_reference)
        )
        if self.schema_version != "1.0" or not isinstance(self.cue_id, str) or not self.cue_id:
            raise ValueError("Invalid cue identity or schema version")
        if self.event_id is not None and (not isinstance(self.event_id, str) or not self.event_id):
            raise ValueError("Invalid event_id")
        if self.cue_type not in CUE_TYPES:
            raise ValueError("Unsupported cue_type")
        for name in ("timestamp_ms", "azimuth_deg", "elevation_deg", "expires_at_ms"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.timestamp_ms < 0 or self.expires_at_ms <= self.timestamp_ms:
            raise ValueError("Invalid cue time window")
        if not -180 <= self.azimuth_deg <= 180 or not -90 <= self.elevation_deg <= 90:
            raise ValueError("Cue angle outside contract")
        if type(self.priority) is not int or not 0 <= self.priority <= 100:
            raise ValueError("priority must be an integer in 0..100")
        confidence = self.direction_confidence
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Invalid direction confidence")
        if self.direction_method not in {
            "unavailable",
            "approximate_screen",
            "camera_geometry",
            "simulated",
            "manual_test",
        }:
            raise ValueError("Invalid direction method")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> "SpatialCue":
        raw = json.loads(value)
        if not isinstance(raw, dict) or "schema_version" not in raw:
            raise ValueError("Missing cue schema")
        return cls(**raw)
