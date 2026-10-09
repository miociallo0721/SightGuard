"""Plain serializable Phase 2 interfaces. Scores are not collision probabilities."""

import json
from dataclasses import asdict, dataclass
from enum import StrEnum

RISK_SCHEMA_VERSION = "2.0"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @property
    def rank(self) -> int:
        return (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH).index(self)


class EventState(StrEnum):
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    RESOLVED = "RESOLVED"


class JsonRecord:
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class SpatialFeatures:
    bbox_normalized: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    width: float
    height: float
    area: float
    bottom_center_xy: tuple[float, float]
    focus_relevance: float
    bbox_clipped: bool


@dataclass(frozen=True)
class PerceptionQuality:
    detection_confidence: float
    continuity: float
    valid_observations: int
    recent_observations: int
    track_id_missing: bool
    interrupted: bool
    motion_reliable: bool
    timestamp_source: str
    score: float
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class RiskFeatures:
    spatial: SpatialFeatures
    observation_duration_ms: float
    observation_interval_ms: float | None
    scale_change_per_s: float | None
    center_change_per_s: tuple[float, float] | None
    inward_change_per_s: float | None
    detection_stability: float
    quality: PerceptionQuality
    motion_interval_ms: float | None = None


@dataclass(frozen=True)
class RiskAssessment:
    spatial_score: float
    appearance_score: float
    motion_score: float | None
    raw_score: float
    score: float
    level: RiskLevel
    quality: PerceptionQuality
    degraded: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RiskObservation:
    detection_index: int
    source_track_id: int | None
    class_id: int
    class_name: str
    event_id: str | None
    association_method: str | None
    features: RiskFeatures | None
    assessment: RiskAssessment | None
    error: str | None = None


@dataclass(frozen=True)
class EventSnapshot:
    event_id: str
    state: EventState
    class_id: int
    class_name: str
    first_seen_ms: float
    last_seen_ms: float
    valid_observations: int
    confirmed: bool
    source_track_ids: tuple[int, ...]
    last_track_id: int | None
    observed_now: bool
    assessment: RiskAssessment | None


@dataclass(frozen=True)
class EventChange(JsonRecord):
    stream_id: str
    event_id: str
    timestamp_ms: float
    event_kind: str
    action: str
    state: EventState
    reason: str
    source_track_id: int | None = None
    association_method: str | None = None
    snapshot: EventSnapshot | None = None
    schema_version: str = RISK_SCHEMA_VERSION


@dataclass(frozen=True)
class AlertEvent(JsonRecord):
    stream_id: str
    alert_id: str
    event_id: str
    timestamp_ms: float
    risk_level: RiskLevel | None
    risk_score: float | None
    alert_type: str
    reason: str
    source_track_ids: tuple[int, ...]
    quality_score: float | None = None
    schema_version: str = RISK_SCHEMA_VERSION


@dataclass(frozen=True)
class PerceptionStatus:
    status: str
    event_id: str | None
    state: EventState | None
    since_ms: float | None
    reason: str | None


@dataclass(frozen=True)
class RiskFrame(JsonRecord):
    stream_id: str
    frame_index: int
    timestamp_ms: float
    timestamp_source: str
    input_status: str
    perception: PerceptionStatus
    observations: tuple[RiskObservation, ...]
    events: tuple[EventSnapshot, ...]
    warnings: tuple[str, ...]
    schema_version: str = RISK_SCHEMA_VERSION


@dataclass(frozen=True)
class StepResult:
    frame: RiskFrame
    changes: tuple[EventChange, ...]
    alerts: tuple[AlertEvent, ...]
