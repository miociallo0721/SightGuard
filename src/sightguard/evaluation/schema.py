"""Strict independent annotation and strategy output interfaces (schema 3.0)."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from sightguard.risk.schema import JsonRecord, RiskLevel

SCHEMA_VERSION = "3.0"


def finite_time(value: float, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


def text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class ExcludedInterval:
    start_ms: float
    end_ms: float
    reason: str

    def __post_init__(self) -> None:
        finite_time(self.start_ms, "start_ms")
        finite_time(self.end_ms, "end_ms")
        if self.end_ms <= self.start_ms:
            raise ValueError("Excluded interval must have positive duration")
        text(self.reason, "exclusion reason")


@dataclass(frozen=True)
class TargetRegion:
    start_ms: float
    end_ms: float
    bbox_normalized: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        finite_time(self.start_ms, "region start_ms")
        finite_time(self.end_ms, "region end_ms")
        if self.end_ms <= self.start_ms:
            raise ValueError("Target region duration must be positive")
        box = tuple(self.bbox_normalized)
        if (
            len(box) != 4
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in box
            )
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            raise ValueError("Target bbox must be a positive-area normalized xyxy box")
        object.__setattr__(self, "bbox_normalized", box)


@dataclass(frozen=True)
class AlertOpportunity:
    opportunity_id: str
    kind: str
    window_start_ms: float
    deadline_ms: float
    minimum_severity: RiskLevel

    def __post_init__(self) -> None:
        text(self.opportunity_id, "opportunity_id")
        if self.kind not in ("initial", "escalation"):
            raise ValueError("Opportunity kind must be initial or escalation")
        finite_time(self.window_start_ms, "window_start_ms")
        finite_time(self.deadline_ms, "deadline_ms")
        if self.deadline_ms < self.window_start_ms:
            raise ValueError("Opportunity deadline precedes window")
        object.__setattr__(self, "minimum_severity", RiskLevel(self.minimum_severity))


@dataclass(frozen=True)
class LabeledEvent:
    video_id: str
    event_id: str
    event_type: str
    start_ms: float
    end_ms: float
    severity: RiskLevel
    annotation_status: str
    alert_opportunities: tuple[AlertOpportunity, ...]
    annotator_id: str
    notes: str
    class_ids: tuple[int, ...]
    target_regions: tuple[TargetRegion, ...] = ()

    def __post_init__(self) -> None:
        for name in ("video_id", "event_id", "event_type", "annotator_id"):
            text(getattr(self, name), name)
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")
        finite_time(self.start_ms, "event start_ms")
        finite_time(self.end_ms, "event end_ms")
        if self.end_ms <= self.start_ms:
            raise ValueError("Event duration must be positive")
        object.__setattr__(self, "severity", RiskLevel(self.severity))
        if self.annotation_status not in ("confirmed", "uncertain", "excluded"):
            raise ValueError("Event status must be confirmed, uncertain, or excluded")
        if not self.class_ids or any(type(c) is not int or c < 0 for c in self.class_ids):
            raise ValueError("Independent semantic class_ids are required")
        object.__setattr__(self, "class_ids", tuple(self.class_ids))
        object.__setattr__(self, "alert_opportunities", tuple(self.alert_opportunities))
        object.__setattr__(self, "target_regions", tuple(self.target_regions))
        ids = [op.opportunity_id for op in self.alert_opportunities]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate opportunity IDs")
        initials = [op for op in self.alert_opportunities if op.kind == "initial"]
        if self.annotation_status == "confirmed" and len(initials) != 1:
            raise ValueError(
                "Confirmed alert-worthy events require exactly one initial opportunity"
            )
        for op in self.alert_opportunities:
            if not self.start_ms <= op.window_start_ms <= op.deadline_ms <= self.end_ms:
                raise ValueError("Opportunity lies outside event interval")
            if op.minimum_severity.rank > self.severity.rank:
                raise ValueError("Opportunity severity exceeds event maximum severity")
            if (
                initials
                and op.kind == "escalation"
                and (
                    op.window_start_ms < initials[0].window_start_ms
                    or op.minimum_severity.rank <= initials[0].minimum_severity.rank
                )
            ):
                raise ValueError("Escalation must follow initial onset and require higher severity")
        for region in self.target_regions:
            if not self.start_ms <= region.start_ms < region.end_ms <= self.end_ms:
                raise ValueError("Target region lies outside event")


@dataclass(frozen=True)
class VideoAnnotations(JsonRecord):
    video_id: str
    duration_ms: float
    data_kind: str
    annotation_status: str
    annotator_id: str
    label_source: str
    source_video_sha256: str
    notes: str
    events: tuple[LabeledEvent, ...]
    excluded_intervals: tuple[ExcludedInterval, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        text(self.video_id, "video_id")
        text(self.annotator_id, "annotator_id")
        finite_time(self.duration_ms, "duration_ms")
        if self.duration_ms <= 0 or self.schema_version != SCHEMA_VERSION:
            raise ValueError("Positive duration and annotation schema 3.0 required")
        if self.data_kind not in ("real", "synthetic") or self.annotation_status not in (
            "draft",
            "reviewed",
        ):
            raise ValueError("Invalid data kind or document annotation status")
        if not isinstance(self.notes, str) or not isinstance(self.source_video_sha256, str):
            raise ValueError("Invalid annotation provenance")
        if self.annotation_status == "reviewed":
            required = "human_raw_video" if self.data_kind == "real" else "synthetic_scenario"
            if self.label_source != required or self.annotator_id == "UNASSIGNED":
                raise ValueError(
                    "Reviewed labels require explicit independent provenance and author"
                )
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "excluded_intervals", tuple(self.excluded_intervals))
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("Duplicate ground-truth event_id")
        for event in self.events:
            if event.video_id != self.video_id or event.end_ms > self.duration_ms:
                raise ValueError("Event video or duration mismatch")
            if any(op.window_start_ms >= self.duration_ms for op in event.alert_opportunities):
                raise ValueError("Opportunity must start before the exclusive video end")
        if any(interval.end_ms > self.duration_ms for interval in self.excluded_intervals):
            raise ValueError("Excluded interval outside video duration")


def load_annotations(path: Path, *, require_reviewed: bool = True) -> VideoAnnotations:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Annotation document must use schema 3.0")
        converted = []
        for item in raw["events"]:
            event = dict(item)
            event["alert_opportunities"] = tuple(
                AlertOpportunity(**op) for op in event["alert_opportunities"]
            )
            event["target_regions"] = tuple(
                TargetRegion(**region) for region in event.get("target_regions", [])
            )
            converted.append(LabeledEvent(**event))
        raw["events"] = tuple(converted)
        raw["excluded_intervals"] = tuple(
            ExcludedInterval(**item) for item in raw.get("excluded_intervals", [])
        )
        result = VideoAnnotations(**raw)
        if require_reviewed and result.annotation_status != "reviewed":
            raise ValueError("Draft annotation is not an evaluated negative video; review it first")
        return result
    except (TypeError, KeyError) as exc:
        raise ValueError(f"Invalid annotation structure: {exc}") from exc


@dataclass(frozen=True)
class StrategyAlert(JsonRecord):
    video_id: str
    strategy: str
    alert_id: str
    event_id: str
    timestamp_ms: float
    risk_level: RiskLevel | None
    risk_score: float | None
    alert_type: str
    reason: str
    source_track_ids: tuple[int, ...]
    class_id: int | None
    bbox_normalized: tuple[float, float, float, float] | None
    frame_index: int = 0
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("video_id", "strategy", "alert_id", "event_id", "reason"):
            text(getattr(self, name), name)
        finite_time(self.timestamp_ms, "alert timestamp")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("Alert frame_index must be a nonnegative integer")
        if self.class_id is not None and (type(self.class_id) is not int or self.class_id < 0):
            raise ValueError("Alert class_id must be a nonnegative integer")
        if any(type(identity) is not int or identity < 0 for identity in self.source_track_ids):
            raise ValueError("Alert source IDs must be nonnegative integers")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported strategy alert schema")
        if self.risk_level is not None:
            object.__setattr__(self, "risk_level", RiskLevel(self.risk_level))
        if self.risk_score is not None and (
            not math.isfinite(self.risk_score) or not 0 <= self.risk_score <= 1
        ):
            raise ValueError("Invalid strategy risk score")
        if self.alert_type not in (
            "first",
            "escalation",
            "repeat",
            "trigger",
            "perception_limited",
        ):
            raise ValueError("Invalid strategy alert type")
        if self.alert_type != "perception_limited" and (
            self.risk_level is None or self.class_id is None
        ):
            raise ValueError("Scene alerts require semantic class and severity")
        if self.bbox_normalized is not None:
            box = TargetRegion(0, 1, self.bbox_normalized).bbox_normalized
            object.__setattr__(self, "bbox_normalized", box)
        object.__setattr__(self, "source_track_ids", tuple(self.source_track_ids))


@dataclass(frozen=True)
class MatchRecord(JsonRecord):
    video_id: str
    strategy: str
    alert_id: str | None
    timestamp_ms: float | None
    status: str
    gt_event_id: str | None
    opportunity_id: str | None
    opportunity_kind: str | None
    latency_ms: float | None
    reason: str
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class EvaluationResult:
    video_id: str
    strategy: str
    metrics: dict
    matches: tuple[MatchRecord, ...]
    event_rows: tuple[dict, ...]
