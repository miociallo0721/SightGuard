"""Conservative deterministic one-to-one association and risk event lifecycle."""

import math
from collections import Counter
from dataclasses import dataclass, field

from sightguard.data import Detection, FrameRecord
from sightguard.risk.config import AssociationConfig
from sightguard.risk.schema import (
    EventChange,
    EventSnapshot,
    EventState,
    RiskAssessment,
    SpatialFeatures,
)


def bbox_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class RiskEvent:
    event_id: str
    class_id: int
    class_name: str
    first_seen_ms: float
    last_seen_ms: float
    spatial: SpatialFeatures
    last_track_id: int | None
    state: EventState = EventState.CANDIDATE
    confirmed: bool = False
    valid_observations: int = 0
    confirmation_start_ms: float | None = None
    confirmation_count: int = 0
    source_track_ids: list[int] = field(default_factory=list)
    observed_now: bool = False
    assessment: RiskAssessment | None = None

    def snapshot(self) -> EventSnapshot:
        return EventSnapshot(
            self.event_id,
            self.state,
            self.class_id,
            self.class_name,
            self.first_seen_ms,
            self.last_seen_ms,
            self.valid_observations,
            self.confirmed,
            tuple(self.source_track_ids),
            self.last_track_id,
            self.observed_now,
            self.assessment,
        )


@dataclass(frozen=True)
class Association:
    detection_index: int
    event: RiskEvent
    method: str
    created: bool


class EventManager:
    def __init__(self, config: AssociationConfig, stream_id: str) -> None:
        self.config, self.stream_id = config, stream_id
        self.events: dict[str, RiskEvent] = {}
        self.next_id = 1

    def _change(
        self,
        event: RiskEvent,
        timestamp: float,
        action: str,
        reason: str,
        method: str | None = None,
    ) -> EventChange:
        return EventChange(
            self.stream_id,
            event.event_id,
            timestamp,
            "risk",
            action,
            event.state,
            reason,
            event.last_track_id,
            method,
            event.snapshot(),
        )

    def prepare(self, timestamp: float, complete: bool) -> tuple[EventChange, ...]:
        changes = []
        for event in self.live_events():
            event.observed_now = False
            if complete and timestamp - event.last_seen_ms > self.config.recovery_window_ms:
                event.state = EventState.RESOLVED
                changes.append(
                    self._change(event, timestamp, "resolved", "observation_window_expired")
                )
        return tuple(changes)

    def live_events(self) -> list[RiskEvent]:
        return [
            self.events[key]
            for key in sorted(self.events)
            if self.events[key].state != EventState.RESOLVED
        ]

    def associate(
        self, frame: FrameRecord, spatial: dict[int, SpatialFeatures]
    ) -> list[Association]:
        ordered = sorted(
            spatial,
            key=lambda i: (
                frame.detections[i].class_id,
                spatial[i].bbox_normalized,
                -1 if frame.detections[i].track_id is None else frame.detections[i].track_id,
                -frame.detections[i].confidence,
                i,
            ),
        )
        live = self.live_events()
        available_indices, available_events = set(ordered), {e.event_id for e in live}
        assigned: dict[int, tuple[RiskEvent, str]] = {}
        detection_ids = Counter(
            frame.detections[i].track_id
            for i in ordered
            if frame.detections[i].track_id is not None
        )
        event_ids = Counter(e.last_track_id for e in live if e.last_track_id is not None)
        for method in ("track_id", "geometry"):
            edges = []
            for index in ordered:
                if index not in available_indices:
                    continue
                detection, geometry = frame.detections[index], spatial[index]
                for event in live:
                    if event.event_id not in available_events:
                        continue
                    class_switch = event.class_id != detection.class_id
                    if class_switch and not (
                        method == "track_id" and self.config.allow_class_switch_for_track_id
                    ):
                        continue
                    dt = frame.timestamp_ms - event.last_seen_ms
                    if not 0 <= dt <= self.config.recovery_window_ms:
                        continue
                    distance = math.dist(geometry.center_xy, event.spatial.center_xy)
                    overlap = bbox_iou(geometry.bbox_normalized, event.spatial.bbox_normalized)
                    if method == "track_id":
                        if (
                            detection.track_id is None
                            or event.last_track_id != detection.track_id
                            or detection_ids[detection.track_id] != 1
                            or event_ids[detection.track_id] != 1
                            or distance > self.config.track_max_center_distance
                            or (class_switch and overlap < self.config.geometry_min_iou)
                        ):
                            continue
                        score = overlap
                    else:
                        if (
                            distance > self.config.geometry_max_center_distance
                            or overlap < self.config.geometry_min_iou
                        ):
                            continue
                        score = self.config.iou_weight * overlap + (1 - self.config.iou_weight) * (
                            1 - distance / self.config.geometry_max_center_distance
                        )
                        if score < self.config.geometry_min_score:
                            continue
                    edges.append((score, event.event_id, index))
            # Mutual unique best pairs; ambiguous endpoints stay unmatched.
            by_detection: dict[int, list[tuple[float, str, int]]] = {}
            by_event: dict[str, list[tuple[float, str, int]]] = {}
            for edge in sorted(edges, key=lambda pair: (-pair[0], pair[1], pair[2])):
                by_detection.setdefault(edge[2], []).append(edge)
                by_event.setdefault(edge[1], []).append(edge)
            for edge in sorted(edges, key=lambda pair: (-pair[0], pair[1], pair[2])):
                score, event_id, index = edge
                left, right = by_detection[index], by_event[event_id]
                if edge != left[0] or edge != right[0]:
                    continue
                if any(
                    len(options) > 1 and score - options[1][0] <= self.config.ambiguity_margin
                    for options in (left, right)
                ):
                    continue
                event = self.events[event_id]
                association_method = (
                    "track_id_class_switch"
                    if method == "track_id" and event.class_id != frame.detections[index].class_id
                    else method
                )
                assigned[index] = (event, association_method)
                available_indices.remove(index)
                available_events.remove(event_id)
        result = []
        for index in ordered:
            created = index not in assigned
            if created:
                detection = frame.detections[index]
                event = RiskEvent(
                    f"E{self.next_id:06d}",
                    detection.class_id,
                    detection.class_name,
                    frame.timestamp_ms,
                    frame.timestamp_ms,
                    spatial[index],
                    detection.track_id,
                )
                self.next_id += 1
                self.events[event.event_id] = event
                method = "new"
            else:
                event, method = assigned[index]
            result.append(Association(index, event, method, created))
        return result

    def observe(
        self,
        association: Association,
        detection: Detection,
        spatial: SpatialFeatures,
        assessment: RiskAssessment,
        timestamp: float,
    ) -> tuple[EventChange, ...]:
        event = association.event
        was_lost = event.state == EventState.LOST
        if (
            event.confirmation_start_ms is None
            or timestamp - event.last_seen_ms > self.config.confirmation_max_gap_ms
        ):
            event.confirmation_start_ms, event.confirmation_count = timestamp, 0
        event.confirmation_count += 1
        event.valid_observations += 1
        event.last_seen_ms, event.spatial = timestamp, spatial
        if association.method == "track_id_class_switch":
            event.class_id, event.class_name = detection.class_id, detection.class_name
        event.last_track_id, event.observed_now, event.assessment = (
            detection.track_id,
            True,
            assessment,
        )
        if detection.track_id is not None and detection.track_id not in event.source_track_ids:
            event.source_track_ids.append(detection.track_id)
        if was_lost:
            event.state = EventState.ACTIVE if event.confirmed else EventState.CANDIDATE
        action = "created" if association.created else "recovered" if was_lost else "updated"
        changes = [self._change(event, timestamp, action, "valid_observation", association.method)]
        if (
            not event.confirmed
            and event.confirmation_count >= self.config.confirmation_observations
            and timestamp - event.confirmation_start_ms >= self.config.confirmation_ms
        ):
            event.confirmed, event.state = True, EventState.ACTIVE
            changes.append(self._change(event, timestamp, "activated", "candidate_confirmed"))
        return tuple(changes)

    def end_frame(self, timestamp: float, complete: bool) -> tuple[EventChange, ...]:
        changes = []
        for event in self.live_events():
            if event.observed_now:
                continue
            event.confirmation_start_ms, event.confirmation_count = None, 0
            if complete and event.state != EventState.LOST:
                event.state = EventState.LOST
                changes.append(
                    self._change(event, timestamp, "lost", "not_observed_in_complete_frame")
                )
        return tuple(changes)

    def finish(self, timestamp: float) -> tuple[EventChange, ...]:
        return tuple(
            self._change(event, timestamp, "censored", "input_ended_without_resolution")
            for event in self.live_events()
        )
