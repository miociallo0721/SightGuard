"""Translate emitted risk alerts into spatial cues without scoring risk."""

from __future__ import annotations

import math
from collections import OrderedDict

from sightguard.data import FrameRecord
from sightguard.risk.schema import AlertEvent, EventState, RiskLevel, StepResult
from sightguard.spatial.config import SpatialConfig
from sightguard.spatial.coordinates import camera_azimuth
from sightguard.spatial.schemas import CueSource, DirectionReference, SpatialCue

_MAX_REMEMBERED_ALERTS = 4096
_SUPPORTED_ALERT_TYPES = {"first", "escalation", "perception_limited"}


class RiskCueAdapter:
    """Create at most one cue per actual alert identity within this session."""

    def __init__(self, config: SpatialConfig) -> None:
        self.config = config
        self._seen: OrderedDict[str, None] = OrderedDict()

    def convert(
        self,
        record: FrameRecord,
        step: StepResult,
        now_ms: float,
    ) -> tuple[SpatialCue, ...]:
        """Convert current ``first``/``escalation`` and health alerts only.

        Alerts remain audible with a centered direction when matching frame
        geometry is unavailable or stale. This adapter never infers or changes
        risk; it only maps already-emitted ``AlertEvent`` records.
        """
        if (
            isinstance(now_ms, bool)
            or not isinstance(now_ms, (int, float))
            or not math.isfinite(float(now_ms))
        ):
            raise ValueError("now_ms must be finite")
        now = float(now_ms)

        cues: list[SpatialCue] = []
        for alert in step.alerts:
            if alert.alert_type not in _SUPPORTED_ALERT_TYPES:
                continue
            identity = self._identity(alert)
            if identity is None or identity in self._seen:
                continue
            self._remember(identity)

            timestamp = alert.timestamp_ms
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)
                or timestamp < 0
            ):
                continue
            timestamp = float(timestamp)
            expires_at = timestamp + self.config.risk_ttl_ms
            if not math.isfinite(expires_at) or now >= expires_at:
                continue

            if alert.alert_type == "perception_limited":
                cues.append(
                    SpatialCue(
                        cue_id=f"risk-alert:{identity}",
                        source=CueSource.SYSTEM,
                        event_id=alert.event_id or None,
                        cue_type="perception_limited",
                        timestamp_ms=timestamp,
                        azimuth_deg=0.0,
                        elevation_deg=0.0,
                        direction_reference=DirectionReference.LISTENER,
                        direction_confidence=None,
                        priority=self.config.system_priority,
                        expires_at_ms=expires_at,
                        direction_method="unavailable",
                    )
                )
                continue

            direction = self._direction(record, step, alert, now)
            cue_type = "risk_escalation" if alert.alert_type == "escalation" else "risk_initial"
            high_priority = alert.alert_type == "escalation" or alert.risk_level == RiskLevel.HIGH
            cues.append(
                SpatialCue(
                    cue_id=f"risk-alert:{identity}",
                    source=CueSource.RISK,
                    event_id=alert.event_id,
                    cue_type=cue_type,
                    timestamp_ms=timestamp,
                    azimuth_deg=direction[0],
                    elevation_deg=0.0,
                    direction_reference=DirectionReference.CAMERA,
                    direction_confidence=None,
                    priority=(
                        self.config.high_risk_priority
                        if high_priority
                        else self.config.risk_priority
                    ),
                    expires_at_ms=expires_at,
                    direction_method=direction[1],
                )
            )
        return tuple(cues)

    @staticmethod
    def _identity(alert: AlertEvent) -> str | None:
        stream_id, alert_id = alert.stream_id, alert.alert_id
        if not isinstance(stream_id, str) or not stream_id:
            return None
        if not isinstance(alert_id, str) or not alert_id:
            return None
        return f"{len(stream_id)}:{stream_id}:{len(alert_id)}:{alert_id}"

    def _remember(self, identity: str) -> None:
        self._seen[identity] = None
        self._seen.move_to_end(identity)
        if len(self._seen) > _MAX_REMEMBERED_ALERTS:
            self._seen.popitem(last=False)

    def _direction(
        self,
        record: FrameRecord,
        step: StepResult,
        alert: AlertEvent,
        now_ms: float,
    ) -> tuple[float, str]:
        frame = step.frame
        age_ms = now_ms - alert.timestamp_ms
        if (
            record.status != "ok"
            or frame.input_status != "ok"
            or frame.stream_id != alert.stream_id
            or frame.frame_index != record.frame_index
            or frame.timestamp_ms != record.timestamp_ms
            or frame.timestamp_ms != alert.timestamp_ms
            or age_ms < 0
            or age_ms > self.config.max_direction_age_ms
        ):
            return 0.0, "unavailable"

        events = [event for event in frame.events if event.event_id == alert.event_id]
        observations = [
            observation
            for observation in frame.observations
            if observation.event_id == alert.event_id
        ]
        if len(events) != 1 or len(observations) != 1:
            return 0.0, "unavailable"

        event, observation = events[0], observations[0]
        index = observation.detection_index
        if (
            event.state != EventState.ACTIVE
            or not event.confirmed
            or not event.observed_now
            or event.last_seen_ms != alert.timestamp_ms
            or type(index) is not int
            or index < 0
            or index >= len(record.detections)
            or observation.features is None
            or observation.error is not None
            or observation.assessment is None
            or observation.features.spatial is None
        ):
            return 0.0, "unavailable"

        detection = record.detections[index]
        if (
            observation.class_id != detection.class_id
            or observation.class_name != detection.class_name
            or observation.source_track_id != detection.track_id
            or event.class_id != detection.class_id
            or event.class_name != detection.class_name
            or event.last_track_id != detection.track_id
        ):
            return 0.0, "unavailable"

        x1, y1, x2, y2 = detection.bbox_xyxy
        box = tuple(
            max(0, min(1, value))
            for value in (
                x1 / record.frame_width,
                y1 / record.frame_height,
                x2 / record.frame_width,
                y2 / record.frame_height,
            )
        )
        if (
            box[2] <= box[0]
            or box[3] <= box[1]
            or not all(
                math.isclose(a, b, abs_tol=1e-9)
                for a, b in zip(box, observation.features.spatial.bbox_normalized, strict=True)
            )
        ):
            return 0.0, "unavailable"

        try:
            return camera_azimuth(
                observation.features.spatial.center_xy[0],
                horizontal_fov_deg=self.config.horizontal_fov_deg,
                approximate_half_angle_deg=self.config.approximate_half_angle_deg,
            )
        except (IndexError, TypeError, ValueError):
            return 0.0, "unavailable"
