"""Alert eligibility and stable priority, independent of perception and audio hardware."""

from dataclasses import dataclass

from sightguard.decision.health import PerceptionMonitor
from sightguard.risk.config import AlertConfig
from sightguard.risk.schema import AlertEvent, EventSnapshot, EventState, RiskLevel


@dataclass
class _AlertHistory:
    last_ms: float
    highest_level: RiskLevel
    highest_score: float


class AlertScheduler:
    def __init__(self, config: AlertConfig, stream_id: str) -> None:
        self.config, self.stream_id = config, stream_id
        self.history: dict[str, _AlertHistory] = {}
        self.next_id = 1

    def schedule(
        self, events: tuple[EventSnapshot, ...], monitor: PerceptionMonitor, timestamp: float
    ) -> tuple[AlertEvent, ...]:
        # Higher level, focus relevance, risk score, then deterministic event ID.
        eligible = []
        for event in events:
            risk = event.assessment
            if (
                event.state != EventState.ACTIVE
                or not event.observed_now
                or risk is None
                or risk.level.rank < RiskLevel(self.config.minimum_level).rank
            ):
                continue
            history = self.history.get(event.event_id)
            alert_type = None
            if history is None:
                alert_type = "first"
            elif (
                risk.level.rank > history.highest_level.rank
                and risk.score >= history.highest_score + self.config.escalation_min_score_delta
                and timestamp - history.last_ms >= self.config.escalation_interval_ms
            ):
                alert_type = "escalation"
            elif (
                self.config.repeat_enabled
                and timestamp - history.last_ms >= self.config.ordinary_interval_ms
            ):
                alert_type = "repeat"
            if alert_type is not None:
                eligible.append(
                    (
                        (-risk.level.rank, -risk.spatial_score, -risk.score, event.event_id),
                        event,
                        alert_type,
                    )
                )
        alerts = []
        health = monitor.current
        # Persistent unavailable perception gets priority over scene-specific alerts.
        if health.state == EventState.ACTIVE and not monitor.alerted:
            alerts.append(
                AlertEvent(
                    self.stream_id,
                    f"A{self.next_id:06d}",
                    health.event_id,
                    timestamp,
                    None,
                    None,
                    "perception_limited",
                    f"{health.status}: {health.reason}; environment unknown",
                    (),
                )
            )
            self.next_id += 1
            monitor.alerted = True
        for _, event, alert_type in sorted(eligible, key=lambda candidate: candidate[0]):
            if len(alerts) >= self.config.max_alerts_per_frame:
                break
            risk = event.assessment
            reason = "; ".join(risk.reasons)
            if risk.quality.score < self.config.limited_quality_threshold:
                reason += "; perception quality limited; risk is not reduced by quality"
            alerts.append(
                AlertEvent(
                    self.stream_id,
                    f"A{self.next_id:06d}",
                    event.event_id,
                    timestamp,
                    risk.level,
                    risk.score,
                    alert_type,
                    reason,
                    event.source_track_ids,
                    risk.quality.score,
                )
            )
            self.next_id += 1
            previous = self.history.get(event.event_id)
            highest = (
                risk.level
                if previous is None
                else max(risk.level, previous.highest_level, key=lambda level: level.rank)
            )
            highest_score = (
                risk.score if previous is None else max(risk.score, previous.highest_score)
            )
            self.history[event.event_id] = _AlertHistory(timestamp, highest, highest_score)
        return tuple(alerts)
