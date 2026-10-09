"""Independent perception-limited episodes. An unavailable scene is not an empty scene."""

from sightguard.risk.config import AlertConfig
from sightguard.risk.schema import EventChange, EventState, PerceptionStatus


class PerceptionMonitor:
    def __init__(self, config: AlertConfig, stream_id: str) -> None:
        self.config, self.stream_id = config, stream_id
        self.next_id = 1
        self.current = PerceptionStatus("available", None, None, None, None)
        self.alerted = False

    def update(self, timestamp: float, status: str, reason: str | None) -> tuple[EventChange, ...]:
        previous = self.current
        changes = []
        if status == "available":
            if previous.event_id is not None:
                changes.append(
                    EventChange(
                        self.stream_id,
                        previous.event_id,
                        timestamp,
                        "perception",
                        "resolved",
                        EventState.RESOLVED,
                        "perception_recovered",
                    )
                )
            self.current = PerceptionStatus(status, None, None, None, None)
            self.alerted = False
            return tuple(changes)
        if previous.event_id is None:
            event_id = f"H{self.next_id:06d}"
            self.next_id += 1
            previous = PerceptionStatus(status, event_id, EventState.CANDIDATE, timestamp, reason)
            self.alerted = False
            changes.append(
                EventChange(
                    self.stream_id,
                    event_id,
                    timestamp,
                    "perception",
                    "created",
                    EventState.CANDIDATE,
                    reason or status,
                )
            )
        state = previous.state
        if state == EventState.CANDIDATE and (
            timestamp - previous.since_ms >= self.config.perception_confirm_ms
        ):
            state = EventState.ACTIVE
            changes.append(
                EventChange(
                    self.stream_id,
                    previous.event_id,
                    timestamp,
                    "perception",
                    "activated",
                    state,
                    "persistent_perception_limitation",
                )
            )
        elif previous.status != status:
            changes.append(
                EventChange(
                    self.stream_id,
                    previous.event_id,
                    timestamp,
                    "perception",
                    "updated",
                    state,
                    reason or status,
                )
            )
        self.current = PerceptionStatus(status, previous.event_id, state, previous.since_ms, reason)
        return tuple(changes)

    def finish(self, timestamp: float) -> tuple[EventChange, ...]:
        if self.current.event_id is None:
            return ()
        return (
            EventChange(
                self.stream_id,
                self.current.event_id,
                timestamp,
                "perception",
                "censored",
                self.current.state,
                "input_ended_with_perception_limitation",
            ),
        )
