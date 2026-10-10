"""Deterministic bounded scheduling core; one audible source at a time."""

from collections import deque
from dataclasses import dataclass

from sightguard.spatial.schemas import CueSource, SpatialCue


@dataclass(frozen=True)
class PlaybackDecision:
    stop: bool = False
    cue: SpatialCue | None = None
    reason: str = "idle"


class CueScheduler:
    def __init__(self, capacity: int = 16) -> None:
        if type(capacity) is not int or not 1 <= capacity <= 256:
            raise ValueError("Invalid queue capacity")
        self.capacity = capacity
        self.pending: list[SpatialCue] = []
        self.active: SpatialCue | None = None
        self.navigation_paused = False
        self.seen: set[str] = set()
        self.history: deque[str] = deque()
        self.cancelled: set[str] = set()

    def submit(self, cue: SpatialCue, now_ms: float) -> bool:
        self.pending = [item for item in self.pending if item.expires_at_ms > now_ms]
        if cue.cue_id in self.seen or cue.expires_at_ms <= now_ms or cue.timestamp_ms > now_ms:
            return False
        if len(self.pending) >= self.capacity:
            worst = max(
                self.pending, key=lambda item: (-item.priority, item.timestamp_ms, item.cue_id)
            )
            if cue.priority <= worst.priority:
                return False
            self.pending.remove(worst)
        self.pending.append(cue)
        self.seen.add(cue.cue_id)
        self.history.append(cue.cue_id)
        if len(self.history) > 4096:
            self.seen.discard(self.history.popleft())
        return True

    def cancel(self, cue_id: str) -> None:
        self.pending = [item for item in self.pending if item.cue_id != cue_id]
        if self.active and self.active.cue_id == cue_id:
            self.cancelled.add(cue_id)

    def cancel_navigation(self) -> None:
        self.pending = [item for item in self.pending if item.source != CueSource.NAVIGATION]
        if self.active and self.active.source == CueSource.NAVIGATION:
            self.cancelled.add(self.active.cue_id)

    def clear(self) -> None:
        self.pending.clear()
        self.active = None
        self.cancelled.clear()

    def _resume_later(self, cue: SpatialCue) -> None:
        # The active source is outside the queue; never exceed queue capacity on preemption.
        if len(self.pending) < self.capacity:
            self.pending.append(cue)

    def update(self, now_ms: float, *, playing: bool) -> PlaybackDecision:
        self.pending = [item for item in self.pending if item.expires_at_ms > now_ms]
        stopped = False
        if self.active:
            active = self.active
            if active.expires_at_ms <= now_ms or active.cue_id in self.cancelled:
                self.cancelled.discard(active.cue_id)
                self.active = None
                stopped = playing
            elif not playing:
                self.active = None
            elif self.navigation_paused and active.source == CueSource.NAVIGATION:
                self._resume_later(active)
                self.active = None
                stopped = playing
        eligible = [
            item
            for item in self.pending
            if not (self.navigation_paused and item.source == CueSource.NAVIGATION)
        ]
        best = (
            min(eligible, key=lambda item: (-item.priority, item.timestamp_ms, item.cue_id))
            if eligible
            else None
        )
        if self.active and best and best.priority > self.active.priority:
            if self.active.source == CueSource.NAVIGATION:
                # Restart the short navigation cue only if its original deadline still holds.
                self._resume_later(self.active)
            self.active = None
            stopped = playing
        if self.active is None and best:
            self.pending.remove(best)
            self.active = best
            return PlaybackDecision(stopped, best, "preempt" if stopped else "play")
        return PlaybackDecision(stopped, reason="stop" if stopped else "idle")
