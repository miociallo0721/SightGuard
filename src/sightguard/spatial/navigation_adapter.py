"""Translate explicitly simulated navigation directions into spatial cues."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from sightguard.spatial.config import SpatialConfig
from sightguard.spatial.coordinates import normalize_azimuth
from sightguard.spatial.schemas import CueSource, DirectionReference, SpatialCue


@dataclass(frozen=True, slots=True)
class NavigationCue:
    """Serializable simulated direction for a goal, with an optional TTL override."""

    goal_id: str
    azimuth_deg: float
    elevation_deg: float = 0.0
    lifetime_ms: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.goal_id, str) or not self.goal_id.strip():
            raise ValueError("goal_id must be a non-empty string")
        azimuth = normalize_azimuth(self.azimuth_deg)
        if (
            isinstance(self.elevation_deg, bool)
            or not isinstance(self.elevation_deg, (int, float))
            or not math.isfinite(self.elevation_deg)
            or not -90 <= self.elevation_deg <= 90
        ):
            raise ValueError("elevation_deg must be finite and in -90..90")
        object.__setattr__(self, "azimuth_deg", azimuth)
        if self.lifetime_ms is not None and (
            isinstance(self.lifetime_ms, bool)
            or not isinstance(self.lifetime_ms, (int, float))
            or not math.isfinite(self.lifetime_ms)
            or self.lifetime_ms <= 0
        ):
            raise ValueError("lifetime_ms must be finite and positive")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> NavigationCue:
        raw = json.loads(value)
        if not isinstance(raw, dict):
            raise ValueError("Navigation cue JSON must contain an object")
        return cls(**raw)


class NavigationAdapter:
    """Map simulated goal directions to listener-relative navigation cues."""

    def __init__(self, config: SpatialConfig) -> None:
        self.config = config

    def convert(self, cue: NavigationCue, now_ms: float) -> SpatialCue:
        if (
            isinstance(now_ms, bool)
            or not isinstance(now_ms, (int, float))
            or not math.isfinite(float(now_ms))
            or now_ms < 0
        ):
            raise ValueError("now_ms must be finite and non-negative")
        issued_at_ms = float(now_ms)
        lifetime_ms = self.config.navigation_ttl_ms if cue.lifetime_ms is None else cue.lifetime_ms
        cue_identity = f"navigation-goal:{len(cue.goal_id)}:{cue.goal_id}:{issued_at_ms.hex()}"
        return SpatialCue(
            cue_id=cue_identity,
            source=CueSource.NAVIGATION,
            event_id=cue.goal_id,
            cue_type="navigation",
            timestamp_ms=issued_at_ms,
            azimuth_deg=cue.azimuth_deg,
            elevation_deg=cue.elevation_deg,
            direction_reference=DirectionReference.LISTENER,
            direction_confidence=None,
            priority=self.config.navigation_priority,
            expires_at_ms=issued_at_ms + lifetime_ms,
            direction_method="simulated",
        )
