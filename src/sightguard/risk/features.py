"""Causal normalized geometry and time-based feature extraction, standard library only."""

import math
from collections import deque
from dataclasses import dataclass, field

from sightguard.data import Detection, FrameRecord
from sightguard.risk.config import FeatureConfig
from sightguard.risk.schema import PerceptionQuality, RiskFeatures, SpatialFeatures


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _segment_distance(
    point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = clamp(((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy))
    return math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy)


def focus_relevance(point: tuple[float, float], config: FeatureConfig) -> float:
    """Soft point-to-trapezoid relevance; no real-world path interpretation."""
    x, y = point
    top, bottom = config.focus_top_y, config.focus_bottom_y
    half_top, half_bottom = config.focus_top_half_width, config.focus_bottom_half_width
    cx = config.focus_center_x
    half_width = half_top + (half_bottom - half_top) * clamp((y - top) / (bottom - top))
    if top <= y <= bottom and abs(x - cx) <= half_width:
        return 1.0
    polygon = (
        (cx - half_top, top),
        (cx + half_top, top),
        (cx + half_bottom, bottom),
        (cx - half_bottom, bottom),
    )
    distance = min(_segment_distance(point, polygon[i], polygon[(i + 1) % 4]) for i in range(4))
    return clamp(1 - distance / config.focus_soft_margin)


def spatial_features(
    detection: Detection, width: int, height: int, config: FeatureConfig
) -> SpatialFeatures:
    x1, y1, x2, y2 = detection.bbox_xyxy
    original = (x1 / width, y1 / height, x2 / width, y2 / height)
    box = tuple(clamp(value) for value in original)
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("invalid_bbox: zero area or fully outside the image")
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    bottom = (center[0], y2)
    return SpatialFeatures(
        box,
        center,
        x2 - x1,
        y2 - y1,
        (x2 - x1) * (y2 - y1),
        bottom,
        focus_relevance(bottom, config),
        box != original,
    )


@dataclass
class _History:
    first_ms: float
    last_ms: float | None = None
    spatial: SpatialFeatures | None = None
    observations: int = 0
    opportunities: deque[tuple[float, bool | None]] = field(default_factory=deque)
    rates: tuple[float, float, float, float] | None = None
    motion_baseline_ms: float | None = None
    motion_baseline_spatial: SpatialFeatures | None = None


class FeatureExtractor:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.histories: dict[str, _History] = {}

    def begin_frame(self, frame: FrameRecord, complete: bool) -> None:
        for history in self.histories.values():
            history.opportunities.append((frame.timestamp_ms, False if complete else None))
            while history.opportunities and (
                frame.timestamp_ms - history.opportunities[0][0] > self.config.history_window_ms
            ):
                history.opportunities.popleft()

    def forget(self, event_id: str) -> None:
        self.histories.pop(event_id, None)

    def observe(
        self, event_id: str, detection: Detection, frame: FrameRecord, spatial: SpatialFeatures
    ) -> RiskFeatures:
        history = self.histories.setdefault(event_id, _History(frame.timestamp_ms))
        if not history.opportunities or history.opportunities[-1][0] != frame.timestamp_ms:
            history.opportunities.append((frame.timestamp_ms, True))
        else:
            history.opportunities[-1] = (frame.timestamp_ms, True)
        dt = None if history.last_ms is None else frame.timestamp_ms - history.last_ms
        previous_opportunity = list(history.opportunities)[-2:-1]
        interrupted = history.observations > 0 and (
            not previous_opportunity
            or previous_opportunity[0][1] is not True
            or dt > self.config.motion_max_interval_ms
        )
        rates = None
        motion_dt = (
            None
            if history.motion_baseline_ms is None
            else frame.timestamp_ms - history.motion_baseline_ms
        )
        reset_baseline = (
            interrupted or motion_dt is None or motion_dt > self.config.motion_max_interval_ms
        )
        if (
            not reset_baseline
            and self.config.motion_min_interval_ms
            <= motion_dt
            <= self.config.motion_max_interval_ms
        ):
            seconds = motion_dt / 1000
            previous = history.motion_baseline_spatial
            scale = math.log(math.sqrt(spatial.area / previous.area)) / seconds
            vx = (spatial.center_xy[0] - previous.center_xy[0]) / seconds
            vy = (spatial.center_xy[1] - previous.center_xy[1]) / seconds
            inward = (
                abs(previous.center_xy[0] - self.config.focus_center_x)
                - abs(spatial.center_xy[0] - self.config.focus_center_x)
            ) / seconds
            rates = (scale, vx, vy, inward)
            if self.config.motion_smoothing_tau_ms > 0 and history.rates is not None:
                alpha = 1 - math.exp(-motion_dt / self.config.motion_smoothing_tau_ms)
                rates = tuple(
                    old + alpha * (new - old) for old, new in zip(history.rates, rates, strict=True)
                )
        if reset_baseline or rates is not None:
            history.motion_baseline_ms = frame.timestamp_ms
            history.motion_baseline_spatial = spatial
        if reset_baseline:
            history.rates = None
        elif rates is not None:
            history.rates = rates
        history.observations += 1
        seen = sum(observed is True for _, observed in history.opportunities)
        known = sum(observed is not None for _, observed in history.opportunities)
        continuity = seen / known if known else 0.0
        limitations = []
        if detection.track_id is None:
            limitations.append("missing_track_id")
        if interrupted:
            limitations.append("observation_interrupted")
        if rates is None:
            limitations.append("motion_unavailable")
        if spatial.bbox_clipped:
            limitations.append("bbox_clipped")
        if frame.timestamp_source == "fps_fallback":
            limitations.append("timestamp_fps_fallback")
        quality_score = (
            0.5 * detection.confidence
            + 0.3 * continuity
            + 0.2 * min(1, seen / self.config.quality_observation_target)
        )
        if detection.track_id is None:
            quality_score *= 0.85
        if interrupted:
            quality_score *= 0.75
        if frame.timestamp_source == "fps_fallback":
            quality_score *= self.config.fallback_time_quality
        quality = PerceptionQuality(
            detection.confidence,
            continuity,
            history.observations,
            seen,
            detection.track_id is None,
            interrupted,
            rates is not None,
            frame.timestamp_source,
            clamp(quality_score),
            tuple(limitations),
        )
        result = RiskFeatures(
            spatial,
            frame.timestamp_ms - history.first_ms,
            dt,
            rates[0] if rates is not None else None,
            (rates[1], rates[2]) if rates is not None else None,
            rates[3] if rates is not None else None,
            continuity,
            quality,
            motion_dt if rates is not None else None,
        )
        history.last_ms, history.spatial = frame.timestamp_ms, spatial
        return result
