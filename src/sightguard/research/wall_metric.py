"""Same-frame optical-axis depth statistics; never calibrated distance or speech."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from sightguard.scene.spatial_grid.schema import FrameKey


@dataclass(frozen=True)
class WallMetricConfig:
    minimum_confidence: float = 0.55
    minimum_component_pixels: int = 64
    minimum_roi_coverage: float = 0.08
    minimum_valid_fraction: float = 0.75
    maximum_source_age_ms: float = 500
    roi_xyxy: tuple[float, float, float, float] = (0.25, 0.2, 0.75, 0.9)

    def __post_init__(self) -> None:
        for value in (
            self.minimum_confidence,
            self.minimum_roi_coverage,
            self.minimum_valid_fraction,
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("invalid wall fraction")
        if (
            self.minimum_component_pixels < 1
            or not math.isfinite(self.maximum_source_age_ms)
            or self.maximum_source_age_ms <= 0
        ):
            raise ValueError("invalid wall threshold")
        x1, y1, x2, y2 = self.roi_xyxy
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError("invalid wall ROI")


@dataclass(frozen=True)
class MetricDepthField:
    key: FrameKey
    values: np.ndarray
    model_id: str
    max_depth_m: float
    quantity: str = "predicted_optical_axis_z_m"
    source_age_ms: float = 0
    model_latency_ms: float = 0


@dataclass(frozen=True)
class WallMetricResult:
    key: FrameKey
    status: str
    wall_present: bool | None
    predicted_metric_depth: float | None
    median_m: float | None
    p10_m: float | None
    p90_m: float | None
    valid_pixel_fraction: float
    wall_coverage: float
    semantic_confidence: float | None
    component_count: int
    wall_pixels: int
    source_frame_age_ms: float
    model_latency_ms: float | None
    rejection_reasons: tuple[str, ...]
    model_id: str | None
    quantity: str = "predicted_optical_axis_z_m_not_euclidean_range"
    uncertainty_status: str = "uncalibrated_P10_P90_are_spatial_not_accuracy_bounds"
    automatic_alert_enabled: bool = False
    schema_version: str = "sightguard.wall-metric.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WallMetricEstimator:
    def __init__(self, config: WallMetricConfig | None = None) -> None:
        self.config = config or WallMetricConfig()

    def mask(
        self,
        key: FrameKey,
        classes: np.ndarray,
        confidence: np.ndarray,
        foreground_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int, float, float | None]:
        shape = (key.height, key.width)
        if classes.shape != shape or confidence.shape != shape:
            raise ValueError("semantic_source_geometry_mismatch")
        valid = np.isfinite(confidence) & (confidence >= self.config.minimum_confidence)
        valid &= confidence <= 1
        wall = ((classes == 0) & valid).astype(np.uint8)
        if foreground_mask is not None:
            if foreground_mask.shape != shape:
                raise ValueError("foreground_source_geometry_mismatch")
            wall[foreground_mask.astype(bool)] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(wall, connectivity=8)
        keep = [
            i
            for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= self.config.minimum_component_pixels
        ]
        wall = np.isin(labels, keep)
        x1, y1, x2, y2 = self.config.roi_xyxy
        roi = np.zeros(shape, bool)
        roi[
            int(y1 * key.height) : int(y2 * key.height), int(x1 * key.width) : int(x2 * key.width)
        ] = True
        selected = wall & roi
        coverage = float(selected.sum() / max(1, roi.sum()))
        mean_confidence = float(confidence[selected].mean()) if selected.any() else None
        return selected, len(keep), coverage, mean_confidence

    def estimate(
        self,
        key: FrameKey,
        classes: np.ndarray,
        confidence: np.ndarray,
        depth: MetricDepthField | None,
        *,
        semantic_key: FrameKey | None = None,
        semantic_age_ms: float = 0,
        foreground_mask: np.ndarray | None = None,
        semantic_from_model: bool = True,
        scene_current: bool = True,
    ) -> WallMetricResult:
        reasons: list[str] = []
        selected = np.zeros((key.height, key.width), bool)
        components, coverage, support = 0, 0.0, None
        if semantic_key != key:
            reasons.append("semantic_frame_mismatch")
        if not semantic_from_model:
            reasons.append("semantic_source_not_real_model")
        if not math.isfinite(semantic_age_ms) or not (
            0 <= semantic_age_ms <= self.config.maximum_source_age_ms
        ):
            reasons.append("semantic_source_stale")
        if not scene_current:
            reasons.append("source_scene_changed")
        try:
            selected, components, coverage, support = self.mask(
                key, classes, confidence, foreground_mask
            )
        except ValueError as exc:
            reasons.append(str(exc))
        if coverage < self.config.minimum_roi_coverage:
            reasons.append("insufficient_central_wall_support")
        semantic_rejected = bool(reasons)
        present: bool | None = None if semantic_rejected else True
        if "insufficient_central_wall_support" in reasons and len(reasons) == 1:
            present = False
        values = np.empty(0)
        fraction = 0.0
        if depth is None:
            reasons.append("metric_depth_unavailable")
        elif depth.key != key:
            reasons.append("depth_frame_mismatch")
        elif depth.quantity != "predicted_optical_axis_z_m":
            reasons.append("depth_is_not_metric_z")
        elif depth.values.shape != selected.shape:
            reasons.append("depth_geometry_mismatch")
        else:
            if not depth.model_id:
                reasons.append("depth_model_identity_missing")
            if not math.isfinite(depth.model_latency_ms) or depth.model_latency_ms < 0:
                reasons.append("depth_latency_invalid")
            if not math.isfinite(depth.source_age_ms) or not (
                0 <= depth.source_age_ms <= self.config.maximum_source_age_ms
            ):
                reasons.append("depth_source_stale")
            if not math.isfinite(depth.max_depth_m) or depth.max_depth_m <= 0:
                reasons.append("invalid_model_depth_range")
            raw = depth.values[selected]
            valid = np.isfinite(raw) & (raw > 0) & (raw < depth.max_depth_m * 0.99)
            values = raw[valid]
            fraction = float(valid.mean()) if valid.size else 0.0
            if fraction < self.config.minimum_valid_fraction:
                reasons.append("insufficient_valid_metric_pixels")
        p10, median, p90 = (
            map(float, np.quantile(values, [0.1, 0.5, 0.9])) if values.size else (None, None, None)
        )
        if median is not None and (p90 - p10) / median > 0.8:
            reasons.append("heterogeneous_wall_depth_or_multiple_planes")
        return WallMetricResult(
            key,
            "ABSTAIN" if reasons else "PREDICTED_METRIC_DEPTH",
            present,
            median if not reasons else None,
            median,
            p10,
            p90,
            fraction,
            coverage,
            support,
            components,
            int(selected.sum()),
            max(semantic_age_ms, depth.source_age_ms if depth else 0),
            depth.model_latency_ms if depth else None,
            tuple(reasons),
            depth.model_id if depth else None,
        )


def depth_errors(prediction: np.ndarray, truth_z_m: np.ndarray) -> dict[str, Any]:
    """Evaluate only finite positive registered metric Z; no fitted scale alignment."""
    if prediction.shape != truth_z_m.shape:
        raise ValueError("metric_truth_geometry_mismatch")
    valid = np.isfinite(prediction) & (prediction > 0)
    valid &= np.isfinite(truth_z_m) & (truth_z_m > 0)
    difference = prediction[valid].astype(np.float64) - truth_z_m[valid]
    absolute = np.abs(difference)
    result: dict[str, Any] = {
        "valid_pixels": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
    }
    if not absolute.size:
        return {**result, "mae_m": None, "rmse_m": None, "abs_rel": None}
    result.update(
        mae_m=float(absolute.mean()),
        rmse_m=float(np.sqrt((difference**2).mean())),
        abs_rel=float((absolute / truth_z_m[valid]).mean()),
        absolute_error_p50_m=float(np.quantile(absolute, 0.5)),
        absolute_error_p90_m=float(np.quantile(absolute, 0.9)),
        absolute_error_p95_m=float(np.quantile(absolute, 0.95)),
    )
    return result
