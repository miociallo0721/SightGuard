"""Versioned neutral motion and past-window contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from sightguard.data import FrameRecord
from sightguard.scene.spatial_grid.schema import FrameKey


@dataclass(frozen=True)
class MotionConfig:
    schema_version: str = "sightguard.motion-config.v1"
    resize_width: int = 640
    max_features: int = 500
    feature_quality: float = 0.01
    feature_min_distance: float = 8.0
    lk_window: int = 21
    lk_levels: int = 3
    lk_iterations: int = 30
    lk_epsilon: float = 0.01
    lk_max_error: float = 30.0
    fb_max_error: float = 1.5
    detection_confidence: float = 0.5
    mask_margin_pixels: float = 12.0
    min_background_fraction: float = 0.2
    min_features: int = 40
    min_matches: int = 30
    preferred_matches: int = 80
    ransac_threshold_pixels: float = 3.0
    ransac_iterations: int = 2000
    ransac_confidence: float = 0.99
    rng_seed: int = 41
    min_inlier_ratio: float = 0.35
    preferred_inlier_ratio: float = 0.6
    min_feature_coverage: float = 0.25
    preferred_feature_coverage: float = 0.5
    residual_p95_limit: float = 4.0
    residual_p95_preferred: float = 2.0
    blur_unknown_variance: float = 10.0
    blur_limited_variance: float = 40.0
    min_dt_ms: float = 10.0
    max_dt_ms: float = 120.0
    stable_diagonals_per_second: float = 0.08
    rapid_diagonals_per_second: float = 0.5
    discontinuity_rate_delta: float = 0.75
    max_frame_displacement_fraction: float = 0.25
    max_rotation_degrees_per_frame: float = 20.0
    min_scale: float = 0.85
    max_scale: float = 1.15
    window_ms: float = 500.0
    min_window_valid_fraction: float = 0.6
    min_window_time_coverage: float = 0.8
    max_window_gap_ms: float = 120.0
    fresh_stable_ms: float = 500.0
    fresh_moderate_ms: float = 250.0
    fresh_rapid_ms: float = 100.0
    evidence_change_threshold: float = 0.2
    close_cost_margin: float = 0.03
    simulated_consumer_delays_ms: tuple[float, ...] = (0.0, 100.0, 250.0, 500.0, 1000.0)
    temporal_shadow_enabled: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != "sightguard.motion-config.v1":
            raise ValueError("unsupported motion configuration")
        integer_fields = {
            "resize_width",
            "max_features",
            "lk_window",
            "lk_levels",
            "lk_iterations",
            "min_features",
            "min_matches",
            "preferred_matches",
            "ransac_iterations",
            "rng_seed",
        }
        for name, value in asdict(self).items():
            if name in ("schema_version", "simulated_consumer_delays_ms"):
                continue
            if name == "temporal_shadow_enabled":
                if type(value) is not bool or value:
                    raise ValueError(
                        "temporal stabilization is not implemented; must stay disabled"
                    )
            elif name in integer_fields:
                if type(value) is not int or value < 1:
                    raise ValueError(f"invalid integer {name}")
            elif isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid positive parameter {name}")
        for name in (
            "feature_quality",
            "detection_confidence",
            "min_background_fraction",
            "ransac_confidence",
            "min_inlier_ratio",
            "preferred_inlier_ratio",
            "min_feature_coverage",
            "preferred_feature_coverage",
            "min_window_valid_fraction",
            "min_window_time_coverage",
        ):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} exceeds one")
        if (
            self.resize_width < 32
            or self.max_features < self.preferred_matches
            or self.min_matches < 3
            or self.preferred_matches < self.min_matches
            or self.min_features > self.max_features
            or self.lk_window % 2 != 1
            or self.min_dt_ms >= self.max_dt_ms
            or self.min_scale >= 1
            or self.max_scale <= 1
            or self.stable_diagonals_per_second >= self.rapid_diagonals_per_second
            or self.min_inlier_ratio > self.preferred_inlier_ratio
            or self.min_feature_coverage > self.preferred_feature_coverage
            or self.blur_unknown_variance > self.blur_limited_variance
        ):
            raise ValueError("inconsistent motion thresholds")
        if (
            not self.simulated_consumer_delays_ms
            or len(set(self.simulated_consumer_delays_ms)) != len(self.simulated_consumer_delays_ms)
            or any(
                isinstance(x, bool) or not math.isfinite(x) or x < 0
                for x in self.simulated_consumer_delays_ms
            )
        ):
            raise ValueError("invalid simulated consumer delays")


@dataclass(frozen=True)
class MotionFrame:
    key: FrameKey
    stream_id: str
    image: np.ndarray | None
    tracks: FrameRecord | None
    timestamp_source: str = "video_pts"
    reasons: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MotionQuality:
    label: str
    reasons: tuple[str, ...]
    absolute_camera_pose_status: str = "UNKNOWN"
    scope: str = "engineering_image_motion_not_body_motion_or_absolute_pose"


@dataclass(frozen=True)
class FrameMotionEstimate:
    key: FrameKey
    stream_id: str
    previous_key: FrameKey | None
    dt_ms: float | None
    masking_mode: str
    estimation_status: str
    quality: MotionQuality
    feature_count: int
    match_count: int
    inlier_count: int
    inlier_ratio: float | None
    background_available_fraction: float | None
    background_feature_coverage: float | None
    transform_previous_to_current: tuple[tuple[float, ...], ...] | None
    translation_source_pixels: tuple[float, float] | None
    rotation_2d_degrees: float | None
    scale_2d_ratio: float | None
    median_displacement_source_pixels: float | None
    motion_diagonals_per_second: float | None
    residual_inlier_median_pixels: float | None
    residual_inlier_p95_pixels: float | None
    residual_all_p95_pixels: float | None
    blur_variance: float | None
    flow_samples: tuple[dict[str, Any], ...]
    algorithm: str
    config: dict[str, Any]
    provenance: dict[str, Any]
    schema_version: str = "sightguard.frame-motion.v1"
    absolute_camera_pose_status: str = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotionWindowSummary:
    key: FrameKey
    stream_id: str
    start_ms: float
    end_ms: float
    observed_start_ms: float | None
    observed_end_ms: float | None
    sample_count: int
    valid_count: int
    valid_fraction: float | None
    time_coverage_fraction: float
    maximum_gap_ms: float | None
    motion_rate_p50: float | None
    motion_rate_p95: float | None
    labels: dict[str, int]
    source_frame_indices: tuple[int, ...]
    quality: MotionQuality
    source_references: tuple[str, ...]
    schema_version: str = "sightguard.motion-window.v1"
    absolute_camera_pose_status: str = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
