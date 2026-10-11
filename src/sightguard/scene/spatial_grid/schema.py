"""Contracts for exact-frame image evidence, never metric occupancy."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = "sightguard.spatial-grid.v1"
CONFIG_VERSION = "sightguard.spatial-grid-config.v1"
GROUND_CLASSES = {3: "floor", 6: "road", 11: "sidewalk"}


@dataclass(frozen=True)
class SpatialGridConfig:
    schema_version: str = CONFIG_VERSION
    rows: int = 15
    cols: int = 15
    min_pixel_confidence: float = 0.35
    ground_fraction_threshold: float = 0.25
    min_detection_confidence: float = 0.35
    occupancy_coverage_threshold: float = 0.25
    depth_spread_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_VERSION:
            raise ValueError("unsupported grid configuration")
        for value in (self.rows, self.cols):
            if type(value) is not int or not 1 <= value <= 128:
                raise ValueError("grid dimensions must be integers in [1, 128]")
        for name in (
            "min_pixel_confidence",
            "ground_fraction_threshold",
            "min_detection_confidence",
            "occupancy_coverage_threshold",
            "depth_spread_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite in (0, 1]")


@dataclass(frozen=True)
class FrameKey:
    video_id: str
    frame_index: int
    timestamp_ms: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.video_id or type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("invalid frame identity")
        if not math.isfinite(self.timestamp_ms) or self.timestamp_ms < 0:
            raise ValueError("invalid video timestamp")
        if any(type(x) is not int or x < 1 for x in (self.width, self.height)):
            raise ValueError("invalid source dimensions")


@dataclass(frozen=True)
class EvidenceSource:
    key: FrameKey
    status: str
    quality: str = "limited"
    reference: str | None = None
    sha256: str | None = None
    schema_version: str | None = None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpatialGridInput:
    key: FrameKey
    semantic_source: EvidenceSource | None = None
    depth_source: EvidenceSource | None = None
    detection_source: EvidenceSource | None = None
    class_map: np.ndarray | None = None
    confidence_map: np.ndarray | None = None
    candidate_components: np.ndarray | None = None
    depth_map: np.ndarray | None = None
    detections: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class DepthEvidence:
    valid_count: int
    invalid_fraction: float
    negative_fraction: float
    nonfinite_fraction: float
    median: float | None
    q10: float | None
    q90: float | None
    frame_percentile: float | None
    spread_fraction: float | None
    quality: str


@dataclass(frozen=True)
class GroundEvidence:
    status: str
    support_fraction: float | None
    class_fractions: dict[str, float]
    mean_confidence: float | None
    candidate_fraction: float | None
    bottom_connected_component_ids: tuple[int, ...]
    scene_quality: str


@dataclass(frozen=True)
class BboxCoverage:
    detection_index: int
    track_id: int | None
    class_id: int
    class_name: str | None
    confidence: float
    cell_fraction: float
    bbox_fraction: float
    candidate: bool
    quality: str


@dataclass(frozen=True)
class SpatialGridCell:
    cell_id: int
    row: int
    col: int
    bounds_xyxy: tuple[int, int, int, int]
    state: str
    quality: str
    ground: GroundEvidence
    depth: DepthEvidence
    detection_coverage: tuple[BboxCoverage, ...]
    source_status: dict[str, str]
    conflict_reasons: tuple[str, ...]
    unknown_reasons: tuple[str, ...]
    provenance: dict[str, str | None]


@dataclass(frozen=True)
class ObjectGridRelation:
    detection_index: int
    track_id: int | None
    class_id: int | None
    confidence: float | None
    bbox_xyxy: tuple[float, ...] | None
    clipped_bbox_xyxy: tuple[float, ...] | None
    cell_coverage: tuple[dict[str, float | int], ...]
    depth: DepthEvidence
    ground_fraction: float | None
    candidate_component_ids: tuple[int, ...]
    image_bearing: str | None
    status: str
    quality: str
    reasons: tuple[str, ...]
    detection_reference: str | None


@dataclass(frozen=True)
class SpatialGridFrame:
    key: FrameKey
    rows: int
    cols: int
    status: str
    sources: dict[str, EvidenceSource | None]
    source_status: dict[str, str]
    cells: tuple[SpatialGridCell, ...]
    object_relations: tuple[ObjectGridRelation, ...]
    failures: tuple[str, ...]
    frame_depth_quantiles: tuple[float, ...]
    schema_version: str = SCHEMA_VERSION
    coordinate_space: str = "source_image_pixels_half_open"
    depth_semantics: str = "relative_inverse_depth_larger_is_nearer"
    production_decision_changed: bool = False
    limitations: tuple[str, ...] = field(
        default=(
            "not_metric_occupancy_or_safe_path",
            "bbox_contains_background_not_instance_mask",
            "camera_pose_unobserved",
            "no_cross_frame_depth_scale_or_tracking_of_cells",
        )
    )

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible result without source arrays or execution-time fields."""
        return asdict(self)
