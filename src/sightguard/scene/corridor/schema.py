"""Serializable observation bands and evidence costs, never navigation commands."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from sightguard.scene.spatial_grid.schema import FrameKey

SCHEMA_VERSION = "sightguard.corridor-assessment.v1"
CONFIG_VERSION = "sightguard.corridor-config.v1"


@dataclass(frozen=True)
class CandidateConfig:
    names: tuple[str, ...] = ("IMAGE_LEFT", "IMAGE_CENTER", "IMAGE_RIGHT")
    top_centers: tuple[float, ...] = (0.2, 0.5, 0.8)
    bottom_centers: tuple[float, ...] = (0.35, 0.5, 0.65)
    start_row_fraction: float = 0.4
    top_width_fraction: float = 0.2
    bottom_width_fraction: float = 1 / 3
    near_row_fraction: float = 0.8
    row_weight_gain: float = 1.0

    def __post_init__(self) -> None:
        if (
            not 1 <= len(self.names) <= 9
            or len(set(self.names)) != len(self.names)
            or len(self.top_centers) != len(self.names)
            or len(self.bottom_centers) != len(self.names)
            or any(not isinstance(x, str) or not x.startswith("IMAGE_") for x in self.names)
        ):
            raise ValueError("candidate names/centers must be unique and aligned")
        for value in (
            *self.top_centers,
            *self.bottom_centers,
            self.start_row_fraction,
            self.top_width_fraction,
            self.bottom_width_fraction,
            self.near_row_fraction,
        ):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError("candidate fractions must be finite in (0,1)")
        if self.near_row_fraction <= self.start_row_fraction:
            raise ValueError("near rows must be below the first candidate row")
        if (
            isinstance(self.row_weight_gain, bool)
            or not math.isfinite(self.row_weight_gain)
            or not 0 <= self.row_weight_gain <= 5
        ):
            raise ValueError("invalid row attention gain")


@dataclass(frozen=True)
class EvaluationConfig:
    min_ground_support: float = 0.2
    min_bottom_ground: float = 0.15
    min_continuity: float = 0.2
    min_depth_coverage: float = 0.8
    max_unknown: float = 0.45
    max_conflict: float = 0.4
    max_near_bbox_coverage: float = 0.65
    min_scene_confidence: float = 0.45
    require_observed_orientation: bool = True
    tie_margin: float = 0.03
    depth_near_fraction: float = 0.75
    road_support_factor: float = 0.5
    floor_support_factor: float = 0.8
    sidewalk_support_factor: float = 1.0
    ground_weight: float = 0.25
    continuity_weight: float = 0.2
    obstacle_weight: float = 0.2
    depth_weight: float = 0.15
    unknown_weight: float = 0.15
    conflict_weight: float = 0.05

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "require_observed_orientation":
                if type(value) is not bool:
                    raise ValueError("orientation gate must be boolean")
            elif isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite in [0,1]")
        weights = [value for name, value in asdict(self).items() if name.endswith("_weight")]
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("explicit cost weights must sum to one")
        if self.min_depth_coverage <= 0 or self.min_ground_support <= 0:
            raise ValueError("evidence minima must be positive")


@dataclass(frozen=True)
class CorridorConfig:
    candidates: CandidateConfig = CandidateConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    schema_version: str = CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_VERSION:
            raise ValueError("unsupported corridor config")


@dataclass(frozen=True)
class CandidateCorridor:
    candidate_id: str
    key: FrameKey
    cell_ids: tuple[int, ...]
    row_bounds_xyxy: tuple[tuple[int, int, int, int], ...]
    polygon_xy: tuple[tuple[int, int], ...]
    near_cell_ids: tuple[int, ...]
    geometry: dict[str, Any]
    generation_status: str = "GEOMETRY_TEMPLATE"
    origin: str = "deterministic_image_grid_band"
    reasons: tuple[str, ...] = ("not_world_route", "lower_image_is_attention_not_metric_near_field")


@dataclass(frozen=True)
class ObjectCorridorRelation:
    key: FrameKey
    candidate_id: str
    detection_index: int
    track_id: int | None
    class_id: int | None
    class_name: str | None
    confidence: float | None
    cell_ids: tuple[int, ...]
    clipped_bbox_fraction: float
    candidate_bbox_proxy_fraction: float
    near_bbox_proxy_fraction: float
    ground_fraction: float | None
    relative_depth_percentile: float | None
    background_depth_percentile: float | None
    relative_depth_contrast: float | None
    quality: str
    reasons: tuple[str, ...]
    detection_reference: str | None
    event_reference: str | None = None


@dataclass(frozen=True)
class CorridorEvidence:
    ground_support: float
    class_fractions: dict[str, float]
    class_adjusted_ground_support: float
    bottom_ground_support: float
    ground_continuity: float
    interrupted_rows: tuple[int, ...]
    unknown_exposure: float
    ground_unknown_exposure: float
    bbox_proxy_coverage: float
    near_bbox_proxy_coverage: float
    near_relative_depth_proxy: float | None
    relative_depth_variation: float | None
    depth_coverage: float
    conflict_coverage: float
    semantic_confidence: float
    source_status: dict[str, str]
    camera_orientation: str
    quality_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CandidateAssessment:
    candidate: CandidateCorridor
    evidence: CorridorEvidence
    evaluation_status: str
    eligibility_reasons: tuple[str, ...]
    cost_components: dict[str, float | None]
    weighted_contributions: dict[str, float | None]
    total_cost: float | None
    missing_components: tuple[str, ...]
    object_relations: tuple[ObjectCorridorRelation, ...]


@dataclass(frozen=True)
class CorridorAssessment:
    key: FrameKey
    ablation: str
    candidates: tuple[CandidateAssessment, ...]
    status: str
    best_supported_candidate: str | None
    tied_candidates: tuple[str, ...]
    ranking: tuple[str, ...]
    abstain_reasons: tuple[str, ...]
    provenance: dict[str, Any]
    schema_version: str = SCHEMA_VERSION
    production_decision_changed: bool = False
    coordinate_space: str = "source_image_pixels_not_world_corridors"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
