"""Strict nested YAML configuration for Phase 2, separate from Phase 1."""

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from sightguard.risk.schema import RiskLevel


def _number(name: str, value: float, minimum: float = 0, maximum: float | None = None) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"Invalid {name}: {value}")


@dataclass(frozen=True)
class FeatureConfig:
    focus_center_x: float = 0.5
    focus_top_y: float = 0.35
    focus_bottom_y: float = 1.0
    focus_top_half_width: float = 0.12
    focus_bottom_half_width: float = 0.45
    focus_soft_margin: float = 0.15
    history_window_ms: float = 1500
    motion_min_interval_ms: float = 10
    motion_max_interval_ms: float = 750
    motion_smoothing_tau_ms: float = 150
    quality_observation_target: int = 4
    fallback_time_quality: float = 0.8

    def __post_init__(self) -> None:
        for name in (
            "focus_center_x",
            "focus_top_y",
            "focus_bottom_y",
            "focus_top_half_width",
            "focus_bottom_half_width",
            "fallback_time_quality",
        ):
            _number(name, getattr(self, name), 0, 1)
        if self.focus_top_y >= self.focus_bottom_y:
            raise ValueError("Focus top must be above bottom")
        if not 0 < self.focus_top_half_width <= self.focus_bottom_half_width:
            raise ValueError("Focus widths must be positive and widen toward bottom")
        if self.focus_bottom_half_width > min(self.focus_center_x, 1 - self.focus_center_x):
            raise ValueError("Focus region must fit inside the image")
        for name in (
            "focus_soft_margin",
            "history_window_ms",
            "motion_min_interval_ms",
            "motion_max_interval_ms",
        ):
            _number(name, getattr(self, name), 1e-9)
        _number("motion_smoothing_tau_ms", self.motion_smoothing_tau_ms)
        if self.motion_min_interval_ms > self.motion_max_interval_ms:
            raise ValueError("Motion interval range is inverted")
        if type(self.quality_observation_target) is not int or self.quality_observation_target < 1:
            raise ValueError("quality_observation_target must be a positive integer")


@dataclass(frozen=True)
class ScoringConfig:
    spatial_weight: float = 0.45
    appearance_weight: float = 0.35
    motion_weight: float = 0.20
    area_full_score: float = 0.20
    bottom_start: float = 0.30
    bottom_full_score: float = 0.95
    appearance_area_weight: float = 0.70
    scale_rate_full_score: float = 0.50
    inward_rate_full_score: float = 0.25
    motion_scale_weight: float = 0.75
    medium_threshold: float = 0.35
    high_threshold: float = 0.65
    smoothing_tau_ms: float = 0
    upgrade_margin: float = 0.02
    downgrade_margin: float = 0.05
    downgrade_hold_ms: float = 400

    def __post_init__(self) -> None:
        for name in (
            "spatial_weight",
            "appearance_weight",
            "motion_weight",
            "appearance_area_weight",
            "motion_scale_weight",
            "medium_threshold",
            "high_threshold",
            "upgrade_margin",
            "downgrade_margin",
            "bottom_start",
            "bottom_full_score",
        ):
            _number(name, getattr(self, name), 0, 1)
        if not math.isclose(self.spatial_weight + self.appearance_weight + self.motion_weight, 1):
            raise ValueError("Scoring weights must sum to 1")
        if self.spatial_weight + self.appearance_weight <= 0:
            raise ValueError("Missing-motion scoring needs spatial/appearance weight")
        if not 0 < self.medium_threshold < self.high_threshold < 1:
            raise ValueError("Risk thresholds must be ordered within (0,1)")
        if self.bottom_start >= self.bottom_full_score:
            raise ValueError("Bottom normalization range is inverted")
        if self.high_threshold + self.upgrade_margin > 1:
            raise ValueError("Upgrade threshold exceeds 1")
        if self.downgrade_margin >= self.medium_threshold:
            raise ValueError("Downgrade margin prevents low-risk recovery")
        for name in ("area_full_score", "scale_rate_full_score", "inward_rate_full_score"):
            _number(name, getattr(self, name), 1e-9)
        for name in ("smoothing_tau_ms", "downgrade_hold_ms"):
            _number(name, getattr(self, name))


@dataclass(frozen=True)
class AssociationConfig:
    recovery_window_ms: float = 750
    track_max_center_distance: float = 0.35
    geometry_max_center_distance: float = 0.15
    geometry_min_iou: float = 0.10
    geometry_min_score: float = 0.35
    iou_weight: float = 0.60
    ambiguity_margin: float = 0.08
    confirmation_ms: float = 250
    confirmation_observations: int = 2
    confirmation_max_gap_ms: float = 500

    def __post_init__(self) -> None:
        for name in (
            "recovery_window_ms",
            "track_max_center_distance",
            "geometry_max_center_distance",
            "confirmation_max_gap_ms",
        ):
            _number(name, getattr(self, name), 1e-9)
        for name in ("geometry_min_iou", "geometry_min_score", "iou_weight", "ambiguity_margin"):
            _number(name, getattr(self, name), 0, 1)
        _number("confirmation_ms", self.confirmation_ms)
        if type(self.confirmation_observations) is not int or self.confirmation_observations < 2:
            raise ValueError("Ordinary events require at least two observations")


@dataclass(frozen=True)
class AlertConfig:
    minimum_level: str = "MEDIUM"
    ordinary_interval_ms: float = 3000
    escalation_interval_ms: float = 1000
    escalation_min_score_delta: float = 0.05
    repeat_enabled: bool = False
    max_alerts_per_frame: int = 1
    perception_confirm_ms: float = 500
    limited_quality_threshold: float = 0.30
    max_frame_gap_ms: float = 1000

    def __post_init__(self) -> None:
        RiskLevel(self.minimum_level)
        for name in ("ordinary_interval_ms", "escalation_interval_ms", "perception_confirm_ms"):
            _number(name, getattr(self, name))
        _number("max_frame_gap_ms", self.max_frame_gap_ms, 1e-9)
        for name in ("escalation_min_score_delta", "limited_quality_threshold"):
            _number(name, getattr(self, name), 0, 1)
        if type(self.repeat_enabled) is not bool:
            raise ValueError("repeat_enabled must be boolean")
        if type(self.max_alerts_per_frame) is not int or self.max_alerts_per_frame < 1:
            raise ValueError("max_alerts_per_frame must be positive integer")


@dataclass(frozen=True)
class RiskConfig:
    schema_version: str = "2.0"
    features: FeatureConfig = field(default_factory=FeatureConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    def __post_init__(self) -> None:
        if self.schema_version != "2.0":
            raise ValueError("Unsupported Phase 2 configuration schema")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_risk_config(path: Path | None = None) -> RiskConfig:
    if path is None:
        return RiskConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(RiskConfig)}:
        raise ValueError("Risk config must be a mapping with known keys")
    types = {
        "features": FeatureConfig,
        "scoring": ScoringConfig,
        "association": AssociationConfig,
        "alerts": AlertConfig,
    }
    try:
        for name, config_type in types.items():
            if name in raw:
                if not isinstance(raw[name], dict):
                    raise ValueError(f"{name} must be a mapping")
                raw[name] = config_type(**raw[name])
        return RiskConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"Invalid risk configuration: {exc}") from exc
