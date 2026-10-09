"""Predeclared baseline and matching parameters, never selected from test outcomes."""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sightguard.risk.schema import RiskLevel


@dataclass(frozen=True)
class MatchingConfig:
    minimum_iou: float = 0.10
    important_severities: tuple[str, ...] = ("MEDIUM", "HIGH")

    def __post_init__(self) -> None:
        if (
            type(self.minimum_iou) not in (int, float)
            or not math.isfinite(self.minimum_iou)
            or not 0 <= self.minimum_iou <= 1
        ):
            raise ValueError("minimum_iou must be within [0,1]")
        if not self.important_severities:
            raise ValueError("important_severities must not be empty")
        object.__setattr__(
            self,
            "important_severities",
            tuple(RiskLevel(v).value for v in self.important_severities),
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    schema_version: str = "3.0"
    baseline_classes: tuple[int, ...] = (0, 1, 3)
    baseline_focus_threshold: float = 0.8
    baseline_severity: str = "MEDIUM"
    fixed_risk_threshold: float = 0.35
    fixed_cooldown_ms: float = 3000
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    generate_figures: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != "3.0":
            raise ValueError("Unsupported benchmark config schema")
        if not self.baseline_classes or any(
            type(v) is not int or v < 0 for v in self.baseline_classes
        ):
            raise ValueError("baseline_classes must be non-empty nonnegative class IDs")
        object.__setattr__(self, "baseline_classes", tuple(self.baseline_classes))
        RiskLevel(self.baseline_severity)
        for name in ("baseline_focus_threshold", "fixed_risk_threshold", "fixed_cooldown_ms"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
            if name != "fixed_cooldown_ms" and value > 1:
                raise ValueError(f"Invalid {name}")
        if type(self.generate_figures) is not bool:
            raise ValueError("generate_figures must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_benchmark_config(path: Path | None = None) -> BenchmarkConfig:
    if path is None:
        return BenchmarkConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Benchmark config must be a mapping")
    try:
        if "matching" in raw:
            raw["matching"] = MatchingConfig(**raw["matching"])
        return BenchmarkConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"Invalid benchmark config: {exc}") from exc
