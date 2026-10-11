"""Strict versioned audio configuration; no perception or decision thresholds."""

import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SpatialConfig:
    schema_version: str = "1.0"
    library_path: str | None = None
    device_name: str | None = None
    request_hrtf: bool = True
    gain: float = 0.12
    queue_capacity: int = 16
    risk_ttl_ms: float = 900
    navigation_ttl_ms: float = 4000
    max_direction_age_ms: float = 750
    horizontal_fov_deg: float | None = None
    approximate_half_angle_deg: float = 45
    virtual_radius: float = 1
    system_priority: int = 100
    high_risk_priority: int = 80
    risk_priority: int = 60
    navigation_priority: int = 20
    poll_interval_ms: float = 10
    join_timeout_s: float = 3

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or type(self.request_hrtf) is not bool:
            raise ValueError("Invalid spatial version or HRTF option")
        for name in ("library_path", "device_name"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"Invalid {name}")
        for item in fields(self):
            name, value = item.name, getattr(self, item.name)
            if name in {
                "schema_version",
                "library_path",
                "device_name",
                "request_hrtf",
                "horizontal_fov_deg",
            }:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid numeric {name}")
            if name.endswith("priority"):
                if type(value) is not int or not 0 <= value <= 100:
                    raise ValueError(f"Invalid priority {name}")
            elif name == "gain":
                if not 0 <= value <= 1:
                    raise ValueError("gain must be in 0..1")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")
        if type(self.queue_capacity) is not int or self.queue_capacity > 256:
            raise ValueError("queue_capacity must be an integer in 1..256")
        if self.approximate_half_angle_deg > 90:
            raise ValueError("Approximate half angle cannot exceed 90")
        if self.horizontal_fov_deg is not None and (
            isinstance(self.horizontal_fov_deg, bool)
            or not isinstance(self.horizontal_fov_deg, (int, float))
            or not math.isfinite(self.horizontal_fov_deg)
            or not 0 < self.horizontal_fov_deg < 180
        ):
            raise ValueError("Invalid horizontal FOV")


def load_spatial_config(path: Path | None = None) -> SpatialConfig:
    if path is None:
        return SpatialConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Spatial configuration must be a mapping")
    return SpatialConfig(**raw)
