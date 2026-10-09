"""Strict, versioned Phase 1 configuration (no inference imports)."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PerceptionConfig:
    schema_version: str = "1.0"
    model_path: str = "models/yolo26n.pt"
    confidence: float = 0.25
    device: str = "cpu"
    image_size: int = 640
    class_ids: tuple[int, ...] | None = (0, 1, 3)
    tracker: str = "botsort.yaml"
    log_every_frames: int = 30

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("Unsupported config schema_version")
        if not isinstance(self.model_path, str) or not self.model_path:
            raise ValueError("model_path must be a non-empty string")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        if not 0 < self.confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        for name in ("image_size", "log_every_frames"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a string, e.g. cpu or 0")
        if self.tracker != "botsort.yaml":
            raise ValueError("Phase 1 requires the bundled botsort.yaml tracker")
        if self.class_ids is not None:
            if not isinstance(self.class_ids, (tuple, list)) or not self.class_ids:
                raise ValueError("class_ids must be a non-empty sequence or null")
            if any(type(item) is not int or item < 0 for item in self.class_ids):
                raise ValueError("class_ids must contain non-negative integers")
            object.__setattr__(self, "class_ids", tuple(self.class_ids))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path | None = None) -> PerceptionConfig:
    """Relative model paths are resolved from the command's working directory."""
    if path is None:
        return PerceptionConfig()
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    try:
        return PerceptionConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"Invalid configuration keys: {exc}") from exc
