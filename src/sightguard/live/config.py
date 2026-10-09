"""Small live-specific configuration; risk parameters remain in RiskConfig."""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LiveConfig:
    camera: int = 0
    width: int = 640
    height: int = 480
    stale_after_ms: float = 1000
    alert_banner_ms: float = 2000
    failure_display_ms: float = 1200
    join_timeout_s: float = 10

    def __post_init__(self) -> None:
        if type(self.camera) is not int or self.camera < 0:
            raise ValueError("camera must be a non-negative integer")
        for name in ("width", "height"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("stale_after_ms", "alert_banner_ms", "failure_display_ms", "join_timeout_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def load_live_config(path: Path | None = None) -> LiveConfig:
    if path is None:
        return LiveConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Live config must be a mapping")
    try:
        return LiveConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"Invalid live configuration: {exc}") from exc


def resolve_device(device: str) -> str:
    """Use the existing Ultralytics device strings after live-only alias resolution."""
    if device == "auto":
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "0"
    if device.startswith("cuda:"):
        return device.removeprefix("cuda:")
    return device
