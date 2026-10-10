"""Explicit local runtimes: never download weights during a live session."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    python: str
    path: str
    model_id: str
    revision: str
    device: str = "cpu"
    precision: str = "fp32"
    backend: str = "pytorch"
    onnx_path: str | None = None
    license: str | None = None
    code_revision: str | None = None

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("Invalid auxiliary device")
        if self.precision not in {"fp32", "fp16", "int8"}:
            raise ValueError("Invalid model precision")
        if self.backend not in {"pytorch", "onnx"}:
            raise ValueError("Invalid model backend")
        if self.backend == "onnx" and (not self.onnx_path or self.device != "cpu"):
            raise ValueError("ONNX baseline requires an explicit file and CPU backend")


@dataclass(frozen=True)
class HybridConfig:
    models: dict[str, ModelConfig]
    frequency_hz: float = 1.0
    timeout_s: float = 8.0
    load_timeout_s: float = 90.0
    stale_after_ms: float = 2000.0
    max_objects: int = 8
    fusion: str = "conservative_max"
    output_dir: str = "outputs/hybrid"
    retain_images: bool = False
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or set(self.models) != {"laya", "vision"}:
            raise ValueError("Hybrid config requires schema 1.0 and laya/vision models")
        for value in (self.frequency_hz, self.timeout_s, self.load_timeout_s, self.stale_after_ms):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Hybrid timing parameters must be positive finite numbers")
        if self.frequency_hz > 3 or self.max_objects < 1 or self.max_objects > 64:
            raise ValueError("frequency_hz must be <=3 and max_objects within 1..64")
        if self.fusion not in {"conservative_max", "consensus"}:
            raise ValueError("Unknown experimental fusion policy")
        if type(self.retain_images) is not bool:
            raise ValueError("retain_images must be boolean")

    def to_dict(self) -> dict:
        return asdict(self)


def load_hybrid_config(path: Path, root: Path) -> HybridConfig:
    """Resolve configured local paths against the project, independent of process cwd."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Hybrid config must be a mapping")
    payload = dict(payload)
    models = {}
    for name, values in payload.pop("models", {}).items():
        values = dict(values)
        for key in ("python", "path", "onnx_path"):
            if values.get(key):
                values[key] = str((root / values[key]).resolve())
        models[name] = ModelConfig(**values)
    return HybridConfig(models=models, **payload)
