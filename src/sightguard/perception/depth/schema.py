"""Serializable depth contract independent of Torch, Transformers, and UI."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
CHECKPOINT_SHA256 = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
CONFIG_SHA256 = "c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75"
PROCESSOR_SHA256 = "d41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84"
SCHEMA_VERSION = "sightguard.depth.v1"
SEMANTICS = "relative_inverse_depth_larger_is_nearer"


@dataclass(frozen=True)
class DepthModelConfig:
    """Pinned local Small checkpoint; import and inference never download files."""

    checkpoint_dir: Path = Path("models/depth_anything_v2_small")
    model_id: str = MODEL_ID
    model_revision: str = REVISION
    checkpoint_sha256: str = CHECKPOINT_SHA256
    config_sha256: str = CONFIG_SHA256
    processor_sha256: str = PROCESSOR_SHA256
    device: str = "auto"
    input_size: int = 280
    allow_cpu_fallback: bool = True
    min_relative_spread: float = 1e-5

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        if self.model_id != MODEL_ID or self.model_revision != REVISION:
            raise ValueError("Depth v1 supports only the pinned official Small revision")
        if (self.checkpoint_sha256, self.config_sha256, self.processor_sha256) != (
            CHECKPOINT_SHA256,
            CONFIG_SHA256,
            PROCESSOR_SHA256,
        ):
            raise ValueError("Depth v1 requires the pinned official artifact hashes")
        if self.device not in ("auto", "cpu") and not re.fullmatch(r"cuda(?::\d+)?", self.device):
            raise ValueError("device must be auto, cpu, cuda or cuda:N")
        if type(self.input_size) is not int or not 56 <= self.input_size <= 518:
            raise ValueError("input_size must be an integer in [56, 518]")
        if self.input_size % 14:
            raise ValueError("input_size must be a multiple of the ViT patch size 14")
        if type(self.allow_cpu_fallback) is not bool:
            raise ValueError("allow_cpu_fallback must be boolean")
        if not math.isfinite(self.min_relative_spread) or not 0 < self.min_relative_spread < 1:
            raise ValueError("min_relative_spread must be finite and in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["checkpoint_dir"] = str(self.checkpoint_dir)
        return value


@dataclass(frozen=True)
class DepthFrameResult:
    """One frame's evidence, never a metric distance or safety assessment.

    Status is valid/unknown; inference_status distinguishes successful but unusable
    evidence from model failure. References are relative to the video output folder.
    No calibrated confidence is available. Valid means engineering checks passed.
    """

    video_id: str
    frame_index: int
    timestamp_ms: float
    source_frame_width: int
    source_frame_height: int
    timestamp_source: str = "unknown"
    stream_id: str | None = None
    model_id: str = MODEL_ID
    model_revision: str = REVISION
    checkpoint_sha256: str = CHECKPOINT_SHA256
    device: str = "unavailable"
    status: str = "unknown"
    inference_status: str = "model_unavailable"
    quality_reasons: tuple[str, ...] = ()
    error: str | None = None
    input_shape: tuple[int, int] | None = None
    depth_map_shape: tuple[int, int] | None = None
    model_output_shape: tuple[int, int] | None = None
    depth_data_reference: str | None = None
    depth_data_sha256: str | None = None
    visualization_reference: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    statistics: dict[str, float] = field(default_factory=dict)
    runtime_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.video_id or type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("video_id and non-negative integer frame_index required")
        if not math.isfinite(self.timestamp_ms) or self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be finite and non-negative")
        if any(
            type(v) is not int or v <= 0
            for v in (self.source_frame_width, self.source_frame_height)
        ):
            raise ValueError("source dimensions must be positive integers")
        if self.status not in ("valid", "unknown"):
            raise ValueError("invalid depth status")
        if self.status == "valid" and self.inference_status != "completed":
            raise ValueError("valid evidence requires completed inference")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            **asdict(self),
            "depth_semantics": SEMANTICS,
            "depth_scale_type": "frame_relative_uncalibrated",
            "dtype": "float32",
            "storage_conversion": "lossless_npz_float32; no normalization",
            "spatial_restore": "bicubic_align_corners_false_to_source_hw",
            "inference_latency_ms": self.timings_ms.get("inference"),
            "confidence_status": "not_calibrated_or_provided",
            "temporal_validity": "exact_source_frame_only; no interpolation_or_reuse",
            "production_decision_changed": False,
        }

    def matches_frame(self, video_id: str, frame_index: int, timestamp_ms: float) -> bool:
        """Future consumers must explicitly match all keys; no freshness guessing."""
        return (
            self.status == "valid"
            and self.video_id == video_id
            and self.frame_index == frame_index
            and self.timestamp_ms == timestamp_ms
        )
