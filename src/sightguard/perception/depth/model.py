"""Local-only HF Depth Anything V2 Small adapter with explicit lifecycle."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from sightguard.perception.depth.artifacts import sha256_file
from sightguard.perception.depth.schema import DepthFrameResult, DepthModelConfig


class DepthModelUnavailableError(RuntimeError):
    """Missing, altered, incompatible or unavailable local model/runtime."""


class InvalidDepthOutputError(ValueError):
    """Model returned an unusable spatial output."""


@dataclass(frozen=True)
class DepthPrediction:
    result: DepthFrameResult
    depth_map: np.ndarray | None = None
    model_output: np.ndarray | None = None


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Validate uint8 OpenCV pixels and make an independent contiguous RGB array."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
        or min(frame.shape[:2]) < 1
        or frame.dtype != np.uint8
    ):
        raise ValueError("expected non-empty uint8 HxWx3 BGR image")
    return frame[:, :, ::-1].copy()


def check_depth(
    depth: np.ndarray, shape: tuple[int, int], min_relative_spread: float
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Engineering checks only, not calibrated prediction reliability."""
    if not isinstance(depth, np.ndarray) or depth.ndim != 2 or depth.shape != shape:
        return ("invalid_output_size",), {}
    if not np.isfinite(depth).all():
        return ("non_finite_output",), {}
    lo, hi = np.percentile(depth, [1, 99])
    spread = float((hi - lo) / max(float(np.max(np.abs(depth))), 1e-12))
    stats = {
        "min": float(depth.min()),
        "max": float(depth.max()),
        "mean": float(depth.mean()),
        "p01": float(lo),
        "p99": float(hi),
        "relative_p01_p99_spread": spread,
    }
    return (("insufficient_output_variation",) if spread < min_relative_spread else ()), stats


class DepthEstimator:
    """Load once, infer independently on each frame, explicitly release resources."""

    def __init__(self, config: DepthModelConfig) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None
        self.device_name = "unavailable"
        self.metadata: dict[str, Any] = {}
        self.load_error: str | None = None
        self._fallback_used = False

    def initialize(self) -> None:
        if self.model is not None:
            return
        started = time.perf_counter()
        try:
            for name, expected in (
                ("model.safetensors", self.config.checkpoint_sha256),
                ("config.json", self.config.config_sha256),
                ("preprocessor_config.json", self.config.processor_sha256),
            ):
                path = self.config.checkpoint_dir / name
                if not path.is_file():
                    raise DepthModelUnavailableError(f"Local model artifact missing: {path}")
                if sha256_file(path) != expected:
                    raise DepthModelUnavailableError(
                        f"Local model artifact SHA-256 mismatch: {path}"
                    )
            import torch
            from transformers import DepthAnythingForDepthEstimation, DPTImageProcessor

            self.torch = torch
            requested = self.config.device
            self.device_name = (
                ("cuda:0" if torch.cuda.is_available() else "cpu")
                if requested == "auto"
                else requested
            )
            if self.device_name.startswith("cuda") and not torch.cuda.is_available():
                raise DepthModelUnavailableError("CUDA requested but unavailable")
            self.processor = DPTImageProcessor.from_pretrained(
                self.config.checkpoint_dir, local_files_only=True, use_fast=False
            )
            self.model, loading = DepthAnythingForDepthEstimation.from_pretrained(
                self.config.checkpoint_dir, local_files_only=True, output_loading_info=True
            )
            if any(
                loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
            ):
                raise DepthModelUnavailableError(f"Checkpoint loading mismatch: {loading}")
            if self.model.config.depth_estimation_type != "relative":
                raise DepthModelUnavailableError("Metric checkpoint is outside Depth v1")
            self.model.eval()
            try:
                self.model.to(self.device_name)
                self._synchronize()
            except torch.cuda.OutOfMemoryError:
                if not self.config.allow_cpu_fallback:
                    raise
                self._move_to_cpu()
            self.load_error = None
            self.metadata = {
                "load_ms": (time.perf_counter() - started) * 1000,
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "device": self.device_name,
                "cuda_available": torch.cuda.is_available(),
                "torch_cuda": torch.version.cuda,
                "gpu_names": [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ],
                "load_cpu_fallback": self._fallback_used,
                "checkpoint_url": f"https://huggingface.co/{self.config.model_id}/resolve/"
                f"{self.config.model_revision}/model.safetensors",
                "loading_info": {
                    k: sorted(v) if isinstance(v, set) else v for k, v in loading.items()
                },
            }
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.close()
            raise DepthModelUnavailableError(self.load_error) from exc

    def _synchronize(self) -> None:
        if self.device_name.startswith("cuda"):
            self.torch.cuda.synchronize(self.device_name)

    def _move_to_cpu(self) -> None:
        self.model.to("cpu")
        self.device_name = "cpu"
        self._fallback_used = True
        gc.collect()
        self.torch.cuda.empty_cache()

    def infer(
        self,
        frame_bgr: np.ndarray | None,
        *,
        video_id: str,
        frame_index: int,
        timestamp_ms: float,
        source_width: int,
        source_height: int,
        timestamp_source: str = "unknown",
        stream_id: str | None = None,
    ) -> DepthPrediction:
        result = DepthFrameResult(
            video_id,
            frame_index,
            timestamp_ms,
            source_width,
            source_height,
            timestamp_source=timestamp_source,
            stream_id=stream_id,
            device=self.device_name,
        )
        try:
            rgb = bgr_to_rgb(frame_bgr)
            if rgb.shape[:2] != (source_height, source_width):
                raise ValueError("input size differs from source context")
        except ValueError as exc:
            return DepthPrediction(
                replace(
                    result,
                    inference_status="image_read_failed",
                    quality_reasons=("invalid_input",),
                    error=str(exc),
                )
            )
        if self.model is None:
            return DepthPrediction(
                replace(
                    result,
                    quality_reasons=("model_unavailable",),
                    error=self.load_error,
                )
            )
        notes: tuple[str, ...] = ()
        try:
            native, aligned, input_shape, timings = self._predict(rgb)
        except InvalidDepthOutputError as exc:
            return DepthPrediction(
                replace(
                    result,
                    inference_status="completed",
                    quality_reasons=("invalid_output_size",),
                    error=str(exc),
                )
            )
        except self.torch.cuda.OutOfMemoryError as exc:
            if (
                self.device_name.startswith("cuda")
                and self.config.allow_cpu_fallback
                and not self._fallback_used
            ):
                try:
                    self._move_to_cpu()
                    notes = ("cuda_oom_one_cpu_retry",)
                    native, aligned, input_shape, timings = self._predict(rgb)
                except Exception as fallback_exc:
                    return self._failed(result, fallback_exc, "cpu_fallback_failed")
            else:
                return self._failed(result, exc, "cuda_oom_no_retry")
        except Exception as exc:
            return self._failed(result, exc, "inference_failed")
        native_reasons, _ = check_depth(native, input_shape, self.config.min_relative_spread)
        reasons, stats = check_depth(
            aligned, (source_height, source_width), self.config.min_relative_spread
        )
        reasons = tuple(dict.fromkeys((*native_reasons, *reasons)))
        result = replace(
            result,
            device=self.device_name,
            status="unknown" if reasons else "valid",
            inference_status="completed",
            quality_reasons=reasons,
            input_shape=input_shape,
            depth_map_shape=aligned.shape,
            model_output_shape=native.shape,
            timings_ms=timings,
            statistics=stats,
            runtime_notes=notes,
        )
        # Finite constant outputs are retained for diagnosis, but remain UNKNOWN.
        if "non_finite_output" in reasons or "invalid_output_size" in reasons:
            return DepthPrediction(result)
        return DepthPrediction(result, aligned, native)

    def _failed(self, result: DepthFrameResult, exc: Exception, reason: str) -> DepthPrediction:
        return DepthPrediction(
            replace(
                result,
                device=self.device_name,
                inference_status="inference_failed",
                quality_reasons=(reason,),
                error=f"{type(exc).__name__}: {exc}",
            )
        )

    def _predict(
        self, rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int], dict[str, float]]:
        self._synchronize()
        started = time.perf_counter()
        inputs = self.processor(
            images=rgb,
            size={"height": self.config.input_size, "width": self.config.input_size},
            return_tensors="pt",
        ).to(self.device_name)
        input_shape = tuple(int(v) for v in inputs["pixel_values"].shape[-2:])
        self._synchronize()
        pre_end = time.perf_counter()
        with self.torch.inference_mode():
            native_tensor = self.model(**inputs).predicted_depth
            self._synchronize()
            forward_end = time.perf_counter()
            if native_tensor.ndim != 3 or tuple(native_tensor.shape) != (1, *input_shape):
                raise InvalidDepthOutputError(
                    f"invalid native output dimensions: {native_tensor.shape}"
                )
            native = native_tensor[0].to("cpu", dtype=self.torch.float32).numpy().copy()
            aligned_tensor = self.torch.nn.functional.interpolate(
                native_tensor[:, None],
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            )[0, 0]
            aligned = aligned_tensor.to("cpu", dtype=self.torch.float32).numpy().copy()
            self._synchronize()
        ended = time.perf_counter()
        return (
            native,
            aligned,
            input_shape,
            {
                "preprocess": (pre_end - started) * 1000,
                "forward": (forward_end - pre_end) * 1000,
                "postprocess": (ended - forward_end) * 1000,
                "inference": (ended - started) * 1000,
            },
        )

    def resource_stats(self) -> dict[str, Any]:
        if self.torch is None or not self.torch.cuda.is_available():
            return {"cuda_available": False}
        cuda = self.torch.cuda
        return {
            "cuda_available": True,
            "allocated_mib": cuda.memory_allocated() / 2**20,
            "reserved_mib": cuda.memory_reserved() / 2**20,
            "peak_allocated_mib": cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": cuda.max_memory_reserved() / 2**20,
        }

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self.device_name = "unavailable"
