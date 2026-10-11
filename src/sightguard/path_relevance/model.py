"""SegFormer MIT-B0 ADE20K scene perception adapter.

The checkpoint is published by OpenMMLab in MMSegmentation format. This module
loads its SegFormer encoder through Transformers and adapts the MMSegmentation
projection head without importing the MMSegmentation runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SegmentationOutput:
    """Semantic output aligned to the source frame."""

    class_map: np.ndarray
    confidence_map: np.ndarray
    inference_latency_ms: float
    input_width: int
    input_height: int


class ModelUnavailableError(RuntimeError):
    """Raised when the optional runtime or local checkpoint is unavailable."""


class InvalidSegmentationOutputError(ValueError):
    """Raised when a model returns invalid or spatially misaligned output."""


class SegformerB0Perceptor:
    """Lightweight ADE20K semantic segmentation on CPU or CUDA."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str = "auto",
        input_width: int = 512,
    ) -> None:
        if input_width < 32 or input_width % 32:
            raise ValueError("input_width must be a positive multiple of 32")
        self.checkpoint_path = Path(checkpoint_path)
        self.input_width = input_width
        self.device_name = device
        if not self.checkpoint_path.is_file():
            raise ModelUnavailableError(f"SegFormer checkpoint not found: {self.checkpoint_path}")
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as functional
            from transformers import SegformerConfig, SegformerModel
        except ImportError as exc:
            raise ModelUnavailableError(
                "Path Relevance requires the optional Transformers runtime and PyTorch"
            ) from exc
        if device == "auto":
            resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        try:
            self.device = torch.device(resolved_device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise ModelUnavailableError(f"CUDA device requested but unavailable: {device}")
            self.torch = torch
            self.functional = functional
            config = SegformerConfig(
                num_labels=150,
                num_channels=3,
                hidden_sizes=[32, 64, 160, 256],
                depths=[2, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                sr_ratios=[8, 4, 2, 1],
                mlp_ratios=[4, 4, 4, 4],
                hidden_act="gelu",
                drop_path_rate=0.1,
                layer_norm_eps=1e-6,
                decoder_hidden_size=256,
            )
            self.encoder = SegformerModel(config).eval()
            state = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            state_dict = state.get("state_dict", state)
            encoder_weights, decoder_weights = _convert_checkpoint(state_dict, torch)
            self.encoder.load_state_dict(encoder_weights, strict=True)
            self.decoder = _MMSegmentationHead(nn, decoder_weights).eval()
            self.encoder.to(self.device)
            self.decoder.to(self.device)
            self.class_count = 150
        except ModelUnavailableError:
            raise
        except self.torch.cuda.OutOfMemoryError:
            # Preserve the OOM type so the offline runner can perform an explicit
            # CPU fallback. Other model errors remain UNKNOWN at the call site.
            self.torch.cuda.empty_cache()
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not initialize SegFormer on {resolved_device}: {type(exc).__name__}: {exc}"
            ) from exc
        self.device_name = str(self.device)

    def _predict_tensor(self, tensor: Any) -> Any:
        """Run the encoder and its checkpoint-compatible ADE20K decode head."""
        outputs = self.encoder(pixel_values=tensor, output_hidden_states=True, return_dict=True)
        return self.decoder(outputs.hidden_states)

    def predict(self, frame_bgr: np.ndarray) -> SegmentationOutput:
        """Segment an OpenCV BGR frame without altering its aspect ratio."""
        if (
            not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.shape[0] < 1
            or frame_bgr.shape[1] < 1
        ):
            raise ValueError("frame must be a non-empty HxWx3 BGR image")
        import cv2

        torch = self.torch
        source_height, source_width = frame_bgr.shape[:2]
        resized_height = max(
            32, int(round(source_height * self.input_width / source_width / 32)) * 32
        )
        resized = cv2.resize(
            frame_bgr,
            (self.input_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = ((rgb - mean) / std).transpose(2, 0, 1).copy()
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                logits = self._predict_tensor(tensor)
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=(resized_height, self.input_width),
                    mode="bilinear",
                    align_corners=False,
                )
                probabilities = logits.softmax(dim=1)
                confidence, class_map = probabilities.max(dim=1)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        except Exception as exc:
            raise RuntimeError(f"SegFormer inference failed: {type(exc).__name__}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        class_array = class_map[0].to("cpu", dtype=torch.uint8).numpy()
        confidence_array = confidence[0].to("cpu", dtype=torch.float32).numpy()
        if (
            class_array.shape != (resized_height, self.input_width)
            or confidence_array.shape != (resized_height, self.input_width)
            or not np.isfinite(confidence_array).all()
        ):
            raise InvalidSegmentationOutputError(
                "SegFormer produced a non-finite or misaligned output"
            )
        return SegmentationOutput(
            class_map=class_array,
            confidence_map=confidence_array,
            inference_latency_ms=latency_ms,
            input_width=self.input_width,
            input_height=resized_height,
        )


class _MMSegmentationHead:
    """Small ADE20K decode head with OpenMMLab checkpoint-compatible layers."""

    def __new__(cls, nn: Any, weights: dict[str, Any]) -> Any:
        torch = __import__("torch")
        functional = torch.nn.functional

        class Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                channels = (32, 64, 160, 256)
                self.projections = nn.ModuleList(
                    nn.Sequential(
                        nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
                        nn.BatchNorm2d(256, eps=1e-5),
                        nn.ReLU(inplace=False),
                    )
                    for in_channels in channels
                )
                self.fusion = nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1, bias=False),
                    nn.BatchNorm2d(256, eps=1e-5),
                    nn.ReLU(inplace=False),
                )
                self.classifier = nn.Conv2d(256, 150, kernel_size=1)
                self.load_mmseg_weights(weights)

            def load_mmseg_weights(self, checkpoint: dict[str, Any]) -> None:
                own = self.state_dict()
                converted: dict[str, Any] = {}
                for index in range(4):
                    prefix = f"decode_head.convs.{index}"
                    target = f"projections.{index}"
                    converted[f"{target}.0.weight"] = checkpoint[f"{prefix}.conv.weight"]
                    for suffix in (
                        "weight",
                        "bias",
                        "running_mean",
                        "running_var",
                        "num_batches_tracked",
                    ):
                        converted[f"{target}.1.{suffix}"] = checkpoint[f"{prefix}.bn.{suffix}"]
                converted["fusion.0.weight"] = checkpoint["decode_head.fusion_conv.conv.weight"]
                for suffix in (
                    "weight",
                    "bias",
                    "running_mean",
                    "running_var",
                    "num_batches_tracked",
                ):
                    converted[f"fusion.1.{suffix}"] = checkpoint[
                        f"decode_head.fusion_conv.bn.{suffix}"
                    ]
                converted["classifier.weight"] = checkpoint["decode_head.conv_seg.weight"]
                converted["classifier.bias"] = checkpoint["decode_head.conv_seg.bias"]
                if set(converted) != set(own):
                    raise ValueError(
                        f"Decode head mismatch: missing={sorted(set(own) - set(converted))}, "
                        f"unexpected={sorted(set(converted) - set(own))}"
                    )
                self.load_state_dict(converted, strict=True)

            def forward(self, features: tuple[Any, ...]) -> Any:
                target_size = features[0].shape[-2:]
                projected = [
                    functional.interpolate(
                        layer(feature),
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    for layer, feature in zip(self.projections, features, strict=True)
                ]
                return self.classifier(self.fusion(torch.cat(projected, dim=1)))

        return Head()


def _convert_checkpoint(state: dict[str, Any], torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert OpenMMLab MIT-B0 encoder keys and split fused Q/K/V weights."""
    encoder: dict[str, Any] = {}
    decoder = {key: value for key, value in state.items() if key.startswith("decode_head.")}
    for key, value in state.items():
        if not key.startswith("backbone.layers."):
            continue
        parts = key.split(".")
        stage = int(parts[2])
        if parts[3] == "0":
            name = {
                "projection": "patch_embeddings.proj",
                "norm": "patch_embeddings.layer_norm",
            }[parts[4]]
            encoder[f"stages.{stage}.{name}.{parts[5]}"] = value
        elif parts[3] == "1":
            block = int(parts[4])
            body = parts[5:]
            prefix = f"stages.{stage}.blocks.{block}."
            if body[0] == "norm1":
                encoder[f"{prefix}layernorm_before.{body[1]}"] = value
            elif body[0] == "norm2":
                encoder[f"{prefix}layernorm_after.{body[1]}"] = value
            elif body[0] == "attn":
                if body[1:3] == ["attn", "in_proj_weight"]:
                    query, key_weight, val = torch.chunk(value, 3, dim=0)
                    for label, tensor in zip(
                        ("q_proj", "k_proj", "v_proj"), (query, key_weight, val), strict=True
                    ):
                        encoder[f"{prefix}attention.{label}.weight"] = tensor
                elif body[1:3] == ["attn", "in_proj_bias"]:
                    query, key_bias, val = torch.chunk(value, 3, dim=0)
                    for label, tensor in zip(
                        ("q_proj", "k_proj", "v_proj"), (query, key_bias, val), strict=True
                    ):
                        encoder[f"{prefix}attention.{label}.bias"] = tensor
                elif body[1] == "attn":
                    encoder[f"{prefix}attention.o_proj.{body[3]}"] = value
                elif body[1] == "sr":
                    encoder[
                        f"{prefix}attention.sequence_reduction.sequence_reduction.{body[2]}"
                    ] = value
                elif body[1] == "norm":
                    encoder[f"{prefix}attention.sequence_reduction.layer_norm.{body[2]}"] = value
                else:
                    raise ValueError(f"Unknown checkpoint attention key: {key}")
            elif body[0] == "ffn":
                layer = int(body[2])
                if layer == 0:
                    target = "mlp.fc1"
                    value = value.squeeze(-1).squeeze(-1)
                elif layer == 1:
                    target = "mlp.dwconv.dwconv"
                elif layer == 4:
                    target = "mlp.fc2"
                    value = value.squeeze(-1).squeeze(-1)
                else:
                    raise ValueError(f"Unexpected checkpoint MLP layer: {key}")
                encoder[f"{prefix}{target}.{body[3]}"] = value
            else:
                raise ValueError(f"Unknown checkpoint block key: {key}")
        elif parts[3] == "2":
            encoder[f"stages.{stage}.layer_norm.{parts[4]}"] = value
        else:
            raise ValueError(f"Unknown checkpoint stage key: {key}")

    return encoder, decoder
