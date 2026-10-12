"""Official Metric3Dv2 image geometry, isolated from formal risk/feedback."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Metric3DGeometry:
    source_hw: tuple[int, int]
    input_hw: tuple[int, int]
    padding_tblr: tuple[int, int, int, int]
    scaled_focal_x: float


def prepare_metric3d(
    rgb: np.ndarray, focal_x: float, *, input_hw: tuple[int, int] = (616, 1064)
) -> tuple[np.ndarray, Metric3DGeometry]:
    """Preserve official RGB 0..255 normalization and scaled-focal contract."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Metric3D requires uint8 RGB HWC")
    if not math.isfinite(focal_x) or focal_x <= 0:
        raise ValueError("verified positive focal length required")
    height, width = rgb.shape[:2]
    target_h, target_w = input_hw
    if min(height, width, target_h, target_w) <= 0:
        raise ValueError("invalid image geometry")
    scale = min(target_h / height, target_w / width)
    resized = cv2.resize(
        rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR
    )
    ph, pw = target_h - resized.shape[0], target_w - resized.shape[1]
    padding = (ph // 2, ph - ph // 2, pw // 2, pw - pw // 2)
    canvas = cv2.copyMakeBorder(
        resized, *padding, cv2.BORDER_CONSTANT, value=(123.675, 116.28, 103.53)
    )
    values = canvas.astype(np.float32)
    values -= np.array([123.675, 116.28, 103.53], dtype=np.float32)
    values /= np.array([58.395, 57.12, 57.375], dtype=np.float32)
    geometry = Metric3DGeometry((height, width), input_hw, padding, focal_x * scale)
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None]), geometry


def restore_metric3d_depth(canonical_z: np.ndarray, geometry: Metric3DGeometry) -> np.ndarray:
    """Unpad, bilinearly restore, then decanonicalize predicted optical Z."""
    if canonical_z.shape != geometry.input_hw or not np.isfinite(canonical_z).all():
        raise ValueError("invalid canonical depth geometry or nonfinite output")
    top, bottom, left, right = geometry.padding_tblr
    height, width = canonical_z.shape
    crop = canonical_z[top : height - bottom, left : width - right]
    restored = cv2.resize(crop, geometry.source_hw[::-1], interpolation=cv2.INTER_LINEAR)
    return np.clip(restored * (geometry.scaled_focal_x / 1000.0), 0, 300).astype(np.float32)
