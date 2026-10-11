"""Lossless depth storage and separate image-space diagnostic visualization."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from sightguard.perception.depth.model import DepthPrediction
    from sightguard.perception.depth.schema import DepthFrameResult


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def load_depth_map(directory: Path, row: dict[str, Any]) -> np.ndarray:
    """Verify valid evidence and lossless storage before use by a future consumer."""
    from sightguard.perception.depth.schema import SCHEMA_VERSION, SEMANTICS

    if (
        row.get("schema_version") != SCHEMA_VERSION
        or row.get("status") != "valid"
        or row.get("depth_semantics") != SEMANTICS
    ):
        raise ValueError("unsupported or UNKNOWN depth evidence")
    reference = row.get("depth_data_reference")
    if not isinstance(reference, str) or Path(reference).is_absolute():
        raise ValueError("depth reference must be relative")
    root = directory.resolve()
    path = (root / reference).resolve()
    if not path.is_relative_to(root):
        raise ValueError("depth reference escapes its video directory")
    if sha256_file(path) != row.get("depth_data_sha256"):
        raise ValueError("depth artifact SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as data:
        depth = data["depth"]
    expected = (row["source_frame_height"], row["source_frame_width"])
    if depth.shape != expected or depth.dtype != np.float32 or not np.isfinite(depth).all():
        raise ValueError("invalid stored depth dimensions/dtype/values")
    return depth


def visualize_depth(frame_bgr: np.ndarray, depth: np.ndarray | None, label: str) -> np.ndarray:
    """Per-frame min/max display normalization only; never mutate predictions."""
    height, width = frame_bgr.shape[:2]
    if depth is None:
        colored = np.zeros_like(frame_bgr)
    else:
        if depth.shape != (height, width) or not np.isfinite(depth).all():
            raise ValueError("cannot visualize non-finite or misaligned depth")
        spread = float(np.ptp(depth))
        normalized = (depth - depth.min()) / spread if spread > 0 else np.zeros_like(depth)
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    # Smaller preview saves disk while .npz retains full source resolution.
    output = cv2.resize(np.hstack([frame_bgr, colored]), (1280, round(height / width * 640)))
    header = np.zeros((80, 1280, 3), dtype=np.uint8)
    for y, text in (
        (22, label),
        (46, "RGB source | Relative inverse depth: larger / brighter = nearer"),
        (68, "Per-frame display scaling only; UNKNOWN is not safe; no meters or safe path"),
    ):
        cv2.putText(
            header,
            (text[:160]),
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([header, output])


def save_prediction(
    directory: Path, prediction: DepthPrediction, frame_bgr: np.ndarray
) -> DepthFrameResult:
    """Exclusive writes; each NPZ contains aligned values AND original model output."""
    result = prediction.result
    stem = f"{result.frame_index:06d}"
    data_ref = None
    data_hash = None
    started = time.perf_counter()
    if prediction.depth_map is not None:
        data_ref = f"depth/{stem}.npz"
        path = directory / data_ref
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(
                stream, depth=prediction.depth_map, model_output=prediction.model_output
            )
        data_hash = sha256_file(path)
    data_end = time.perf_counter()
    vis_ref = f"previews/{stem}.jpg"
    path = directory / vis_ref
    path.parent.mkdir(exist_ok=True)
    stat = result.statistics
    preview = visualize_depth(
        frame_bgr,
        prediction.depth_map,
        f"{result.video_id} frame={result.frame_index} t={result.timestamp_ms:.1f}ms "
        f"{result.status.upper()} source={result.source_frame_width}x{result.source_frame_height} "
        f"input={result.input_shape} min={stat.get('min', 0):.3f} max={stat.get('max', 0):.3f}",
    )
    ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise OSError("depth preview encoding failed")
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())
    return replace(
        result,
        depth_data_reference=data_ref,
        depth_data_sha256=data_hash,
        visualization_reference=vis_ref,
        timings_ms={
            **result.timings_ms,
            "save_depth": (data_end - started) * 1000,
            "save_preview": (time.perf_counter() - data_end) * 1000,
        },
    )
